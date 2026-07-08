import copy
import json
import multiprocessing
import os
import random
import time


import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Palette images with Transparency")


DATA_DIR = r"C:/Users/32992/Desktop/spacemit_houseplant/house_plant"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VALID_DIR = os.path.join(DATA_DIR, "valid")
PLANT_NAME_FILE = os.path.join(DATA_DIR, "plant_to_name.json")

NUM_CLASSES = 42
BATCH_SIZE = 64
STAGE1_EPOCHS = 12
STAGE2_EPOCHS = 13
RANDOM_SEED = 42

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "best.pt")
CURVE_PATH = os.path.join(OUTPUT_DIR, "training_curves.png")
PREVIEW_PATH = os.path.join(OUTPUT_DIR, "prediction_preview.png")

NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_transforms():
    return {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
        ]),
        "valid": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
        ]),
    }


def build_dataloaders(batch_size, num_workers, device):
    data_transforms = build_transforms()
    image_datasets = {
        phase: datasets.ImageFolder(os.path.join(DATA_DIR, phase), data_transforms[phase])
        for phase in ["train", "valid"]
    }

    if len(image_datasets["train"].classes) != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} classes, got {len(image_datasets['train'].classes)}."
        )

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }

    dataloaders = {
        "train": torch.utils.data.DataLoader(
            image_datasets["train"],
            shuffle=True,
            **loader_kwargs,
        ),
        "valid": torch.utils.data.DataLoader(
            image_datasets["valid"],
            shuffle=False,
            **loader_kwargs,
        ),
    }
    dataset_sizes = {phase: len(image_datasets[phase]) for phase in ["train", "valid"]}
    return dataloaders, dataset_sizes, image_datasets["train"].classes, image_datasets["train"].class_to_idx


def freeze_backbone(model):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True


def unfreeze_layer4_and_fc(model):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True


def initialize_model(num_classes, use_pretrained=True):
    try:
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if use_pretrained else None
        model = models.resnet18(weights=weights)
    except AttributeError:
        model = models.resnet18(pretrained=use_pretrained)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model


def print_trainable_parameters(model):
    print("Params to learn:")
    trainable_count = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_count += param.numel()
            print(f"\t{name}")
    print(f"Trainable parameter count: {trainable_count:,}")


def format_lrs(optimizer):
    return ", ".join(f"{group['lr']:.7f}" for group in optimizer.param_groups)


def train_model(
    model,
    dataloaders,
    dataset_sizes,
    criterion,
    optimizer,
    scheduler,
    device,
    num_epochs,
    stage_name,
    history,
    best_acc=0.0,
    class_to_idx=None,
):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    model.to(device)

    print(f"\n===== {stage_name} =====")
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 10)

        epoch_record = {"stage": stage_name}

        for phase in ["train", "valid"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            epoch_record[f"{phase}_loss"] = epoch_loss
            epoch_record[f"{phase}_acc"] = epoch_acc

            elapsed = time.time() - since
            print(f"Time elapsed {elapsed // 60:.0f}m {elapsed % 60:.0f}s")
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "valid" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "best_acc": best_acc,
                        "stage": stage_name,
                        "optimizer": optimizer.state_dict(),
                        "class_to_idx": class_to_idx,
                        "num_classes": NUM_CLASSES,
                    },
                    BEST_MODEL_PATH,
                )

        history["stage"].append(stage_name)
        history["train_loss"].append(epoch_record["train_loss"])
        history["valid_loss"].append(epoch_record["valid_loss"])
        history["train_acc"].append(epoch_record["train_acc"])
        history["valid_acc"].append(epoch_record["valid_acc"])
        history["lr"].append(format_lrs(optimizer))

        print(f"Optimizer learning rate: {format_lrs(optimizer)}")
        scheduler.step()
        print()

    elapsed = time.time() - since
    print(f"{stage_name} complete in {elapsed // 60:.0f}m {elapsed % 60:.0f}s")
    print(f"Best val Acc so far: {best_acc:.4f}")

    model.load_state_dict(best_model_wts)
    return model, best_acc


def plot_training_curves(history, output_path):
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train loss")
    plt.plot(epochs, history["valid_loss"], label="Valid loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train acc")
    plt.plot(epochs, history["valid_acc"], label="Valid acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Training curves saved to: {output_path}")


def load_plant_names(path):
    for encoding in ["utf-8", "gbk"]:
        try:
            with open(path, "r", encoding=encoding) as f:
                return json.load(f)
        except (UnicodeDecodeError, json.JSONDecodeError, FileNotFoundError):
            continue
    return {}


def label_name(index, class_names, plant_to_name):
    key = str(int(index))
    if key in plant_to_name:
        return plant_to_name[key]
    if 0 <= int(index) < len(class_names):
        return class_names[int(index)]
    return key


def im_convert(tensor):
    image = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    image = image * np.array(NORMALIZE_STD) + np.array(NORMALIZE_MEAN)
    return np.clip(image, 0, 1)


def save_prediction_preview(model, dataloaders, device, class_names):
    plant_to_name = load_plant_names(PLANT_NAME_FILE)
    images, labels = next(iter(dataloaders["valid"]))

    model.eval()
    with torch.no_grad():
        outputs = model(images.to(device, non_blocking=True))
        _, preds = torch.max(outputs, 1)

    preds = preds.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    fig = plt.figure(figsize=(14, 7))
    columns = 4
    rows = 2
    for idx in range(columns * rows):
        ax = fig.add_subplot(rows, columns, idx + 1, xticks=[], yticks=[])
        ax.imshow(im_convert(images[idx]))
        pred_class = label_name(preds[idx], class_names, plant_to_name)
        true_class = label_name(labels[idx], class_names, plant_to_name)
        color = "green" if preds[idx] == labels[idx] else "red"
        ax.set_title(f"{pred_class} ({true_class})", color=color, fontsize=9)

    plt.tight_layout()
    plt.savefig(PREVIEW_PATH, dpi=150)
    plt.close()
    print(f"Prediction preview saved to: {PREVIEW_PATH}")
    print(f"Output shape: {outputs.shape}")


def main():
    set_seed(RANDOM_SEED)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_workers = min(4, os.cpu_count() or 1)

    print(f"Using device: {device}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"DataLoader num_workers: {num_workers}")

    dataloaders, dataset_sizes, class_names, class_to_idx = build_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=num_workers,
        device=device,
    )
    print(f"Train images: {dataset_sizes['train']}")
    print(f"Valid images: {dataset_sizes['valid']}")
    print(f"Classes: {len(class_names)}")

    model = initialize_model(NUM_CLASSES, use_pretrained=True)
    criterion = nn.CrossEntropyLoss()
    history = {
        "stage": [],
        "train_loss": [],
        "valid_loss": [],
        "train_acc": [],
        "valid_acc": [],
        "lr": [],
    }

    freeze_backbone(model)
    print_trainable_parameters(model)
    optimizer = optim.Adam(model.fc.parameters(), lr=5e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    model, best_acc = train_model(
        model=model,
        dataloaders=dataloaders,
        dataset_sizes=dataset_sizes,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=STAGE1_EPOCHS,
        stage_name="Stage 1: train fc only",
        history=history,
        best_acc=0.0,
        class_to_idx=class_to_idx,
    )

    unfreeze_layer4_and_fc(model)
    print_trainable_parameters(model)
    optimizer = optim.Adam(
        [
            {"params": model.layer4.parameters(), "lr": 1e-4},
            {"params": model.fc.parameters(), "lr": 5e-4},
        ]
    )
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    model, best_acc = train_model(
        model=model,
        dataloaders=dataloaders,
        dataset_sizes=dataset_sizes,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=STAGE2_EPOCHS,
        stage_name="Stage 2: fine-tune layer4 and fc",
        history=history,
        best_acc=best_acc,
        class_to_idx=class_to_idx,
    )

    print("\n===== Training finished =====")
    print(f"Best validation accuracy: {best_acc:.4f}")
    print(f"Best model saved to: {BEST_MODEL_PATH}")

    plot_training_curves(history, CURVE_PATH)
    save_prediction_preview(model, dataloaders, device, class_names)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
