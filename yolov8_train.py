import os
import random
from collections import Counter
from pathlib import Path

# Force CPU mode before torch/ultralytics initialization.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from ultralytics import YOLO
from ultralytics.utils import YAML


ROOT = Path(__file__).resolve().parent
TRAIN_CFG_YAML = ROOT / "yolov8_recall_train.yaml"
BASE_DATA_YAML = ROOT / "houseplant diseases.yaml"
DATA_YAML = ROOT / "houseplant diseases_recall.yaml"
TRAIN_LIST_NAME = "train_oversampled_mold_wilt.txt"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

REQUIRED_WEIGHTS_NAME = "yolov8s.pt"
OVERSAMPLE_CLASS_IDS = {2, 3}  # mold, wilt
OVERSAMPLE_FACTOR = 1.35
OVERSAMPLE_SEED = 42

TORCH_THREADS = 12
TORCH_INTEROP_THREADS = 4

# Keep export disabled during training review; enable after the best training run is selected.
EXPORT_ONNX = False
ONNX_OPSET = 12

def resolve_config_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def resolve_one_path(dataset_root, value):
    path = Path(value)
    return path if path.is_absolute() else dataset_root / path


def resolve_split_path(data, key, dataset_root):
    value = data[key]
    if isinstance(value, (list, tuple)):
        return [resolve_one_path(dataset_root, item) for item in value]
    return resolve_one_path(dataset_root, value)


def load_base_data():
    if not BASE_DATA_YAML.exists():
        raise FileNotFoundError(f"Base dataset yaml not found: {BASE_DATA_YAML}")
    return YAML.load(BASE_DATA_YAML)


def resolve_dataset_root(data):
    raw_path = Path(data.get("path", BASE_DATA_YAML.parent))
    if raw_path.is_absolute():
        candidates = [raw_path]
    else:
        candidates = [
            BASE_DATA_YAML.parent / raw_path,
            BASE_DATA_YAML.parent / "datasets" / raw_path,
            ROOT / "datasets" / raw_path,
        ]

    for candidate in candidates:
        candidate = candidate.resolve()
        if (candidate / "images").exists() and (candidate / "labels").exists():
            return candidate

    tried = "\n  ".join(str(path.resolve()) for path in candidates)
    raise FileNotFoundError(f"Could not resolve dataset root from {BASE_DATA_YAML}. Tried:\n  {tried}")

def list_images(path_or_paths):
    paths = path_or_paths if isinstance(path_or_paths, list) else [path_or_paths]
    images = []
    for path in paths:
        path = Path(path)
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            images.append(path)
        elif path.is_dir():
            images.extend(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    return sorted(images)


def label_path_for_image(image_path):
    parts = list(Path(image_path).parts)
    for index, part in enumerate(parts):
        if part.lower() == "images":
            parts[index] = "labels"
            return Path(*parts).with_suffix(".txt")
    raise ValueError(f"Image path does not contain an images directory: {image_path}")


def read_label_class_ids(image_path):
    label_path = label_path_for_image(image_path)
    if not label_path.exists():
        return []

    class_ids = []
    with label_path.open("r", encoding="utf-8") as file:
        for line in file:
            fields = line.strip().split()
            if fields:
                class_ids.append(int(float(fields[0])))
    return class_ids


def class_name(names, cls_id):
    if isinstance(names, dict):
        return names.get(cls_id, names.get(str(cls_id), str(cls_id)))
    return names[cls_id] if cls_id < len(names) else str(cls_id)


def summarize_class_boxes(image_paths):
    counts = Counter()
    for image_path in image_paths:
        counts.update(read_label_class_ids(image_path))
    return counts


def build_oversampled_train_list(train_images, dataset_root):
    eligible = []
    for image_path in train_images:
        class_ids = set(read_label_class_ids(image_path))
        if class_ids & OVERSAMPLE_CLASS_IDS:
            eligible.append(image_path)

    extra_count = round(len(eligible) * (OVERSAMPLE_FACTOR - 1.0))
    rng = random.Random(OVERSAMPLE_SEED)
    extra_images = eligible[:]
    rng.shuffle(extra_images)
    extra_images = sorted(extra_images[:extra_count])

    oversampled = train_images + extra_images
    train_list_path = dataset_root / TRAIN_LIST_NAME
    train_list_path.write_text(
        "\n".join(path.resolve().as_posix() for path in oversampled) + "\n",
        encoding="utf-8",
    )
    return oversampled, eligible, extra_images, train_list_path


def write_recall_data_yaml(dataset_root, names):
    name_lines = []
    for cls_id in sorted(int(key) for key in names.keys()):
        name_lines.append(f"  {cls_id}: {class_name(names, cls_id)}")

    DATA_YAML.write_text(
        "\n".join(
            [
                "# Recall-oriented data config generated for YOLOv8s leaf symptom training.",
                f"# Source dataset yaml: {BASE_DATA_YAML.as_posix()}",
                f"path: {dataset_root.resolve().as_posix()}",
                f"train: {TRAIN_LIST_NAME}",
                "val: images/val",
                "test:",
                "",
                "names:",
                *name_lines,
                "",
            ]
        ),
        encoding="utf-8",
    )


def prepare_recall_dataset_config():
    base_data = load_base_data()
    dataset_root = resolve_dataset_root(base_data)

    train_path = resolve_split_path(base_data, "train", dataset_root)
    val_path = resolve_split_path(base_data, "val", dataset_root)
    train_images = list_images(train_path)
    val_images = list_images(val_path)
    if not train_images:
        raise FileNotFoundError(f"No train images found: {train_path}")
    if not val_images:
        raise FileNotFoundError(f"No val images found: {val_path}")

    oversampled, eligible, extra_images, train_list_path = build_oversampled_train_list(train_images, dataset_root)
    write_recall_data_yaml(dataset_root, base_data["names"])
    data = YAML.load(DATA_YAML)
    return data, dataset_root, train_list_path, train_images, val_images, oversampled, eligible, extra_images


def load_train_config():
    if not TRAIN_CFG_YAML.exists():
        raise FileNotFoundError(f"Training config yaml not found: {TRAIN_CFG_YAML}")

    cfg = YAML.load(TRAIN_CFG_YAML)
    weights_value = cfg.pop("weights", REQUIRED_WEIGHTS_NAME)
    if Path(weights_value).name.lower() != REQUIRED_WEIGHTS_NAME:
        raise ValueError(f"Only {REQUIRED_WEIGHTS_NAME} is allowed for this task, got: {weights_value}")

    local_weights = resolve_config_path(weights_value)
    weights = local_weights if local_weights.exists() else Path(weights_value)
    if weights.suffix.lower() == ".yaml":
        raise ValueError("Training from a model YAML is not allowed; use COCO pretrained yolov8s.pt.")

    cfg["data"] = str(resolve_config_path(cfg.get("data", DATA_YAML.name)))
    cfg["project"] = str(resolve_config_path(cfg.get("project", "runs/detect")))
    return cfg, weights


def configure_cpu():
    # Limit CPU threads so training does not monopolize the whole Windows desktop.
    torch.set_num_threads(TORCH_THREADS)
    try:
        torch.set_num_interop_threads(TORCH_INTEROP_THREADS)
    except RuntimeError:
        pass

def export_to_onnx(weights_path, imgsz, device):
    model = YOLO(str(weights_path))
    return model.export(
        format="onnx",
        imgsz=imgsz,
        device=device,
        batch=1,
        dynamic=False,
        simplify=False,
        opset=ONNX_OPSET,
        nms=False,
        half=False,
    )


def print_class_counts(title, counts, names):
    print(title)
    for cls_id in sorted(names.keys() if isinstance(names, dict) else range(len(names))):
        print(f"  {class_name(names, cls_id):9s}: {counts.get(int(cls_id), 0)}")


def main():
    data, dataset_root, train_list_path, train_images, val_images, oversampled, eligible, extra_images = (
        prepare_recall_dataset_config()
    )
    train_args, weights = load_train_config()
    configure_cpu()

    base_counts = summarize_class_boxes(train_images)
    sampled_counts = summarize_class_boxes(oversampled)
    project_dir = Path(train_args["project"])
    run_name = train_args["name"]

    print("=" * 80)
    print("YOLOv8s recall-first leaf symptom training")
    print(f"Device        : {train_args['device']}")
    print(f"Torch threads : {TORCH_THREADS}, interop={TORCH_INTEROP_THREADS}")
    print(f"Train config  : {TRAIN_CFG_YAML}")
    print(f"Base data yaml: {BASE_DATA_YAML}")
    print(f"Data yaml     : {DATA_YAML}")
    print(f"Dataset root  : {dataset_root}")
    print(f"Train list    : {train_list_path}")
    print(f"Train images  : {len(train_images)} base + {len(extra_images)} oversampled = {len(oversampled)}")
    print(f"Val images    : {len(val_images)}")
    print(f"Mold/wilt imgs: {len(eligible)} eligible, factor={OVERSAMPLE_FACTOR}")
    print(f"Weights       : {weights}")
    print(
        "Train params  : "
        f"epochs={train_args['epochs']}, batch={train_args['batch']}, imgsz={train_args['imgsz']}, "
        f"optimizer={train_args['optimizer']}, lr0={train_args['lr0']}"
    )
    print(
        "Eval params   : "
        f"conf={train_args['conf']}, iou={train_args['iou']}"
    )
    print_class_counts("Base train boxes:", base_counts, data["names"])
    print_class_counts("Sampled train boxes:", sampled_counts, data["names"])
    print("=" * 80)

    model = YOLO(str(weights))
    model.train(**train_args)

    run_dir = project_dir / run_name
    best_pt = run_dir / "weights" / "best.pt"
    last_pt = run_dir / "weights" / "last.pt"
    final_weights = best_pt if best_pt.exists() else last_pt
    print(f"Final weights: {final_weights}")

    if EXPORT_ONNX and final_weights.exists():
        onnx_path = export_to_onnx(final_weights, train_args["imgsz"], train_args["device"])
        print(f"ONNX export  : {onnx_path}")


if __name__ == "__main__":
    main()

