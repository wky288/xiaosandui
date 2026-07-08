from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils import YAML


ROOT = Path(__file__).resolve().parent
BASE_DATA_YAML = ROOT / "houseplant diseases.yaml"
RECALL_DATA_YAML = ROOT / "houseplant diseases_recall.yaml"
DATA_YAML = RECALL_DATA_YAML if RECALL_DATA_YAML.exists() else BASE_DATA_YAML
PROJECT_DIR = ROOT / "runs" / "detect"
RUN_NAME = "leaf_symptom_recall_yolov8s_cpu"
VAL_RUN_NAME = "leaf_symptom_recall_yolov8s_val"
PREDICT_RUN_NAME = "leaf_symptom_recall_yolov8s_predict"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Match training input size. Validation batch is kept at 1 for CPU memory stability.
DEVICE = "cpu"
IMGSZ = 640
VAL_BATCH_SIZE = 1
CONF_THRES = 0.25
IOU_THRES = 0.5
PREDICT_PREVIEW_LIMIT = 20


def resolve_one_path(dataset_root, value):
    path = Path(value)
    return path if path.is_absolute() else dataset_root / path


def resolve_dataset_root(data):
    raw_path = Path(data.get("path", DATA_YAML.parent))
    if raw_path.is_absolute():
        candidates = [raw_path]
    else:
        candidates = [
            DATA_YAML.parent / raw_path,
            DATA_YAML.parent / "datasets" / raw_path,
            ROOT / "datasets" / raw_path,
        ]

    for candidate in candidates:
        candidate = candidate.resolve()
        if (candidate / "images").exists():
            return candidate

    return (DATA_YAML.parent / raw_path).resolve()


def resolve_split_path(data, key):
    dataset_root = resolve_dataset_root(data)
    value = data[key]
    if isinstance(value, (list, tuple)):
        return [resolve_one_path(dataset_root, item) for item in value]
    return resolve_one_path(dataset_root, value)


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


def find_best_weights():
    preferred = PROJECT_DIR / RUN_NAME / "weights" / "best.pt"
    fallback = PROJECT_DIR / RUN_NAME / "weights" / "last.pt"
    for path in (preferred, fallback):
        if path.exists():
            return path

    candidates = sorted(PROJECT_DIR.glob("*/weights/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0]
    raise FileNotFoundError("No trained weights found. Run yolov8_train.py first.")


def class_name(names, cls_id):
    if isinstance(names, dict):
        return names.get(cls_id, str(cls_id))
    return names[cls_id] if cls_id < len(names) else str(cls_id)


def print_metrics(metrics):
    values = getattr(metrics, "results_dict", None)
    if values is None and isinstance(metrics, dict):
        values = metrics
    if not values:
        return

    print("\nValidation metrics:")
    for key, value in values.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


def main():
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {DATA_YAML}")

    data = YAML.load(DATA_YAML)
    val_path = resolve_split_path(data, "val")
    val_images = list_images(val_path)
    weights_path = find_best_weights()

    print("=" * 80)
    print("YOLOv8 leaf symptom validation")
    print(f"Weights     : {weights_path}")
    print(f"Data yaml   : {DATA_YAML}")
    print(f"Val images  : {len(val_images)} ({val_path})")
    print(
        "Eval params : "
        f"imgsz={IMGSZ}, batch={VAL_BATCH_SIZE}, device={DEVICE}, conf={CONF_THRES}, iou={IOU_THRES}"
    )
    print("=" * 80)

    model = YOLO(str(weights_path))

    # Full val split evaluation reports precision, recall and mAP.
    metrics = model.val(
        data=str(DATA_YAML),
        split="val",
        imgsz=IMGSZ,
        batch=VAL_BATCH_SIZE,
        device=DEVICE,
        workers=0,
        conf=CONF_THRES,
        iou=IOU_THRES,
        plots=True,
        verbose=True,
        project=str(PROJECT_DIR),
        name=VAL_RUN_NAME,
        exist_ok=True,
    )
    print_metrics(metrics)

    # Save a bounded visual sample to verify labels/boxes without a long CPU predict pass.
    sample_images = val_images[:PREDICT_PREVIEW_LIMIT]
    if not sample_images:
        print("No validation images found for prediction preview.")
        return

    results = model.predict(
        source=[str(path) for path in sample_images],
        imgsz=IMGSZ,
        conf=CONF_THRES,
        iou=IOU_THRES,
        device=DEVICE,
        save=True,
        save_txt=True,
        save_conf=True,
        augment=False,
        verbose=False,
        project=str(PROJECT_DIR),
        name=PREDICT_RUN_NAME,
        exist_ok=True,
    )

    class_counts = {name: 0 for name in model.names.values()} if isinstance(model.names, dict) else {}
    total_boxes = 0
    for result in results:
        total_boxes += len(result.boxes)
        for box in result.boxes:
            cls_id = int(box.cls.item())
            name = class_name(model.names, cls_id)
            class_counts[name] = class_counts.get(name, 0) + 1

    print("\nPrediction preview:")
    print(f"  Images checked : {len(sample_images)}")
    print(f"  Boxes detected : {total_boxes}")
    for name, count in class_counts.items():
        if count:
            print(f"  {name}: {count}")
    print(f"  Saved to       : {PROJECT_DIR / PREDICT_RUN_NAME}")


if __name__ == "__main__":
    main()
