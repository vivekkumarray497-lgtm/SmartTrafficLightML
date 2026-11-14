"""Fine-tune YOLO11n for cars, two-wheelers, buses, trucks and auto-rickshaws.

Dataset: Vehicle detection by Shubhams Workspace, Roboflow Universe.
Dataset release: vehicle-detection-eckrb-cppsf/1, CC BY 4.0.
The source labels are consolidated into five project categories. Ambulance and
other out-of-scope labels are left to the dedicated emergency detector or ignored.

Set ROBOFLOW_API_KEY in the environment before running this script.
A model is copied into model/traffic_vehicle_detector.pt only when its held-out
test mAP@50 reaches the configured promotion threshold (default 0.80).
"""
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from roboflow import Roboflow
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
BASE_WEIGHTS = ROOT / "yolo11n.pt"
OUTPUT_WEIGHTS = ROOT / "model" / "traffic_vehicle_detector.pt"
REPORT_PATH = ROOT / "reports" / "traffic_vehicle_detector_evaluation.json"
RUNS_DIR = ROOT / "runs" / "traffic_vehicle_training"
DATASET_ROOT = ROOT / "datasets" / "traffic_vehicle_classes"

WORKSPACE = "shubhams-workspace-wokrm"
PROJECT = "vehicle-detection-eckrb-cppsf"
VERSION = 1
IMAGE_SIZE = int(os.environ.get("TRAFFIC_TRAIN_IMAGE_SIZE", "960"))
EPOCHS = int(os.environ.get("TRAFFIC_TRAIN_EPOCHS", "100"))
MIN_MAP50 = float(os.environ.get("TRAFFIC_MIN_MAP50", "0.80"))
TARGET_NAMES = ["car", "bike", "bus", "truck", "auto_rickshaw"]
NORMALIZED_LABELS = {
    "car": 0,
    "cars": 0,
    "bicycle": 1,
    "bike": 1,
    "motorbike": 1,
    "motorcycle": 1,
    "bus": 2,
    "buses": 2,
    "truck": 3,
    "trucks": 3,
    "lorry": 3,
    "mini truck": 3,
    "mini_truck": 3,
    "auto": 4,
    "auto rickshaw": 4,
    "rickshaw": 4,
    "tuk tuk": 4,
    "tuktuk": 4,
}


def normalize_label(value):
    return str(value).strip().lower().replace("-", " ").replace("_", " ")


def resolve_source_class_map(names):
    if isinstance(names, list):
        source_names = {index: name for index, name in enumerate(names)}
    else:
        source_names = {int(index): name for index, name in names.items()}

    class_map = {}
    for source_id, name in source_names.items():
        normalized = normalize_label(name)
        target_id = NORMALIZED_LABELS.get(normalized)
        if target_id is not None:
            class_map[source_id] = target_id

    present = set(class_map.values())
    missing = [TARGET_NAMES[index] for index in range(len(TARGET_NAMES)) if index not in present]
    if missing:
        raise ValueError(
            f"Source dataset does not contain required categories {missing}. "
            f"Source names: {source_names}"
        )
    return source_names, class_map


def find_split_path(dataset_location, dataset_config, split):
    candidate = dataset_config.get(split)
    if not candidate and split == "val":
        candidate = dataset_config.get("valid")
    if not candidate:
        raise ValueError(f"Dataset config does not define the {split} split.")

    candidate = Path(candidate)
    if candidate.is_absolute() and candidate.is_dir():
        return candidate

    dataset_location = Path(dataset_location).resolve()
    configured_root = Path(dataset_config.get("path", "."))
    if not configured_root.is_absolute():
        configured_root = (dataset_location / configured_root).resolve()

    candidates = [
        (configured_root / candidate).resolve(),
        (dataset_location / candidate).resolve(),
        (dataset_location.parent / candidate).resolve(),
    ]
    aliases = {"train": ("train",), "val": ("valid", "val"), "test": ("test",)}
    candidates.extend(
        (dataset_location / name / "images").resolve()
        for name in aliases[split]
    )
    for path in candidates:
        if path.is_dir():
            return path

    for images_dir in dataset_location.rglob("images"):
        if images_dir.parent.name in aliases[split] and images_dir.is_dir():
            return images_dir.resolve()

    raise FileNotFoundError(
        f"Could not locate {split} images. YAML value={candidate}; "
        f"searched under {dataset_location}"
    )

def prepare_four_class_dataset(dataset_location, source_config):
    source_names, class_map = resolve_source_class_map(source_config.get("names", {}))
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    split_dirs = {}
    object_counts = {name: 0 for name in TARGET_NAMES}

    for split, config_key in (("train", "train"), ("val", "val"), ("test", "test")):
        source_images = find_split_path(dataset_location, source_config, config_key)
        if not source_images.is_dir():
            raise FileNotFoundError(f"Missing {split} image directory: {source_images}")
        source_labels = source_images.parent / "labels"
        if not source_labels.is_dir():
            raise FileNotFoundError(f"Missing {split} labels directory: {source_labels}")

        output_split = DATASET_ROOT / split
        output_images = output_split / "images"
        output_labels = output_split / "labels"
        output_split.mkdir(parents=True, exist_ok=True)
        if output_images.is_symlink() or output_images.exists():
            if output_images.is_dir() and not output_images.is_symlink():
                shutil.rmtree(output_images)
            else:
                output_images.unlink()
        output_images.symlink_to(source_images, target_is_directory=True)
        if output_labels.exists():
            shutil.rmtree(output_labels)
        output_labels.mkdir(parents=True, exist_ok=True)

        label_files = list(source_labels.glob("*.txt"))
        if not label_files:
            raise ValueError(f"No YOLO label files found in {source_labels}")

        for label_file in label_files:
            mapped_lines = []
            for line in label_file.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                try:
                    source_id = int(float(parts[0]))
                except ValueError:
                    continue
                target_id = class_map.get(source_id)
                if target_id is None:
                    continue
                parts[0] = str(target_id)
                mapped_lines.append(" ".join(parts))
                object_counts[TARGET_NAMES[target_id]] += 1
            (output_labels / label_file.name).write_text(
                "\n".join(mapped_lines) + ("\n" if mapped_lines else ""),
                encoding="utf-8",
            )
        split_dirs[split] = output_images

    empty_classes = [name for name, count in object_counts.items() if count == 0]
    if empty_classes:
        raise ValueError(f"No mapped training objects found for classes: {empty_classes}")

    config_path = DATASET_ROOT / "data.yaml"
    config = {
        "path": str(DATASET_ROOT.resolve()),
        "train": str(split_dirs["train"].absolute()),
        "val": str(split_dirs["val"].absolute()),
        "test": str(split_dirs["test"].absolute()),
        "nc": len(TARGET_NAMES),
        "names": TARGET_NAMES,
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path, source_names, class_map, object_counts


def safe_metric(value):
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


def main():
    api_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "Set ROBOFLOW_API_KEY in your Colab session. Never save it in GitHub or source files."
        )
    if not BASE_WEIGHTS.is_file():
        raise FileNotFoundError(f"YOLO base weights missing: {BASE_WEIGHTS}")
    if EPOCHS < 1 or IMAGE_SIZE < 320:
        raise ValueError("Training epochs must be positive and image size must be at least 320.")

    client = Roboflow(api_key=api_key)
    project = client.workspace(WORKSPACE).project(PROJECT)
    dataset = project.version(VERSION).download("yolov8")
    dataset_location = Path(dataset.location)
    source_yaml = dataset_location / "data.yaml"
    if not source_yaml.is_file():
        raise FileNotFoundError(f"Roboflow export has no data.yaml: {source_yaml}")

    source_config = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
    dataset_yaml, source_names, class_map, object_counts = prepare_four_class_dataset(
        dataset_location, source_config
    )

    use_gpu = torch.cuda.is_available()
    device = 0 if use_gpu else "cpu"
    batch = int(os.environ.get("TRAFFIC_TRAIN_BATCH", "12" if use_gpu else "2"))
    print(f"Dataset: {WORKSPACE}/{PROJECT}/{VERSION}")
    print(f"Source classes: {source_names}")
    print(f"Mapped labels: {class_map}")
    print(f"Four-class object counts: {object_counts}")
    print(f"Training on {'GPU' if use_gpu else 'CPU'} for {EPOCHS} epochs at {IMAGE_SIZE}px")

    model = YOLO(str(BASE_WEIGHTS))
    model.train(
        data=str(dataset_yaml),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=batch,
        device=device,
        patience=25,
        workers=2,
        seed=42,
        project=str(RUNS_DIR),
        name="yolo11n_four_class_vehicles",
        exist_ok=True,
        close_mosaic=10,
        plots=True,
        save=True,
        verbose=True,
    )

    trainer = getattr(model, "trainer", None)
    if trainer is None:
        raise RuntimeError("Ultralytics did not expose the training result.")
    run_dir = Path(trainer.save_dir)
    best_weights = Path(trainer.best) if getattr(trainer, "best", None) else run_dir / "weights" / "best.pt"
    if not best_weights.is_file():
        raise FileNotFoundError(f"Training did not create best.pt: {best_weights}")

    trained_model = YOLO(str(best_weights))
    test_metrics = trained_model.val(
        data=str(dataset_yaml),
        split="test",
        imgsz=IMAGE_SIZE,
        conf=0.001,
        iou=0.50,
        plots=True,
        save_json=True,
        project=str(RUNS_DIR),
        name="held_out_test",
        exist_ok=True,
    )
    box = test_metrics.box
    map50 = safe_metric(getattr(box, "map50", None))
    map50_95 = safe_metric(getattr(box, "map", None))
    precision = safe_metric(getattr(box, "mp", None))
    recall = safe_metric(getattr(box, "mr", None))
    class_ids = list(getattr(box, "ap_class_index", []))
    per_class = {}
    for position, class_id in enumerate(class_ids):
        class_id = int(class_id)
        per_class[TARGET_NAMES[class_id]] = {
            "precision": safe_metric(box.p[position]) if len(box.p) > position else None,
            "recall": safe_metric(box.r[position]) if len(box.r) > position else None,
            "map50": safe_metric(box.ap50[position]) if len(box.ap50) > position else None,
            "map50_95": safe_metric(box.ap[position].mean()) if len(box.ap) > position else None,
        }

    promoted = map50 is not None and map50 >= MIN_MAP50
    if promoted:
        OUTPUT_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
        temporary_weights = OUTPUT_WEIGHTS.with_suffix(".tmp.pt")
        shutil.copy2(best_weights, temporary_weights)
        temporary_weights.replace(OUTPUT_WEIGHTS)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": f"{WORKSPACE}/{PROJECT}/{VERSION}",
        "dataset_license": "CC BY 4.0",
        "dataset_attribution": "Shubhams Workspace — Vehicle detection",
        "dataset_images": 2741,
        "source_class_names": source_names,
        "target_class_names": TARGET_NAMES,
        "mapped_object_counts": object_counts,
        "source_class_mapping": {str(key): value for key, value in class_map.items()},
        "base_model": BASE_WEIGHTS.name,
        "training_epochs_requested": EPOCHS,
        "image_size": IMAGE_SIZE,
        "test_split": "held-out Roboflow test split",
        "test_metrics": {
            "precision": precision,
            "recall": recall,
            "map50": map50,
            "map50_95": map50_95,
            "per_class": per_class,
        },
        "promotion_threshold_map50": MIN_MAP50,
        "promoted_to_application": promoted,
        "trained_weights": str(best_weights),
        "application_weights": str(OUTPUT_WEIGHTS) if promoted else None,
        "metric_note": "mAP@50 is object-detection mean average precision; it is not classification accuracy or end-to-end traffic-system accuracy.",
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not promoted:
        raise SystemExit(
            f"Model NOT promoted: held-out mAP@50={map50}; required minimum={MIN_MAP50}. "
            "Collect more representative images, tune the model, and retrain."
        )


if __name__ == "__main__":
    main()
