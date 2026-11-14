"""Fine-tune a dedicated ambulance/fire-truck detector on a labelled Roboflow dataset.

Dataset source:
https://universe.roboflow.com/s-workspace-kc9f0/fire-truck-ambulance-detection
License: CC BY 4.0. Credit the dataset author in project reports.
"""
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import yaml
import torch
from roboflow import Roboflow
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
BASE_WEIGHTS = ROOT / "yolo11n.pt"
OUTPUT_WEIGHTS = ROOT / "model" / "emergency_vehicle_detector.pt"
REPORT_PATH = ROOT / "reports" / "emergency_model_evaluation.json"
RUNS_DIR = ROOT / "runs" / "emergency_training"

WORKSPACE = "s-workspace-kc9f0"
PROJECT = "fire-truck-ambulance-detection"
VERSION = 2
IMAGE_SIZE = 640
EPOCHS = int(os.environ.get("EMERGENCY_TRAIN_EPOCHS", "100"))


def normalize_name(value):
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def read_dataset_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    names = data.get("names", {})
    if isinstance(names, list):
        class_names = {index: name for index, name in enumerate(names)}
    else:
        class_names = {int(index): name for index, name in names.items()}

    normalized = {normalize_name(name) for name in class_names.values()}
    if "ambulance" not in normalized:
        raise ValueError(f"Dataset is missing an ambulance class. Found: {class_names}")
    if not any(name in {"fire_truck", "firetruck", "fire_engine"} for name in normalized):
        raise ValueError(f"Dataset is missing a fire-truck class. Found: {class_names}")

    return data, class_names


def metric_value(value):
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


def main():
    api_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "Set ROBOFLOW_API_KEY in the Colab session before training. "
            "Do not put the API key in source code or commit it to GitHub."
        )

    if not BASE_WEIGHTS.is_file():
        raise FileNotFoundError(f"Base YOLO weights not found: {BASE_WEIGHTS}")

    client = Roboflow(api_key=api_key)
    project = client.workspace(WORKSPACE).project(PROJECT)
    version = project.version(VERSION)
    dataset = version.download("yolov8")
    dataset_yaml = Path(dataset.location) / "data.yaml"
    if not dataset_yaml.is_file():
        raise FileNotFoundError(f"Roboflow export has no data.yaml: {dataset_yaml}")

    dataset_config, class_names = read_dataset_config(dataset_yaml)
    device = 0 if torch.cuda.is_available() else "cpu"
    batch = 16 if torch.cuda.is_available() else 4

    print(f"Dataset: {WORKSPACE}/{PROJECT}/{VERSION}")
    print(f"Classes: {class_names}")
    print(f"Device: {device}")
    print(f"Training: {EPOCHS} epochs at {IMAGE_SIZE}px")

    base_model = YOLO(str(BASE_WEIGHTS))
    training = base_model.train(
        data=str(dataset_yaml),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=batch,
        device=device,
        patience=25,
        workers=2,
        seed=42,
        project=str(RUNS_DIR),
        name="yolo11n_ambulance_fire_truck",
        exist_ok=True,
        close_mosaic=10,
        plots=True,
        save=True,
        verbose=True,
    )

    run_dir = Path(training.save_dir)
    best_weights = run_dir / "weights" / "best.pt"
    if not best_weights.is_file():
        raise FileNotFoundError(f"Training did not produce best.pt: {best_weights}")

    trained_model = YOLO(str(best_weights))
    test_metrics = trained_model.val(
        data=str(dataset_yaml),
        split="test",
        imgsz=IMAGE_SIZE,
        conf=0.25,
        iou=0.50,
        plots=True,
        save_json=True,
        project=str(RUNS_DIR),
        name="held_out_test",
        exist_ok=True,
    )

    box = test_metrics.box
    class_precision = getattr(box, "p", [])
    class_recall = getattr(box, "r", [])
    class_ap50 = getattr(box, "ap50", [])
    class_ap = getattr(box, "ap", [])
    class_ids = getattr(box, "ap_class_index", [])
    per_class = {}

    for position, class_id in enumerate(class_ids):
        class_id = int(class_id)
        class_name = trained_model.names.get(class_id, str(class_id))
        per_class[str(class_name)] = {
            "precision": metric_value(class_precision[position]) if len(class_precision) > position else None,
            "recall": metric_value(class_recall[position]) if len(class_recall) > position else None,
            "map50": metric_value(class_ap50[position]) if len(class_ap50) > position else None,
            "map50_95": metric_value(class_ap[position].mean()) if len(class_ap) > position else None,
        }

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_source": f"{WORKSPACE}/{PROJECT}/{VERSION}",
        "dataset_license": "CC BY 4.0",
        "dataset_attribution": "s Workspace — Fire Truck & Ambulance Detection",
        "dataset_yaml": str(dataset_yaml),
        "base_model": str(BASE_WEIGHTS.name),
        "trained_model": str(OUTPUT_WEIGHTS.name),
        "training_epochs": EPOCHS,
        "image_size": IMAGE_SIZE,
        "test_split": "test",
        "test_metrics": {
            "precision": metric_value(box.mp),
            "recall": metric_value(box.mr),
            "map50": metric_value(box.map50),
            "map50_95": metric_value(box.map),
            "per_class": per_class,
        },
        "limitations": [
            "These metrics are from the dataset test split, not from the user's live camera.",
            "The public dataset test split contains 44 images in version 2.",
            "Independent road-camera validation is still required before any deployment claim.",
            "Only ambulance and fire-truck classes are allowed to activate emergency priority in the application.",
            "Police and ordinary vehicles must never activate emergency priority.",
        ],
    }

    OUTPUT_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_weights, OUTPUT_WEIGHTS)
    with open(REPORT_PATH, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("\nHeld-out emergency model evaluation")
    print(json.dumps(report["test_metrics"], indent=2))
    print(f"Custom emergency weights saved: {OUTPUT_WEIGHTS}")
    print(f"Evaluation report saved: {REPORT_PATH}")
    print("Restart Flask to auto-load the custom emergency model.")


if __name__ == "__main__":
    main()
