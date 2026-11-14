"""Train a traffic-priority model against a transparent, rule-based policy.

The generated target is an operational policy label, not observed ground truth.
The reported test score is policy agreement, not real-world traffic accuracy.
"""
import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_PATH = os.path.join(BASE_DIR, "dataset", "cleaned_traffic_data.csv")
TRAINING_DATA_PATH = os.path.join(BASE_DIR, "dataset", "traffic_priority_training_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "traffic_priority_model.pkl")
REPORT_PATH = os.path.join(BASE_DIR, "reports", "traffic_policy_evaluation.json")

FLOW_SCALE = 1000.0
VEHICLE_SCALE = 1200.0
SPEED_MIN = 20.0
SPEED_MAX = 80.0
LOW_THRESHOLD = 0.40
HIGH_THRESHOLD = 0.60
CLASS_NAMES = ["LOW", "MEDIUM", "HIGH"]

FEATURES = [
    "flow_pressure",
    "vehicle_load_pressure",
    "speed_pressure",
    "traffic_volume",
    "avg_vehicle_speed",
    "vehicle_count_cars",
    "vehicle_count_trucks",
    "vehicle_count_bikes",
    "vehicle_total",
]


def load_policy_data():
    df = pd.read_csv(SOURCE_PATH)
    required = [
        "timestamp",
        "traffic_volume",
        "avg_vehicle_speed",
        "vehicle_count_cars",
        "vehicle_count_trucks",
        "vehicle_count_bikes",
    ]
    missing = sorted(set(required).difference(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    numeric = [
        "traffic_volume",
        "avg_vehicle_speed",
        "vehicle_count_cars",
        "vehicle_count_trucks",
        "vehicle_count_bikes",
    ]
    for column in numeric:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["timestamp"] + numeric).copy()
    df = df.sort_values("timestamp").reset_index(drop=True)
    if len(df) < 100:
        raise ValueError(f"At least 100 valid traffic records are required; found {len(df)}.")

    df["vehicle_total"] = (
        df["vehicle_count_cars"]
        + df["vehicle_count_trucks"]
        + df["vehicle_count_bikes"]
    )

    df["flow_pressure"] = (df["traffic_volume"] / FLOW_SCALE).clip(0, 1)
    df["vehicle_load_pressure"] = (df["vehicle_total"] / VEHICLE_SCALE).clip(0, 1)
    df["speed_pressure"] = (
        (SPEED_MAX - df["avg_vehicle_speed"]) / (SPEED_MAX - SPEED_MIN)
    ).clip(0, 1)

    df["traffic_pressure_index"] = (
        0.40 * df["flow_pressure"]
        + 0.40 * df["vehicle_load_pressure"]
        + 0.20 * df["speed_pressure"]
    ).clip(0, 1)

    df["traffic_priority"] = np.select(
        [
            df["traffic_pressure_index"] < LOW_THRESHOLD,
            df["traffic_pressure_index"] < HIGH_THRESHOLD,
        ],
        ["LOW", "MEDIUM"],
        default="HIGH",
    )

    return df


def main():
    data = load_policy_data()
    data.to_csv(TRAINING_DATA_PATH, index=False)

    X = data[FEATURES].astype(float)
    y = data["traffic_priority"].astype(str)

    split_index = int(len(data) * 0.80)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    if set(y_train.unique()) != set(CLASS_NAMES) or set(y_test.unique()) != set(CLASS_NAMES):
        raise ValueError("Chronological train and test partitions must each contain LOW, MEDIUM and HIGH labels.")

    candidates = {
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=500,
            min_samples_leaf=1,
            max_features=1.0,
            random_state=42,
            n_jobs=-1,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=1,
            max_features=1.0,
            random_state=42,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=180,
            learning_rate=0.05,
            max_depth=2,
            random_state=42,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=0.5,
            random_state=42,
        ),
    }

    cv = TimeSeriesSplit(n_splits=5)
    cv_results = {}
    fitted = {}

    print(f"Valid records: {len(data)}")
    print(f"Chronological split: {len(X_train)} training / {len(X_test)} held-out test")
    print("Selecting the model using training-only time-series cross-validation.")

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    baseline_prediction = baseline.predict(X_test)
    baseline_accuracy = accuracy_score(y_test, baseline_prediction)

    for name, estimator in candidates.items():
        scores = cross_val_score(estimator, X_train, y_train, cv=cv, scoring="accuracy")
        estimator.fit(X_train, y_train)
        cv_results[name] = {
            "mean_accuracy": round(float(scores.mean()), 4),
            "std_accuracy": round(float(scores.std()), 4),
        }
        fitted[name] = estimator
        print(f"{name}: CV {scores.mean() * 100:.2f}% (+/- {scores.std() * 100:.2f}%)")

    selected_name = max(cv_results, key=lambda name: cv_results[name]["mean_accuracy"])
    selected_model = fitted[selected_name]
    prediction = selected_model.predict(X_test)
    agreement = accuracy_score(y_test, prediction)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(selected_model, MODEL_PATH)

    importance = {}
    if hasattr(selected_model, "feature_importances_"):
        importance = {
            feature: round(float(value), 6)
            for feature, value in sorted(
                zip(FEATURES, selected_model.feature_importances_),
                key=lambda item: item[1],
                reverse=True,
            )
        }

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "evaluation_type": "held_out_policy_agreement",
        "metric_name": "Traffic priority policy agreement",
        "dataset_rows_used": int(len(data)),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "split": "chronological first 80% for training, final 20% for testing",
        "selected_model": selected_name,
        "test_policy_agreement": round(float(agreement), 4),
        "test_macro_f1": round(float(f1_score(y_test, prediction, average="macro")), 4),
        "majority_baseline_agreement": round(float(baseline_accuracy), 4),
        "class_distribution": {
            name: int((y == name).sum()) for name in CLASS_NAMES
        },
        "training_class_distribution": {
            name: int((y_train == name).sum()) for name in CLASS_NAMES
        },
        "test_class_distribution": {
            name: int((y_test == name).sum()) for name in CLASS_NAMES
        },
        "cross_validation": cv_results,
        "feature_importance": importance,
        "classification_report": classification_report(
            y_test, prediction, labels=CLASS_NAMES, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            y_test, prediction, labels=CLASS_NAMES
        ).tolist(),
        "policy": {
            "name": "Traffic pressure priority policy",
            "flow_pressure": "clip(traffic_volume / 1000, 0, 1)",
            "vehicle_load_pressure": "clip((cars + trucks + bikes) / 1200, 0, 1)",
            "speed_pressure": "clip((80 - average_speed) / 60, 0, 1)",
            "traffic_pressure_index": "0.40 * flow_pressure + 0.40 * vehicle_load_pressure + 0.20 * speed_pressure",
            "LOW": f"index < {LOW_THRESHOLD}",
            "MEDIUM": f"{LOW_THRESHOLD} <= index < {HIGH_THRESHOLD}",
            "HIGH": f"index >= {HIGH_THRESHOLD}",
        },
        "features": FEATURES,
        "limitations": [
            "Priority labels are generated from the documented policy, not manually observed road outcomes.",
            "The score measures how well the ML model reproduces this policy on later records.",
            "It is not live-camera detection accuracy, signal safety validation, or proof of real-world traffic improvement.",
            "Operational validation requires independently labelled traffic observations and field testing.",
        ],
    }

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"Selected model: {selected_name}")
    print(f"Held-out policy agreement: {agreement * 100:.2f}%")
    print(f"Majority baseline: {baseline_accuracy * 100:.2f}%")
    print(f"Training dataset: {TRAINING_DATA_PATH}")
    print(f"Model: {MODEL_PATH}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
