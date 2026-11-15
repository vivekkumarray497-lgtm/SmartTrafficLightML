"""Train a signal-state model from independently human-verified camera samples.

This dataset is separate from the synthetic historical CSV. The model is not
promoted unless it beats a majority baseline on camera sessions it has not seen.
"""
import json
import os
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "verified_signal_training.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "verified_camera_signal_model.pkl")
REPORT_PATH = os.path.join(BASE_DIR, "reports", "verified_camera_signal_evaluation.json")

LABELS = ["Red", "Yellow", "Green", "All Red"]
FEATURES = [
    "cars", "bikes", "buses", "trucks", "auto_rickshaws",
    "total_vehicles", "pedestrians", "avg_speed", "queue_length",
    "avg_waiting_time", "north_vehicles", "south_vehicles",
    "east_vehicles", "west_vehicles", "north_pedestrians",
    "south_pedestrians", "east_pedestrians", "west_pedestrians",
    "input_frame_width", "input_frame_height",
]
MINIMUM_SAMPLES = 120
MINIMUM_SESSIONS = 4
MINIMUM_PER_CLASS = 15


def load_verified_data():
    if not os.path.isfile(DATA_PATH):
        raise FileNotFoundError(
            "No human-verified camera data exists yet. Save reviewed frames from "
            "the dashboard first."
        )

    data = pd.read_csv(DATA_PATH)
    required = {"scene_id", "session_id", "actual_signal", *FEATURES}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Verified dataset is missing columns: {missing}")

    data = data.drop_duplicates(subset=["scene_id"], keep="first").copy()
    data["actual_signal"] = data["actual_signal"].astype(str).str.strip()
    data["session_id"] = data["session_id"].astype(str).str.strip()

    for feature in FEATURES:
        data[feature] = pd.to_numeric(data[feature], errors="coerce")

    data = data.dropna(subset=FEATURES + ["actual_signal", "session_id"])
    data = data[data["actual_signal"].isin(LABELS)].reset_index(drop=True)

    counts = data["actual_signal"].value_counts().to_dict()
    missing_classes = [label for label in LABELS if counts.get(label, 0) < MINIMUM_PER_CLASS]
    sessions = data["session_id"].nunique()

    if len(data) < MINIMUM_SAMPLES:
        raise ValueError(
            f"Need at least {MINIMUM_SAMPLES} verified camera samples; found {len(data)}."
        )
    if sessions < MINIMUM_SESSIONS:
        raise ValueError(
            f"Need at least {MINIMUM_SESSIONS} separate camera sessions; found {sessions}. "
            "Restart the camera between collection sessions."
        )
    if missing_classes:
        raise ValueError(
            "Need at least "
            f"{MINIMUM_PER_CLASS} verified samples for each signal class. "
            f"Still short: {', '.join(missing_classes)}."
        )
    return data


def grouped_holdout(X, y, groups):
    splitter = GroupShuffleSplit(n_splits=250, test_size=0.25, random_state=42)
    for train_indices, test_indices in splitter.split(X, y, groups):
        train_labels = set(y.iloc[train_indices])
        test_labels = set(y.iloc[test_indices])
        if train_labels == set(LABELS) and test_labels == set(LABELS):
            return train_indices, test_indices
    raise ValueError(
        "Could not create a session-separated holdout containing all four signal "
        "classes. Collect every signal class across multiple camera sessions."
    )


def main():
    data = load_verified_data()
    X = data[FEATURES].astype(float)
    y = data["actual_signal"]
    groups = data["session_id"]

    train_indices, test_indices = grouped_holdout(X, y, groups)
    X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
    train_groups = groups.iloc[train_indices]

    fold_count = min(5, train_groups.nunique())
    if fold_count < 2:
        raise ValueError("At least two training sessions are required for grouped cross-validation.")
    cv = GroupKFold(n_splits=fold_count)

    candidates = {
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=500, min_samples_leaf=2, max_features="sqrt",
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=500, min_samples_leaf=2, max_features="sqrt",
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=180, learning_rate=0.05, max_depth=2, random_state=42
        ),
    }

    cv_results = {}
    fitted = {}
    for name, estimator in candidates.items():
        scores = cross_val_score(
            estimator, X_train, y_train, groups=train_groups,
            cv=cv, scoring="f1_macro", n_jobs=None
        )
        estimator.fit(X_train, y_train)
        cv_results[name] = {
            "grouped_cv_macro_f1": round(float(scores.mean()), 4),
            "grouped_cv_std": round(float(scores.std()), 4),
        }
        fitted[name] = estimator

    selected_name = max(
        cv_results, key=lambda name: cv_results[name]["grouped_cv_macro_f1"]
    )
    candidate = fitted[selected_name]

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    baseline_accuracy = accuracy_score(y_test, baseline.predict(X_test))

    prediction = candidate.predict(X_test)
    accuracy = accuracy_score(y_test, prediction)
    macro_f1 = f1_score(y_test, prediction, labels=LABELS, average="macro", zero_division=0)
    promotion_gate = accuracy >= baseline_accuracy + 0.05 and macro_f1 >= 0.50
    promotion_decision = (
        "Promoted: held-out grouped accuracy beat the majority baseline by at least "
        "5 percentage points and macro-F1 reached 0.50."
        if promotion_gate else
        "Not promoted: candidate did not pass both the majority-baseline and macro-F1 gates."
    )

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "evaluation_type": "human_verified_camera_signal_classification",
        "dataset_samples": int(len(data)),
        "camera_sessions": int(groups.nunique()),
        "train_samples": int(len(train_indices)),
        "test_samples": int(len(test_indices)),
        "train_sessions": int(train_groups.nunique()),
        "test_sessions": int(groups.iloc[test_indices].nunique()),
        "split": "grouped by camera session; no session appears in both train and test",
        "class_counts": {label: int((y == label).sum()) for label in LABELS},
        "majority_baseline_accuracy": round(float(baseline_accuracy), 4),
        "selected_model": selected_name,
        "grouped_cross_validation": cv_results,
        "candidate_holdout_accuracy": round(float(accuracy), 4),
        "candidate_holdout_macro_f1": round(float(macro_f1), 4),
        "promoted": bool(promotion_gate),
        "promotion_decision": promotion_decision,
        "classification_report": classification_report(
            y_test, prediction, labels=LABELS, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_test, prediction, labels=LABELS).tolist(),
        "note": (
            "This score uses only human-verified camera frames and tests on separate "
            "camera sessions. It measures signal-state classification on the collected "
            "camera domain; it is not proof of road-safety or deployment readiness."
        ),
    }

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    if promotion_gate:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(candidate, MODEL_PATH)

    print(f"Verified samples: {len(data)} | camera sessions: {groups.nunique()}")
    print(f"Holdout accuracy: {accuracy * 100:.2f}%")
    print(f"Majority baseline: {baseline_accuracy * 100:.2f}%")
    print(f"Macro-F1: {macro_f1:.3f}")
    print(promotion_decision)
    print(f"Report: {REPORT_PATH}")
    if promotion_gate:
        print(f"Saved model: {MODEL_PATH}")


if __name__ == "__main__":
    main()
