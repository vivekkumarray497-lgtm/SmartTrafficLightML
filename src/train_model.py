"""Compare traffic-signal classifiers on a fixed holdout split.

Run from the repository root:
    python src/train_model.py

GitHub Actions also runs this comparison when the script or dataset changes.

The saved model is replaced only when the selected candidate beats the current
model on the same holdout split. A high score cannot be guaranteed if the target
labels do not contain a learnable relationship with the input features.
"""
import json
import os
import sys
from datetime import datetime

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "cleaned_traffic_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "random_forest_model.pkl")
REPORT_PATH = os.path.join(BASE_DIR, "reports", "model_evaluation.json")

FEATURES = [
    "location_id", "traffic_volume", "avg_vehicle_speed",
    "vehicle_count_cars", "vehicle_count_trucks", "vehicle_count_bikes",
    "weather_condition", "temperature", "humidity", "accident_reported",
    "hour", "minute", "day", "month",
]
WEATHER_MAP = {"Sunny": 0, "Cloudy": 1, "Rainy": 2, "Foggy": 3, "Windy": 4}
SIGNAL_MAP = {"Red": 0, "Yellow": 1, "Green": 2}
SIGNAL_NAMES = ["Red", "Yellow", "Green"]


def load_data():
    df = pd.read_csv(DATASET_PATH)
    raw_required = [
        "timestamp", "location_id", "traffic_volume", "avg_vehicle_speed",
        "vehicle_count_cars", "vehicle_count_trucks", "vehicle_count_bikes",
        "weather_condition", "temperature", "humidity", "accident_reported",
        "signal_status",
    ]
    required = set(raw_required)
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df["hour"] = df["timestamp"].dt.hour
    df["minute"] = df["timestamp"].dt.minute
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["weather_condition"] = df["weather_condition"].astype(str).str.strip().map(WEATHER_MAP)
    df["signal_status"] = df["signal_status"].astype(str).str.strip().map(SIGNAL_MAP)

    data = df[FEATURES + ["signal_status"]].apply(pd.to_numeric, errors="coerce").dropna()
    if data.empty:
        raise ValueError("No valid training rows remain after parsing and cleaning.")

    X = data[FEATURES].astype(float)
    y = data["signal_status"].astype(int)
    if y.nunique() < 2:
        raise ValueError("Training requires at least two signal-status classes.")
    return X, y


def main():
    X, y = load_data()
    split_index = int(len(X) * 0.80)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise ValueError("Chronological train and test partitions must each contain at least two signal classes.")
    cv = TimeSeriesSplit(n_splits=5)

    candidates = {
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=500, min_samples_leaf=2, max_features="sqrt",
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=500, min_samples_leaf=2, max_features="sqrt",
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=250, learning_rate=0.06, l2_regularization=1.0,
            random_state=42
        ),
    }

    print(f"Valid rows: {len(X)} | train: {len(X_train)} | holdout: {len(X_test)}")
    print("Holdout split: first 80% of time for training / final 20% for testing.")
    print("Candidate selection uses training-only expanding-window TimeSeriesSplit CV.")

    majority = DummyClassifier(strategy="most_frequent")
    majority.fit(X_train, y_train)
    majority_accuracy = accuracy_score(y_test, majority.predict(X_test))
    print(f"Majority-class baseline: {majority_accuracy * 100:.2f}%")

    cv_results = {}
    fitted_candidates = {}
    for name, candidate in candidates.items():
        scores = cross_val_score(candidate, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=None)
        candidate.fit(X_train, y_train)
        cv_results[name] = {
            "cv_mean_accuracy": round(float(scores.mean()), 4),
            "cv_std": round(float(scores.std()), 4),
        }
        fitted_candidates[name] = candidate
        print(f"{name}: CV {scores.mean() * 100:.2f}% (+/- {scores.std() * 100:.2f}%)")

    best_name = max(cv_results, key=lambda name: cv_results[name]["cv_mean_accuracy"])
    best_model = fitted_candidates[best_name]
    candidate_accuracy = accuracy_score(y_test, best_model.predict(X_test))

    previous_accuracy = None
    previous_comparison_status = "no previous model"
    previous_protocol_is_comparable = False
    if os.path.exists(REPORT_PATH):
        try:
            with open(REPORT_PATH, "r", encoding="utf-8") as handle:
                previous_report = json.load(handle)
            previous_split = str(previous_report.get("split", "")).lower()
            previous_protocol_is_comparable = previous_split.startswith("chronological first 80%")
        except (OSError, ValueError, TypeError):
            previous_protocol_is_comparable = False

    if os.path.exists(MODEL_PATH) and previous_protocol_is_comparable:
        try:
            previous_model = joblib.load(MODEL_PATH)
            previous_accuracy = accuracy_score(y_test, previous_model.predict(X_test[FEATURES]))
            previous_comparison_status = "compared on the same chronological holdout"
        except Exception as exc:
            previous_comparison_status = f"previous model comparison failed: {exc}"
    elif os.path.exists(MODEL_PATH):
        previous_comparison_status = (
            "previous model was trained/evaluated with a different split protocol; "
            "retraining on chronological training data"
        )

    promoted = previous_accuracy is None or candidate_accuracy > previous_accuracy
    if promoted:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(best_model, MODEL_PATH)
        print(f"Saved {best_name} as the active model.")
    else:
        print(f"Kept existing model ({previous_accuracy * 100:.2f}% holdout); candidate did not improve it.")

    active_model = best_model if promoted else joblib.load(MODEL_PATH)
    y_pred = active_model.predict(X_test[FEATURES])
    active_accuracy = accuracy_score(y_test, y_pred)
    report = {
        "generated_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "dataset_rows_used": int(len(X)),
        "signal_class_distribution": {SIGNAL_NAMES[int(label)]: int(count) for label, count in y.value_counts().sort_index().items()},
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "split": "chronological first 80% for training, final 20% for held-out testing",
        "cross_validation_method": "training-only expanding-window TimeSeriesSplit (5 folds)",
        "majority_baseline_accuracy": round(float(majority_accuracy), 4),
        "previous_model_holdout_accuracy": None if previous_accuracy is None else round(float(previous_accuracy), 4),
        "previous_model_comparison_status": previous_comparison_status,
        "best_candidate_by_training_cv": best_name,
        "selected_estimator": type(best_model).__name__,
        "feature_importance": (
            {name: round(float(score), 6) for name, score in sorted(
                zip(FEATURES, best_model.feature_importances_), key=lambda item: item[1], reverse=True
            )}
            if hasattr(best_model, "feature_importances_") else {}
        ),
        "candidate_holdout_accuracy": round(float(candidate_accuracy), 4),
        "active_model_holdout_accuracy": round(float(active_accuracy), 4),
        "promoted_candidate": bool(promoted),
        "cross_validation": cv_results,
        "classification_report": classification_report(
            y_test, y_pred, labels=[0, 1, 2], target_names=SIGNAL_NAMES,
            output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=[0, 1, 2]).tolist(),
        "note": "This evaluates the tabular signal_status labels only. It is not live-camera or emergency-detection accuracy."
    }
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"Active model holdout accuracy: {active_accuracy * 100:.2f}%")
    print(classification_report(y_test, y_pred, labels=[0, 1, 2], target_names=SIGNAL_NAMES, zero_division=0))
    print(f"Detailed report: {REPORT_PATH}")
    if active_accuracy <= majority_accuracy + 0.05:
        print("WARNING: Accuracy is close to the majority baseline. Improve the dataset labels/features before expecting a reliable gain.")


if __name__ == "__main__":
    main()
