"""Compare traffic-signal classifiers on a fixed holdout split.

Run from the repository root:
    python src/train_model.py

The repository's model-training GitHub Actions workflow is manual-dispatch only and uploads trained model artifacts without pushing model binaries to the branch.

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
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    GradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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


def target_diagnostics(X, y):
    encoded_target = y.astype(float)
    correlations = {}
    for feature in FEATURES:
        values = X[feature].astype(float)
        if values.nunique() < 2 or encoded_target.nunique() < 2:
            correlations[feature] = None
            continue
        correlation = values.corr(encoded_target)
        correlations[feature] = (
            round(float(correlation), 4) if pd.notna(correlation) else None
        )

    class_means = {}
    for label, name in enumerate(SIGNAL_NAMES):
        subset = X.loc[y == label]
        class_means[name] = {
            feature: round(float(subset[feature].mean()), 3)
            for feature in (
                "traffic_volume",
                "avg_vehicle_speed",
                "vehicle_count_cars",
                "vehicle_count_trucks",
                "vehicle_count_bikes",
            )
        }

    valid_correlations = [
        abs(value) for value in correlations.values() if value is not None
    ]
    maximum_absolute_correlation = (
        round(max(valid_correlations), 4) if valid_correlations else None
    )
    warning = None
    if maximum_absolute_correlation is not None and maximum_absolute_correlation < 0.10:
        warning = (
            "The supplied numeric features have very weak individual linear "
            "association with the encoded signal labels. A model score near the "
            "majority baseline is expected until independently verified signal "
            "phase and traffic observations are collected."
        )

    return {
        "pearson_correlation_with_encoded_signal_label": correlations,
        "class_feature_means": class_means,
        "maximum_absolute_feature_correlation": maximum_absolute_correlation,
        "warning": warning,
    }


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
        "SVM_RBF": make_pipeline(
            StandardScaler(),
            SVC(C=1.0, kernel="rbf", class_weight="balanced")
        ),
        "KNN_11": make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=11, weights="distance")
        ),
        "LogisticRegression": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced")
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
        scores = cross_val_score(
            candidate, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=None
        )
        candidate.fit(X_train, y_train)
        cv_results[name] = {
            "cv_mean_accuracy": round(float(scores.mean()), 4),
            "cv_std": round(float(scores.std()), 4),
        }
        fitted_candidates[name] = candidate
        print(f"{name}: CV accuracy {scores.mean() * 100:.2f}% (+/- {scores.std() * 100:.2f}%)")

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
            previous_protocol_is_comparable = (
                previous_split.startswith("chronological first 80%")
                and previous_report.get("evaluation_protocol_version") == "chronological_holdout_v2"
                and previous_report.get("active_model_evaluation_status")
                == "validated_chronological_holdout"
            )
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

    baseline_margin = candidate_accuracy - majority_accuracy
    minimum_baseline_margin = 0.01
    promotion_decision = ""

    if baseline_margin < minimum_baseline_margin:
        promoted = False
        promotion_decision = (
            "Candidate was not promoted because it did not beat the majority "
            "baseline by at least 1 percentage point."
        )
    elif os.path.exists(MODEL_PATH) and previous_accuracy is None:
        promoted = False
        promotion_decision = (
            "Candidate was not promoted because the existing model could not "
            "be compared under the same evaluation protocol. The existing "
            "model was preserved."
        )
    elif previous_accuracy is not None and candidate_accuracy <= previous_accuracy + 0.005:
        promoted = False
        promotion_decision = (
            "Candidate was not promoted because it did not beat the comparable "
            "existing model by at least 0.5 percentage points."
        )
    else:
        promoted = True
        promotion_decision = (
            "Candidate cleared the majority-baseline and model-comparison gates."
        )

    if promoted:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(best_model, MODEL_PATH)
        print(f"Saved {best_name} as the active model.")
    else:
        if os.path.exists(MODEL_PATH):
            print("Kept the existing model. " + promotion_decision)
        else:
            raise RuntimeError(
                "No existing model is available and the candidate failed the "
                "majority-baseline promotion gate. No model was saved."
            )

    candidate_prediction = best_model.predict(X_test)
    candidate_macro_f1 = f1_score(
        y_test, candidate_prediction, labels=[0, 1, 2],
        average="macro", zero_division=0
    )

    active_accuracy = None
    active_macro_f1 = None
    active_model_evaluation_status = "unavailable_legacy_model_not_evaluated"
    active_prediction = None
    if promoted:
        active_accuracy = candidate_accuracy
        active_macro_f1 = candidate_macro_f1
        active_prediction = candidate_prediction
        active_model_evaluation_status = "validated_chronological_holdout"
    elif previous_accuracy is not None:
        previous_model = joblib.load(MODEL_PATH)
        active_prediction = previous_model.predict(X_test[FEATURES])
        active_accuracy = accuracy_score(y_test, active_prediction)
        active_macro_f1 = f1_score(
            y_test, active_prediction, labels=[0, 1, 2],
            average="macro", zero_division=0
        )
        active_model_evaluation_status = "validated_chronological_holdout"

    report = {
        "generated_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "evaluation_protocol_version": "chronological_holdout_v2",
        "dataset_rows_used": int(len(X)),
        "signal_class_distribution": {SIGNAL_NAMES[int(label)]: int(count) for label, count in y.value_counts().sort_index().items()},
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "split": "chronological first 80% for training, final 20% for held-out testing",
        "cross_validation_method": "training-only expanding-window TimeSeriesSplit (5 folds)",
        "majority_baseline_accuracy": round(float(majority_accuracy), 4),
        "candidate_minus_majority_baseline": round(float(baseline_margin), 4),
        "minimum_baseline_margin_for_promotion": minimum_baseline_margin,
        "promotion_decision": promotion_decision,
        "target_diagnostics": target_diagnostics(X, y),
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
        "candidate_holdout_macro_f1": round(float(candidate_macro_f1), 4),
        "active_model_holdout_accuracy": (
            None if active_accuracy is None else round(float(active_accuracy), 4)
        ),
        "active_model_holdout_macro_f1": (
            None if active_macro_f1 is None else round(float(active_macro_f1), 4)
        ),
        "active_model_evaluation_status": active_model_evaluation_status,
        "promoted_candidate": bool(promoted),
        "cross_validation": cv_results,
        "candidate_classification_report": classification_report(
            y_test, candidate_prediction, labels=[0, 1, 2],
            target_names=SIGNAL_NAMES, output_dict=True, zero_division=0
        ),
        "candidate_confusion_matrix": confusion_matrix(
            y_test, candidate_prediction, labels=[0, 1, 2]
        ).tolist(),
        "note": (
            "Candidate accuracy is measured only after fitting on the first 80% "
            "of timestamps. A legacy model trained with a different split is not "
            "evaluated against this holdout because it may already have seen test "
            "records. The score is tabular signal-label accuracy, not live-camera "
            "detection or emergency-detection accuracy."
        )
    }
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"Best candidate by training CV: {best_name}")
    print(f"Candidate chronological holdout accuracy: {candidate_accuracy * 100:.2f}%")
    print(f"Candidate chronological holdout macro-F1: {candidate_macro_f1:.4f}")
    print(f"Majority baseline accuracy: {majority_accuracy * 100:.2f}%")
    if active_accuracy is None:
        print("Active model holdout accuracy: UNVERIFIED (legacy model used a different training split)")
    else:
        print(f"Active model chronological holdout accuracy: {active_accuracy * 100:.2f}%")
    print(classification_report(
        y_test, candidate_prediction, labels=[0, 1, 2],
        target_names=SIGNAL_NAMES, zero_division=0
    ))
    print(f"Detailed report: {REPORT_PATH}")
    if candidate_accuracy <= majority_accuracy + 0.05:
        print("WARNING: Candidate is close to the majority baseline. More reliable labels/features are required before promotion.")


if __name__ == "__main__":
    main()
