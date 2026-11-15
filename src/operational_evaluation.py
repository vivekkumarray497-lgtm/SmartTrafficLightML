import csv
import json
import math
import os
import threading
from datetime import datetime, timezone


class DuplicateSceneError(ValueError):
    """Raised when a camera frame has already been used for validation."""


class SampleIntervalError(ValueError):
    """Raised when validation frames are submitted too close together."""


class OperationalEvaluator:
    FIELDS = ("density", "direction", "emergency_type", "signal")
    EMERGENCY_CLASSES = ("none", "ambulance", "fire_truck")
    EMERGENCY_LABELS = {
        "none": "No emergency",
        "ambulance": "Ambulance",
        "fire_truck": "Fire truck",
    }
    ALLOWED = {
        "density": {"NO TRAFFIC", "LOW", "MEDIUM", "HIGH", "VERY HIGH"},
        "direction": {"North", "South", "East", "West"},
        "emergency_type": set(EMERGENCY_CLASSES),
        "signal": {"Red", "Yellow", "Green", "All Red"},
    }

    def __init__(
        self,
        csv_path,
        report_path,
        minimum_samples=50,
        minimum_sample_interval_seconds=5.0,
        minimum_component_samples=10,
        minimum_emergency_class_samples=10,
    ):
        self.csv_path = csv_path
        self.report_path = report_path
        self.minimum_samples = max(1, int(minimum_samples))
        self.minimum_component_samples = max(1, int(minimum_component_samples))
        self.minimum_emergency_class_samples = max(
            1, int(minimum_emergency_class_samples)
        )
        self.minimum_sample_interval_seconds = max(
            0.0, float(minimum_sample_interval_seconds)
        )
        self.lock = threading.Lock()
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.report_path), exist_ok=True)
        if not os.path.exists(self.csv_path):
            self._write_rows([])
        else:
            self._migrate_schema_if_needed()

    @classmethod
    def _columns(cls):
        columns = ["timestamp", "scene_id"]
        for field in cls.FIELDS:
            columns.extend((f"predicted_{field}", f"actual_{field}"))
        columns.append("exact_match")
        return columns

    def _write_rows(self, rows):
        with open(self.csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._columns())
            writer.writeheader()
            writer.writerows(rows)

    def _migrate_schema_if_needed(self):
        try:
            with open(self.csv_path, newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                old_fields = reader.fieldnames or []
                rows = list(reader)
            if old_fields != self._columns():
                self._write_rows(rows)
        except (OSError, csv.Error):
            self._write_rows([])

    @classmethod
    def _normalize(cls, field, value):
        value = str(value or "").strip()
        if field == "density":
            value = value.upper()
        elif field in {"direction", "signal"}:
            value = value.title()
        else:
            value = value.lower().replace(" ", "_")
        if value not in cls.ALLOWED[field]:
            raise ValueError(f"Invalid {field} value: {value or '(empty)'}")
        return value

    @staticmethod
    def _wilson_interval(correct, total, z=1.96):
        """Return a two-sided 95% Wilson score interval in percent."""
        if total <= 0:
            return None
        proportion = correct / total
        z_squared = z * z
        denominator = 1 + z_squared / total
        center = (proportion + z_squared / (2 * total)) / denominator
        margin = (
            z
            * math.sqrt(
                (proportion * (1 - proportion) / total)
                + (z_squared / (4 * total * total))
            )
            / denominator
        )
        lower = max(0.0, center - margin) * 100
        upper = min(1.0, center + margin) * 100
        return {"lower": round(lower, 1), "upper": round(upper, 1)}

    @classmethod
    def _is_complete_row(cls, row):
        if not str(row.get("scene_id", "")).strip():
            return False
        for field in cls.FIELDS:
            if row.get(f"predicted_{field}") not in cls.ALLOWED[field]:
                return False
            if row.get(f"actual_{field}") not in cls.ALLOWED[field]:
                return False
        return True

    def record(self, prediction, actual, scene_id):
        scene_id = str(scene_id or "").strip()
        if not scene_id:
            raise ValueError("A unique camera frame ID is required for validation.")

        now = datetime.now(timezone.utc)
        row = {
            "timestamp": now.isoformat(timespec="seconds"),
            "scene_id": scene_id,
        }
        matches = []
        for field in self.FIELDS:
            predicted_value = self._normalize(field, prediction.get(field))
            actual_value = self._normalize(field, actual.get(field))
            row[f"predicted_{field}"] = predicted_value
            row[f"actual_{field}"] = actual_value
            matches.append(predicted_value == actual_value)
        row["exact_match"] = int(all(matches))

        with self.lock:
            with open(self.csv_path, newline="", encoding="utf-8") as handle:
                existing_rows = list(csv.DictReader(handle))
            if any(existing.get("scene_id") == scene_id for existing in existing_rows):
                raise DuplicateSceneError(
                    "This exact camera frame has already been submitted. Capture a new scene."
                )

            if self.minimum_sample_interval_seconds > 0:
                previous_times = []
                for existing in existing_rows:
                    if not str(existing.get("scene_id", "")).strip():
                        continue
                    raw_timestamp = str(existing.get("timestamp", "")).strip()
                    if not raw_timestamp:
                        continue
                    try:
                        previous = datetime.fromisoformat(
                            raw_timestamp.replace("Z", "+00:00")
                        )
                        if previous.tzinfo is None:
                            previous = previous.replace(tzinfo=timezone.utc)
                        previous_times.append(previous.astimezone(timezone.utc))
                    except ValueError:
                        continue

                if previous_times:
                    elapsed = (now - max(previous_times)).total_seconds()
                    if elapsed < self.minimum_sample_interval_seconds:
                        remaining = math.ceil(
                            self.minimum_sample_interval_seconds - elapsed
                        )
                        raise SampleIntervalError(
                            f"Wait at least {remaining} more second(s) before "
                            "submitting another validation scene."
                        )

            with open(self.csv_path, "a", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=self._columns()).writerow(row)
            summary = self._summary_unlocked()
            with open(self.report_path, "w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2)
        return summary

    def summary(self):
        with self.lock:
            return self._summary_unlocked()

    def _summary_unlocked(self):
        try:
            with open(self.csv_path, newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        except (OSError, csv.Error):
            rows = []

        # Older or partially migrated rows are retained for audit history but
        # excluded from scores unless all four prediction/ground-truth pairs exist.
        valid_rows = [row for row in rows if self._is_complete_row(row)]
        total = len(valid_rows)
        excluded = len(rows) - total
        matches_by_field = {
            field: sum(
                row.get(f"predicted_{field}") == row.get(f"actual_{field}")
                for row in valid_rows
            )
            for field in self.FIELDS
        }
        correct = sum(
            all(
                row.get(f"predicted_{field}") == row.get(f"actual_{field}")
                for field in self.FIELDS
            )
            for row in valid_rows
        )

        enough_samples = total >= self.minimum_samples
        enough_component_samples = total >= self.minimum_component_samples
        emergency_class_counts = {
            label: sum(row.get("actual_emergency_type") == label for row in valid_rows)
            for label in self.EMERGENCY_CLASSES
        }
        emergency_class_recall = {}
        emergency_class_precision = {}
        emergency_confusion_matrix = {
            actual_label: {
                predicted_label: 0 for predicted_label in self.EMERGENCY_CLASSES
            }
            for actual_label in self.EMERGENCY_CLASSES
        }

        for label in self.EMERGENCY_CLASSES:
            support = emergency_class_counts[label]
            true_positives = sum(
                row.get("actual_emergency_type") == label
                and row.get("predicted_emergency_type") == label
                for row in valid_rows
            )
            predicted_count = sum(
                row.get("predicted_emergency_type") == label for row in valid_rows
            )
            emergency_class_recall[label] = (
                round(true_positives / support * 100, 1) if support else None
            )
            emergency_class_precision[label] = (
                round(true_positives / predicted_count * 100, 1)
                if predicted_count else None
            )

        for row in valid_rows:
            actual_label = row.get("actual_emergency_type")
            predicted_label = row.get("predicted_emergency_type")
            emergency_confusion_matrix[actual_label][predicted_label] += 1

        enough_emergency_examples = all(
            emergency_class_counts[label] >= self.minimum_emergency_class_samples
            for label in self.EMERGENCY_CLASSES
        )
        emergency_macro_recall = (
            round(
                sum(emergency_class_recall[label] for label in self.EMERGENCY_CLASSES)
                / len(self.EMERGENCY_CLASSES),
                1,
            )
            if enough_emergency_examples else None
        )

        positive = emergency_class_counts["ambulance"] + emergency_class_counts["fire_truck"]
        negative = emergency_class_counts["none"]
        remaining_by_class = {
            label: max(
                0,
                self.minimum_emergency_class_samples - emergency_class_counts[label],
            )
            for label in self.EMERGENCY_CLASSES
        }

        result = {
            "accuracy": round(correct / total * 100, 1) if enough_samples and total else None,
            "accuracy_ci_95": (
                self._wilson_interval(correct, total) if enough_samples else None
            ),
            "correct_samples": int(correct),
            "total_samples": total,
            "excluded_incomplete_samples": excluded,
            "minimum_samples": self.minimum_samples,
            "remaining_samples": max(0, self.minimum_samples - total),
            "status": "available" if enough_samples else "collecting",
            "metric": "Human-reviewed strict end-to-end exact-match accuracy",
            "component_accuracy": {
                field: (
                    round(matches_by_field[field] / total * 100, 1)
                    if enough_component_samples else None
                )
                for field in self.FIELDS
            },
            "component_sample_count": total,
            "minimum_component_samples": self.minimum_component_samples,
            "component_status": (
                "available" if enough_component_samples else "collecting"
            ),
            "emergency_positive_samples": int(positive),
            "emergency_negative_samples": int(negative),
            "emergency_class_counts": emergency_class_counts,
            "emergency_class_recall": emergency_class_recall,
            "emergency_class_precision": emergency_class_precision,
            "emergency_confusion_matrix": emergency_confusion_matrix,
            # Kept for compatibility with the existing dashboard field.
            # This is macro recall, not ordinary accuracy dominated by the largest class.
            "emergency_accuracy": emergency_macro_recall,
            "emergency_macro_recall": emergency_macro_recall,
            "emergency_metric": "Three-class macro recall",
            "minimum_emergency_class_samples": self.minimum_emergency_class_samples,
            "emergency_remaining_by_class": remaining_by_class,
            "emergency_validation_status": (
                "available" if enough_emergency_examples else "collecting"
            ),
        }

        if enough_emergency_examples:
            result["emergency_validation_note"] = (
                "Macro recall averages recall for No emergency, Ambulance and Fire truck "
                "equally. Per-class recall and precision are available in the report."
            )
        else:
            coverage = ", ".join(
                f"{self.EMERGENCY_LABELS[label]} "
                f"{emergency_class_counts[label]}/{self.minimum_emergency_class_samples}"
                for label in self.EMERGENCY_CLASSES
            )
            result["emergency_validation_note"] = (
                "Collect the minimum examples for all three classes: " + coverage + "."
            )

        result["limitations"] = [
            "Ground truth is manually entered by a reviewer who independently checks the scene.",
            "Each complete, unique captured camera frame can be submitted only once.",
            f"Validation submissions are separated by at least {self.minimum_sample_interval_seconds:g} seconds.",
            f"Component indicators require at least {self.minimum_component_samples} complete labelled frames.",
            f"Overall exact-match accuracy requires at least {self.minimum_samples} complete labelled frames.",
            "The 95% Wilson interval communicates sampling uncertainty; it does not remove dataset bias.",
            "Unique frame IDs and time spacing do not guarantee independent or diverse scenes.",
            "Emergency macro recall is withheld until No emergency, Ambulance and Fire truck each have enough actual examples.",
            "The result applies only to the labelled validation samples and conditions represented.",
            "This offline validation metric does not certify road safety or authorize real-world deployment.",
        ]
        return result
