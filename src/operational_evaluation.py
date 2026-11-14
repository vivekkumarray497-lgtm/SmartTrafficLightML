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
    ALLOWED = {
        "density": {"NO TRAFFIC", "LOW", "MEDIUM", "HIGH", "VERY HIGH"},
        "direction": {"North", "South", "East", "West"},
        "emergency_type": {"none", "ambulance", "fire_truck"},
        "signal": {"Red", "Yellow", "Green", "All Red"},
    }

    def __init__(
        self,
        csv_path,
        report_path,
        minimum_samples=50,
        minimum_sample_interval_seconds=5.0,
    ):
        self.csv_path = csv_path
        self.report_path = report_path
        self.minimum_samples = max(1, int(minimum_samples))
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

        # Rows from the older schema have no scene_id and cannot be verified
        # as unique camera frames. Keep them in the CSV for audit history,
        # but exclude them from the measured validation sample count.
        valid_rows = [row for row in rows if str(row.get("scene_id", "")).strip()]
        total = len(valid_rows)
        correct = sum(
            str(row.get("exact_match", "0")) == "1" for row in valid_rows
        )
        enough_samples = total >= self.minimum_samples
        result = {
            "accuracy": round(correct / total * 100, 1) if enough_samples and total else None,
            "correct_samples": int(correct),
            "total_samples": total,
            "minimum_samples": self.minimum_samples,
            "remaining_samples": max(0, self.minimum_samples - total),
            "status": "available" if enough_samples else "collecting",
            "metric": "Human-verified end-to-end exact-match accuracy",
            "component_accuracy": {},
            "emergency_positive_samples": 0,
            "emergency_negative_samples": 0,
            "emergency_accuracy": None,
            "emergency_validation_status": "needs at least 5 positive and 5 negative labelled samples",
        }

        for field in self.FIELDS:
            if not enough_samples:
                result["component_accuracy"][field] = None
            else:
                matches = sum(
                    row.get(f"predicted_{field}") == row.get(f"actual_{field}")
                    for row in valid_rows
                )
                result["component_accuracy"][field] = round(matches / total * 100, 1)

        positive = sum(
            row.get("actual_emergency_type") in {"ambulance", "fire_truck"}
            for row in valid_rows
        )
        negative = sum(
            row.get("actual_emergency_type") == "none"
            for row in valid_rows
        )
        result["emergency_positive_samples"] = int(positive)
        result["emergency_negative_samples"] = int(negative)
        if positive >= 5 and negative >= 5:
            matches = sum(
                row.get("predicted_emergency_type") == row.get("actual_emergency_type")
                for row in valid_rows
            )
            result["emergency_accuracy"] = round(matches / total * 100, 1)
            result["emergency_validation_status"] = "available"

        result["limitations"] = [
            "Ground truth is manually entered by a reviewer who independently checks the scene.",
            "Each unique captured camera frame can be submitted only once.",
            f"Validation samples are separated by at least {self.minimum_sample_interval_seconds:g} seconds.",
            "Overall accuracy is strict exact match across density, priority direction, emergency class and signal phase.",
            "The result applies only to the labelled validation samples and conditions represented.",
            "Collect diverse scenes; avoid near-identical consecutive frames.",
            "This offline validation metric does not certify road safety or authorize real-world deployment.",
        ]
        return result
