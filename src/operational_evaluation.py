import csv
import json
import os
import threading
from datetime import datetime, timezone


class OperationalEvaluator:
    FIELDS = ("density", "direction", "emergency_type", "signal")
    ALLOWED = {
        "density": {"NO TRAFFIC", "LOW", "MEDIUM", "HIGH", "VERY HIGH"},
        "direction": {"North", "South", "East", "West"},
        "emergency_type": {"none", "ambulance", "fire_truck"},
        "signal": {"Red", "Yellow", "Green"},
    }

    def __init__(self, csv_path, report_path, minimum_samples=50):
        self.csv_path = csv_path
        self.report_path = report_path
        self.minimum_samples = max(1, int(minimum_samples))
        self.lock = threading.Lock()
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.report_path), exist_ok=True)
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=self._columns()).writeheader()

    @classmethod
    def _columns(cls):
        columns = ["timestamp"]
        for field in cls.FIELDS:
            columns.extend((f"predicted_{field}", f"actual_{field}"))
        columns.append("exact_match")
        return columns

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

    def record(self, prediction, actual):
        row = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        matches = []
        for field in self.FIELDS:
            predicted_value = self._normalize(field, prediction.get(field))
            actual_value = self._normalize(field, actual.get(field))
            row[f"predicted_{field}"] = predicted_value
            row[f"actual_{field}"] = actual_value
            matches.append(predicted_value == actual_value)
        row["exact_match"] = int(all(matches))

        with self.lock:
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
        except OSError:
            rows = []

        total = len(rows)
        correct = sum(str(row.get("exact_match", "0")) == "1" for row in rows)
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
                    for row in rows
                )
                result["component_accuracy"][field] = round(matches / total * 100, 1)

        positive = sum(row.get("actual_emergency_type") in {"ambulance", "fire_truck"} for row in rows)
        negative = sum(row.get("actual_emergency_type") == "none" for row in rows)
        result["emergency_positive_samples"] = int(positive)
        result["emergency_negative_samples"] = int(negative)
        if positive >= 5 and negative >= 5:
            matches = sum(
                row.get("predicted_emergency_type") == row.get("actual_emergency_type")
                for row in rows
            )
            result["emergency_accuracy"] = round(matches / total * 100, 1)
            result["emergency_validation_status"] = "available"

        result["limitations"] = [
            "Ground truth is manually entered by a reviewer who independently checks the scene.",
            "Overall accuracy is strict exact match across density, priority direction, emergency class and signal phase.",
            "The result applies only to the labelled validation samples and conditions represented.",
            "Collect diverse scenes; avoid repeatedly labelling near-identical consecutive frames.",
            "This offline validation metric does not certify road safety or authorize real-world deployment.",
        ]
        return result
