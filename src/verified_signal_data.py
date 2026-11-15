import csv
import os
import threading
from datetime import datetime, timezone


class VerifiedSignalDataStore:
    """Store camera features only after a person verifies the matching signal label."""

    SIGNALS = {"Red", "Yellow", "Green", "All Red"}
    DENSITIES = {"NO TRAFFIC", "LOW", "MEDIUM", "HIGH", "VERY HIGH"}
    DIRECTIONS = {"North", "South", "East", "West"}
    EMERGENCIES = {"none", "ambulance", "fire_truck"}

    NUMERIC_FIELDS = (
        "cars", "bikes", "buses", "trucks", "auto_rickshaws",
        "total_vehicles", "pedestrians", "avg_speed", "queue_length",
        "avg_waiting_time", "north_vehicles", "south_vehicles",
        "east_vehicles", "west_vehicles", "north_pedestrians",
        "south_pedestrians", "east_pedestrians", "west_pedestrians",
        "input_frame_width", "input_frame_height",
    )
    FIELDS = (
        "timestamp_utc", "scene_id", "session_id", "camera_id",
        "camera_direction", "actual_signal", "actual_density",
        "actual_direction", "actual_emergency_type",
    ) + NUMERIC_FIELDS

    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.lock = threading.Lock()
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        if not os.path.exists(csv_path):
            with open(csv_path, "w", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=self.FIELDS).writeheader()

    @staticmethod
    def _number(value):
        try:
            number = float(value)
            if number != number or number in (float("inf"), float("-inf")):
                return 0
            return round(number, 3)
        except (TypeError, ValueError, OverflowError):
            return 0

    def record(self, scene_id, camera_id, features, actual):
        scene_id = str(scene_id or "").strip()
        if not scene_id:
            raise ValueError("A unique camera frame ID is required.")

        signal = str(actual.get("signal", "")).strip()
        density = str(actual.get("density", "")).strip().upper()
        direction = str(actual.get("direction", "")).strip().title()
        emergency = str(actual.get("emergency_type", "")).strip().lower().replace(" ", "_")

        if signal not in self.SIGNALS:
            raise ValueError("A verified signal label is required.")
        if density not in self.DENSITIES:
            raise ValueError("A verified traffic-density label is required.")
        if direction not in self.DIRECTIONS:
            raise ValueError("A verified priority direction is required.")
        if emergency not in self.EMERGENCIES:
            raise ValueError("A verified emergency label is required.")

        features = features if isinstance(features, dict) else {}
        directions = features.get("directions") or {}
        pedestrian_directions = features.get("pedestrian_directions") or {}
        row = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "scene_id": scene_id,
            "session_id": scene_id.split(":", 1)[0],
            "camera_id": str(camera_id or ""),
            "camera_direction": str(features.get("camera_direction", "")).strip(),
            "actual_signal": signal,
            "actual_density": density,
            "actual_direction": direction,
            "actual_emergency_type": emergency,
            "cars": self._number(features.get("cars")),
            "bikes": self._number(features.get("bikes")),
            "buses": self._number(features.get("buses")),
            "trucks": self._number(features.get("trucks")),
            "auto_rickshaws": self._number(features.get("auto_rickshaws")),
            "total_vehicles": self._number(features.get("total_vehicles")),
            "pedestrians": self._number(features.get("pedestrians")),
            "avg_speed": self._number(features.get("avg_speed")),
            "queue_length": self._number(features.get("queue_length")),
            "avg_waiting_time": self._number(features.get("avg_waiting_time")),
            "north_vehicles": self._number(directions.get("North")),
            "south_vehicles": self._number(directions.get("South")),
            "east_vehicles": self._number(directions.get("East")),
            "west_vehicles": self._number(directions.get("West")),
            "north_pedestrians": self._number(pedestrian_directions.get("North")),
            "south_pedestrians": self._number(pedestrian_directions.get("South")),
            "east_pedestrians": self._number(pedestrian_directions.get("East")),
            "west_pedestrians": self._number(pedestrian_directions.get("West")),
            "input_frame_width": self._number(features.get("input_frame_width")),
            "input_frame_height": self._number(features.get("input_frame_height")),
        }

        with self.lock:
            with open(self.csv_path, newline="", encoding="utf-8") as handle:
                existing = csv.DictReader(handle)
                if any(item.get("scene_id") == scene_id for item in existing):
                    raise ValueError("This camera frame is already in the verified training dataset.")
            with open(self.csv_path, "a", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=self.FIELDS).writerow(row)

        return self.summary()

    def summary(self):
        with self.lock:
            try:
                with open(self.csv_path, newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
            except (OSError, csv.Error):
                rows = []

        class_counts = {name: 0 for name in sorted(self.SIGNALS)}
        for row in rows:
            label = row.get("actual_signal")
            if label in class_counts:
                class_counts[label] += 1
        return {
            "total_samples": len(rows),
            "sessions": len({row.get("session_id", "") for row in rows if row.get("session_id")}),
            "class_counts": class_counts,
        }
