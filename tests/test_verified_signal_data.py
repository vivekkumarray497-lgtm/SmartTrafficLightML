import csv
import os
import tempfile
import unittest

from src.verified_signal_data import VerifiedSignalDataStore
from src.train_verified_signal_model import grouped_holdout, LABELS


class VerifiedSignalDataStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.temp.name, "verified.csv")
        self.store = VerifiedSignalDataStore(self.path)
        self.features = {
            "cars": 4,
            "bikes": 2,
            "buses": 1,
            "trucks": 0,
            "auto_rickshaws": 1,
            "total_vehicles": 8,
            "pedestrians": 3,
            "avg_speed": 12.5,
            "queue_length": 5,
            "avg_waiting_time": 2,
            "directions": {"North": 5, "South": 2, "East": 1, "West": 0},
            "pedestrian_directions": {"North": 1, "South": 1, "East": 1, "West": 0},
            "camera_direction": "North",
            "input_frame_width": 1280,
            "input_frame_height": 720,
        }
        self.actual = {
            "signal": "Red",
            "density": "HIGH",
            "direction": "North",
            "emergency_type": "none",
        }

    def tearDown(self):
        self.temp.cleanup()

    def test_saves_human_verified_signal_and_frame_features(self):
        summary = self.store.record("session-a:1", "camera-1", self.features, self.actual)
        self.assertEqual(summary["total_samples"], 1)
        self.assertEqual(summary["sessions"], 1)
        with open(self.path, newline="", encoding="utf-8") as handle:
            row = next(csv.DictReader(handle))
        self.assertEqual(row["session_id"], "session-a")
        self.assertEqual(row["actual_signal"], "Red")
        self.assertEqual(float(row["north_vehicles"]), 5)
        self.assertEqual(float(row["input_frame_width"]), 1280)

    def test_duplicate_frame_is_rejected(self):
        self.store.record("session-a:1", "camera-1", self.features, self.actual)
        with self.assertRaisesRegex(ValueError, "already in the verified training dataset"):
            self.store.record("session-a:1", "camera-1", self.features, self.actual)

    def test_invalid_signal_label_is_rejected(self):
        invalid = dict(self.actual)
        invalid["signal"] = "Unknown"
        with self.assertRaisesRegex(ValueError, "verified signal label"):
            self.store.record("session-a:1", "camera-1", self.features, invalid)
        self.assertEqual(self.store.summary()["total_samples"], 0)

    def test_session_split_keeps_sessions_separate(self):
        rows = []
        for session_index in range(8):
            for label in LABELS:
                rows.append((session_index, label))
        y = __import__("pandas").Series([label for _, label in rows])
        groups = __import__("pandas").Series([f"session-{index}" for index, _ in rows])
        train_indices, test_indices = grouped_holdout(None, y, groups)
        train_sessions = set(groups.iloc[train_indices])
        test_sessions = set(groups.iloc[test_indices])
        self.assertFalse(train_sessions.intersection(test_sessions))
        self.assertEqual(set(y.iloc[train_indices]), set(LABELS))
        self.assertEqual(set(y.iloc[test_indices]), set(LABELS))


if __name__ == "__main__":
    unittest.main()
