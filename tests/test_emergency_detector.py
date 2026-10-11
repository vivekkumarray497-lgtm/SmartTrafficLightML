import unittest

from src.emergency_detector import EmergencyDetector


class EmergencyDetectorUnitTests(unittest.TestCase):
    def test_only_ambulance_and_fire_vehicle_labels_are_emergency(self):
        self.assertEqual(EmergencyDetector._label("emergency ambulance"), "ambulance")
        self.assertEqual(EmergencyDetector._label("fire engine"), "fire truck")
        self.assertEqual(EmergencyDetector._label("police car"), "car")
        self.assertEqual(EmergencyDetector._label("city bus"), "bus")

    def test_overlapping_full_frame_and_crop_boxes_are_deduplicated(self):
        candidates = [
            {"label": "ambulance", "confidence": 0.31, "direction": "East", "box": [10, 10, 110, 100]},
            {"label": "ambulance", "confidence": 0.46, "direction": "East", "box": [15, 12, 108, 98]},
        ]
        result = EmergencyDetector._deduplicate(candidates)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0]["confidence"], 0.46)

    def test_separate_vehicle_boxes_are_preserved(self):
        candidates = [
            {"label": "ambulance", "confidence": 0.41, "direction": "West", "box": [0, 0, 50, 50]},
            {"label": "fire truck", "confidence": 0.38, "direction": "East", "box": [120, 0, 190, 60]},
        ]
        self.assertEqual(len(EmergencyDetector._deduplicate(candidates)), 2)

    def test_direction_uses_frame_quadrants(self):
        self.assertEqual(EmergencyDetector._direction(90, 50, 200, 100), "West")
        self.assertEqual(EmergencyDetector._direction(150, 50, 200, 100), "East")
        self.assertEqual(EmergencyDetector._direction(100, 10, 200, 100), "North")
        self.assertEqual(EmergencyDetector._direction(100, 90, 200, 100), "South")


if __name__ == "__main__":
    unittest.main()
