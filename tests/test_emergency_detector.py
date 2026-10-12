import unittest

from src.emergency_detector import EmergencyDetector

class EmergencyDetectorUnitTests(unittest.TestCase):
    def test_only_two_categories_can_trigger_emergency_priority(self):
        self.assertEqual(EmergencyDetector.EMERGENCY_LABELS, {"ambulance", "fire truck"})
        for label in ("bus", "truck", "van", "car", "police car"):
            with self.subTest(label=label):
                self.assertNotIn(label, EmergencyDetector.EMERGENCY_LABELS)

    def test_emergency_prompts_use_one_canonical_name_per_category(self):
        self.assertEqual(
            EmergencyDetector.PROMPT_CLASSES,
            ["ambulance", "fire truck", "bus", "truck", "van", "car", "police car"],
        )

    def test_default_emergency_confidence_is_not_forced_to_very_low_value(self):
        detector = EmergencyDetector()
        self.assertAlmostEqual(detector.confidence, 0.18)

    def test_only_ambulance_and_fire_vehicle_labels_are_emergency(self):
        self.assertEqual(EmergencyDetector._label("emergency ambulance"), "ambulance")
        self.assertEqual(EmergencyDetector._label("fire engine"), "fire truck")
        self.assertEqual(EmergencyDetector._label("police car"), "car")
        self.assertEqual(EmergencyDetector._label("city bus"), "bus")

    def test_ordinary_vehicles_and_police_never_map_to_emergency_priority(self):
        for label, expected in (
            ("police car", "car"),
            ("city bus", "bus"),
            ("delivery truck", "truck"),
            ("passenger van", "van"),
            ("fire engine", "fire truck"),
            ("ambulance with red cross markings", "ambulance"),
        ):
            with self.subTest(label=label):
                self.assertEqual(EmergencyDetector._label(label), expected)
                if expected not in EmergencyDetector.EMERGENCY_LABELS:
                    self.assertNotIn(expected, EmergencyDetector.EMERGENCY_LABELS)

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

    def test_emergency_candidate_must_overlap_a_base_vehicle_box(self):
        vehicle_boxes = [{"box": [100, 100, 220, 200]}]
        self.assertTrue(
            EmergencyDetector._matches_vehicle_box([110, 110, 210, 190], vehicle_boxes)
        )
        self.assertFalse(
            EmergencyDetector._matches_vehicle_box([300, 300, 400, 400], vehicle_boxes)
        )

    def test_vehicle_overlap_filter_allows_full_frame_fallback_without_boxes(self):
        self.assertTrue(
            EmergencyDetector._matches_vehicle_box([10, 10, 80, 80], [])
        )

    def test_direction_uses_frame_quadrants(self):
        self.assertEqual(EmergencyDetector._direction(90, 50, 200, 100), "West")
        self.assertEqual(EmergencyDetector._direction(150, 50, 200, 100), "East")
        self.assertEqual(EmergencyDetector._direction(100, 10, 200, 100), "North")
        self.assertEqual(EmergencyDetector._direction(100, 90, 200, 100), "South")

if __name__ == "__main__":
    unittest.main()
