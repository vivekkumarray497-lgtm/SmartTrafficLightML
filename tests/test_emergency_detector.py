import unittest

from src.emergency_detector import EmergencyDetector

class EmergencyDetectorUnitTests(unittest.TestCase):
    def test_only_two_categories_can_trigger_emergency_priority(self):
        self.assertEqual(EmergencyDetector.EMERGENCY_LABELS, {"ambulance", "fire truck"})
        for label in ("bus", "truck", "van", "car", "police car"):
            with self.subTest(label=label):
                self.assertNotIn(label, EmergencyDetector.EMERGENCY_LABELS)

    def test_emergency_prompts_are_variants_of_only_allowed_categories(self):
        self.assertEqual(EmergencyDetector.PROMPT_CLASSES, ["ambulance", "fire truck"])
        self.assertIn("ambulance", EmergencyDetector.PROMPT_CLASSES)
        self.assertIn("fire truck", EmergencyDetector.PROMPT_CLASSES)
        for prompt in EmergencyDetector.PROMPT_CLASSES:
            with self.subTest(prompt=prompt):
                mapped = EmergencyDetector._label(prompt)
                self.assertIn(mapped, EmergencyDetector.EMERGENCY_LABELS)
        for label in ("bus", "city bus", "truck", "commercial goods truck",
                      "van", "passenger van", "car", "police car"):
            with self.subTest(label=label):
                self.assertNotIn(label, EmergencyDetector.PROMPT_CLASSES)

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

    def test_crop_inference_uses_lower_confidence_and_higher_resolution(self):
        class FakeModel:
            def __init__(self):
                self.kwargs = None

            def predict(self, image, **kwargs):
                self.kwargs = kwargs
                return [object()]

        detector = EmergencyDetector(confidence=0.18)
        detector.model = FakeModel()
        detector._predict("crop-image", confidence=min(detector.confidence, 0.07), imgsz=1024)
        self.assertAlmostEqual(detector.model.kwargs["conf"], 0.07)
        self.assertEqual(detector.model.kwargs["imgsz"], 1024)
        self.assertFalse(detector.model.kwargs["verbose"])

    def test_crop_selection_checks_multiple_vehicle_boxes_by_default_limit(self):
        boxes = [
            {"box": [0, 0, 20, 20]},
            {"box": [0, 0, 100, 100]},
            {"box": [0, 0, 60, 60]},
            {"box": [0, 0, 40, 40]},
            {"box": [0, 0, 30, 30]},
        ]
        selected = EmergencyDetector._select_crop_boxes(boxes, 4)
        self.assertEqual(len(selected), 4)
        areas = [
            (item["box"][2] - item["box"][0]) * (item["box"][3] - item["box"][1])
            for item in selected
        ]
        self.assertEqual(areas, sorted(areas, reverse=True))

    def test_bus_and_truck_candidates_get_emergency_recheck_priority(self):
        boxes = [
            {"class_id": 2, "box": [0, 0, 200, 200]},
            {"class_id": 5, "box": [0, 0, 40, 40]},
            {"class_id": 7, "box": [0, 0, 30, 30]},
        ]
        selected = EmergencyDetector._select_crop_boxes(boxes, 2)
        self.assertEqual([item["class_id"] for item in selected], [5, 7])

    def test_direction_uses_frame_quadrants(self):
        self.assertEqual(EmergencyDetector._direction(90, 50, 200, 100), "West")
        self.assertEqual(EmergencyDetector._direction(150, 50, 200, 100), "East")
        self.assertEqual(EmergencyDetector._direction(100, 10, 200, 100), "North")
        self.assertEqual(EmergencyDetector._direction(100, 90, 200, 100), "South")

if __name__ == "__main__":
    unittest.main()
