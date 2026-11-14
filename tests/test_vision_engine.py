import unittest
from unittest.mock import patch

import numpy as np

from src.vision_engine import VisionEngine


class FakeBox:
    def __init__(self, class_id, confidence, coordinates):
        self.cls = [class_id]
        self.conf = [confidence]
        self.xyxy = [coordinates]
        self.id = None


class FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes
        self.names = {0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


class FakeModel:
    def __init__(self, boxes, names=None):
        self.result = FakeResult(boxes)
        self.names = names
        self.kwargs = None

    def track(self, frame, **kwargs):
        self.kwargs = kwargs
        return [self.result]


class VisionEngineTests(unittest.TestCase):
    def test_vehicle_confidence_excludes_pedestrian_and_emergency_scores(self):
        model = FakeModel([
            FakeBox(0, 0.95, [10, 10, 50, 90]),
            FakeBox(2, 0.42, [100, 100, 220, 220]),
            FakeBox(5, 0.62, [260, 100, 420, 260]),
        ])
        engine = VisionEngine(model)
        metrics, _ = engine.process(np.zeros((360, 640, 3), dtype=np.uint8))

        self.assertEqual(metrics["total_vehicles"], 2)
        self.assertEqual(metrics["pedestrians"], 1)
        self.assertEqual(metrics["vehicle_confidence"], 52.0)
        self.assertEqual(metrics["live_confidence"], 52.0)
        self.assertEqual(metrics["live_confidence_samples"], 2)
        self.assertEqual(metrics["pedestrian_confidence"], 95.0)
        self.assertEqual(metrics["vehicle_class_confidence"], {"cars": 42.0, "buses": 62.0})

    def test_vehicle_inference_uses_recall_focused_settings(self):
        model = FakeModel([])
        with patch.dict("os.environ", {
            "TRAFFIC_DETECTION_CONF": "0.18",
            "TRAFFIC_IMAGE_SIZE": "960",
        }):
            engine = VisionEngine(model)
            engine.process(np.zeros((360, 640, 3), dtype=np.uint8))

        self.assertAlmostEqual(model.kwargs["conf"], 0.18)
        self.assertEqual(model.kwargs["imgsz"], 960)
        self.assertEqual(model.kwargs["max_det"], 500)
        self.assertEqual(model.kwargs["classes"], [0, 2, 3, 5, 7])

    def test_custom_four_class_model_maps_its_own_class_ids(self):
        model = FakeModel(
            [
                FakeBox(0, 0.91, [10, 100, 110, 220]),
                FakeBox(1, 0.88, [120, 100, 220, 220]),
                FakeBox(2, 0.86, [230, 100, 360, 250]),
                FakeBox(3, 0.84, [370, 100, 520, 260]),
            ],
            names={0: "car", 1: "bike", 2: "bus", 3: "truck"},
        )
        engine = VisionEngine(model)
        metrics, _ = engine.process(np.zeros((360, 640, 3), dtype=np.uint8))

        self.assertEqual(metrics["cars"], 1)
        self.assertEqual(metrics["bikes"], 1)
        self.assertEqual(metrics["buses"], 1)
        self.assertEqual(metrics["trucks"], 1)
        self.assertEqual(metrics["total_vehicles"], 4)
        self.assertEqual(model.kwargs["classes"], [0, 1, 2, 3])

    def test_custom_model_can_keep_person_class_separate(self):
        model = FakeModel(
            [
                FakeBox(0, 0.91, [10, 100, 110, 220]),
                FakeBox(4, 0.93, [120, 20, 170, 100]),
            ],
            names={0: "car", 1: "bike", 2: "bus", 3: "truck", 4: "person"},
        )
        engine = VisionEngine(model)
        metrics, _ = engine.process(np.zeros((360, 640, 3), dtype=np.uint8))

        self.assertEqual(metrics["cars"], 1)
        self.assertEqual(metrics["pedestrians"], 1)
        self.assertEqual(metrics["vehicle_confidence"], 91.0)
        self.assertEqual(metrics["pedestrian_confidence"], 93.0)
        self.assertEqual(model.kwargs["classes"], [0, 1, 2, 3, 4])
    def test_invalid_environment_settings_fall_back_safely(self):
        with patch.dict("os.environ", {
            "TRAFFIC_DETECTION_CONF": "bad",
            "TRAFFIC_IMAGE_SIZE": "bad",
        }):
            engine = VisionEngine(FakeModel([]))
        self.assertAlmostEqual(engine.confidence_threshold, 0.22)
        self.assertEqual(engine.image_size, 960)


if __name__ == "__main__":
    unittest.main()
