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
    def __init__(self, boxes):
        self.result = FakeResult(boxes)
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
