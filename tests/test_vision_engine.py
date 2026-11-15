import unittest
from unittest.mock import patch

import numpy as np

from src.vision_engine import VisionEngine


class FakeBox:
    def __init__(self, class_id, confidence, coordinates, track_id=None):
        self.cls = [class_id]
        self.conf = [confidence]
        self.xyxy = [coordinates]
        self.id = None if track_id is None else [track_id]


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

    def predict(self, frame, **kwargs):
        self.predict_kwargs = kwargs
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
        self.assertEqual(model.kwargs["max_det"], 1000)
        self.assertAlmostEqual(model.kwargs["iou"], 0.65)
        self.assertEqual(model.kwargs["classes"], [0, 1, 2, 3, 5, 7])

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

    def test_custom_model_counts_auto_rickshaws_separately(self):
        model = FakeModel(
            [
                FakeBox(0, 0.91, [10, 100, 110, 220]),
                FakeBox(1, 0.88, [120, 100, 220, 220]),
                FakeBox(2, 0.86, [230, 100, 360, 250]),
                FakeBox(3, 0.84, [370, 100, 520, 260]),
                FakeBox(4, 0.90, [520, 100, 630, 250]),
            ],
            names={
                0: "car", 1: "bike", 2: "bus", 3: "truck",
                4: "auto_rickshaw",
            },
        )
        engine = VisionEngine(model)
        metrics, _ = engine.process(np.zeros((360, 640, 3), dtype=np.uint8))

        self.assertEqual(metrics["cars"], 1)
        self.assertEqual(metrics["bikes"], 1)
        self.assertEqual(metrics["buses"], 1)
        self.assertEqual(metrics["trucks"], 1)
        self.assertEqual(metrics["auto_rickshaws"], 1)
        self.assertEqual(metrics["total_vehicles"], 5)

    def test_custom_vehicle_model_uses_coco_person_model_for_pedestrians(self):
        vehicle_model = FakeModel(
            [FakeBox(0, 0.91, [100, 100, 220, 220])],
            names={0: "car", 1: "bike", 2: "bus", 3: "truck", 4: "auto_rickshaw"},
        )
        person_model = FakeModel([FakeBox(0, 0.93, [20, 20, 70, 100])])
        engine = VisionEngine(vehicle_model, person_model=person_model)
        metrics, _ = engine.process(np.zeros((360, 640, 3), dtype=np.uint8))

        self.assertEqual(metrics["cars"], 1)
        self.assertEqual(metrics["pedestrians"], 1)
        self.assertEqual(metrics["vehicle_confidence"], 91.0)
        self.assertEqual(metrics["pedestrian_confidence"], 93.0)
        self.assertEqual(person_model.predict_kwargs["classes"], [0])

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

    def test_untracked_detections_do_not_create_fake_speed_or_waiting_time(self):
        model = FakeModel([
            FakeBox(2, 0.90, [100, 100, 220, 220]),
        ])
        engine = VisionEngine(model)
        first, _ = engine.process(np.zeros((360, 640, 3), dtype=np.uint8))
        second, _ = engine.process(np.zeros((360, 640, 3), dtype=np.uint8))

        self.assertEqual(first["avg_speed"], 0.0)
        self.assertEqual(second["avg_speed"], 0.0)
        self.assertEqual(second["avg_waiting_time"], 0.0)
        self.assertFalse(engine.history)

    def test_queue_length_uses_configurable_normalized_roi(self):
        model = FakeModel([
            FakeBox(2, 0.90, [100, 240, 200, 340]),
            FakeBox(2, 0.88, [10, 10, 90, 90]),
        ])
        with patch.dict("os.environ", {"TRAFFIC_QUEUE_ROI": "0.20,0.60,0.80,1.0"}):
            engine = VisionEngine(model)
            metrics, _ = engine.process(np.zeros((360, 640, 3), dtype=np.uint8))

        self.assertEqual(metrics["queue_length"], 1)
        self.assertEqual(metrics["queue_roi"], [0.2, 0.6, 0.8, 1.0])

    def test_invalid_queue_roi_uses_safe_default(self):
        with patch.dict("os.environ", {"TRAFFIC_QUEUE_ROI": "0.9,0.8,0.2,1.2"}):
            engine = VisionEngine(FakeModel([]))
        self.assertEqual(engine.queue_roi, (0.0, 0.60, 1.0, 1.0))

    def test_reset_tracking_clears_track_history(self):
        engine = VisionEngine(FakeModel([
            FakeBox(2, 0.90, [100, 100, 220, 220], track_id=7),
        ]))
        engine.process(np.zeros((360, 640, 3), dtype=np.uint8))
        self.assertIn(7, engine.history)
        engine.reset_tracking()
        self.assertFalse(engine.history)
        self.assertFalse(engine.last_seen)

    def test_dense_tile_inference_adds_missed_vehicles_and_removes_duplicates(self):
        base_model = FakeModel([
            FakeBox(2, 0.90, [10, 10, 55, 60]),
            FakeBox(2, 0.88, [80, 10, 125, 60]),
            FakeBox(2, 0.86, [150, 10, 195, 60]),
            FakeBox(2, 0.84, [220, 10, 265, 60]),
        ])

        class TileFakeModel:
            def __init__(self):
                self.tile_calls = 0
                self.predict_kwargs = None

            def predict(self, sources, **kwargs):
                self.predict_kwargs = kwargs
                self.tile_calls += 1
                return [
                    FakeResult([
                        FakeBox(2, 0.82, [300, 100, 360, 170]),
                        FakeBox(2, 0.80, [80, 10, 125, 60]),
                    ]),
                    FakeResult([]),
                    FakeResult([]),
                    FakeResult([]),
                ]

        tile_model = TileFakeModel()
        engine = VisionEngine(base_model, tile_model_factory=lambda: tile_model)
        metrics, _ = engine.process(np.zeros((360, 640, 3), dtype=np.uint8))

        self.assertEqual(tile_model.tile_calls, 1)
        self.assertEqual(tile_model.predict_kwargs["imgsz"], 960)
        self.assertEqual(metrics["primary_vehicle_detections"], 4)
        self.assertEqual(metrics["tiled_detections_added"], 1)
        self.assertTrue(metrics["tiled_inference_used"])
        self.assertEqual(metrics["cars"], 5)
        self.assertEqual(metrics["total_vehicles"], 5)

    def test_dense_tile_inference_skips_light_traffic(self):
        model = FakeModel([
            FakeBox(2, 0.90, [100, 100, 220, 220]),
            FakeBox(3, 0.88, [250, 100, 350, 220]),
        ])
        factory_calls = []
        engine = VisionEngine(
            model,
            tile_model_factory=lambda: factory_calls.append(True),
        )
        metrics, _ = engine.process(np.zeros((360, 640, 3), dtype=np.uint8))

        self.assertFalse(metrics["tiled_inference_used"])
        self.assertEqual(metrics["tiled_detections_added"], 0)
        self.assertEqual(metrics["total_vehicles"], 2)
        self.assertEqual(factory_calls, [])
        self.assertFalse(hasattr(model, "predict_kwargs"))

    def test_dense_tile_inference_can_be_disabled(self):
        model = FakeModel([
            FakeBox(2, 0.90, [10, 10, 55, 60]),
            FakeBox(2, 0.88, [80, 10, 125, 60]),
            FakeBox(2, 0.86, [150, 10, 195, 60]),
            FakeBox(2, 0.84, [220, 10, 265, 60]),
        ])
        with patch.dict("os.environ", {"TRAFFIC_TILED_INFERENCE": "false"}):
            engine = VisionEngine(model)
            metrics, _ = engine.process(np.zeros((360, 640, 3), dtype=np.uint8))

        self.assertFalse(metrics["tiled_inference_used"])
        self.assertEqual(metrics["total_vehicles"], 4)

    def test_invalid_environment_settings_fall_back_safely(self):
        with patch.dict("os.environ", {
            "TRAFFIC_DETECTION_CONF": "bad",
            "TRAFFIC_IMAGE_SIZE": "bad",
        }):
            engine = VisionEngine(FakeModel([]))
        self.assertAlmostEqual(engine.confidence_threshold, 0.18)
        self.assertEqual(engine.image_size, 1280)


if __name__ == "__main__":
    unittest.main()
