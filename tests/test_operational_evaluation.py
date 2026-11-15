import os
import tempfile
import unittest

from src.operational_evaluation import OperationalEvaluator, SampleIntervalError


class OperationalEvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.evaluator = OperationalEvaluator(
            os.path.join(self.temp_dir.name, "validation.csv"),
            os.path.join(self.temp_dir.name, "report.json"),
            minimum_samples=3,
            minimum_sample_interval_seconds=0,
            minimum_component_samples=2,
            minimum_emergency_class_samples=2,
        )
        self.prediction = {
            "density": "MEDIUM",
            "direction": "North",
            "emergency_type": "none",
            "signal": "Green",
        }
        self.actual = dict(self.prediction)

    def tearDown(self):
        self.temp_dir.cleanup()

    def record(self, prediction=None, actual=None, scene_id="frame"):
        return self.evaluator.record(
            prediction or self.prediction,
            actual or self.actual,
            scene_id,
        )

    def test_accuracy_is_not_reported_before_minimum_sample_count(self):
        self.record(scene_id="frame-1")
        summary = self.evaluator.summary()
        self.assertIsNone(summary["accuracy"])
        self.assertEqual(summary["status"], "collecting")
        self.assertEqual(summary["remaining_samples"], 2)

    def test_component_scores_appear_before_overall_exact_match(self):
        evaluator = OperationalEvaluator(
            os.path.join(self.temp_dir.name, "early.csv"),
            os.path.join(self.temp_dir.name, "early.json"),
            minimum_samples=5,
            minimum_sample_interval_seconds=0,
            minimum_component_samples=2,
        )
        for index in range(2):
            evaluator.record(self.prediction, self.actual, f"early-{index}")
        summary = evaluator.summary()
        self.assertIsNone(summary["accuracy"])
        self.assertEqual(summary["component_accuracy"]["density"], 100.0)
        self.assertEqual(summary["component_sample_count"], 2)

    def test_exact_match_accuracy_and_wilson_interval_are_reported_after_minimum(self):
        for index in range(3):
            self.record(scene_id=f"frame-{index}")
        summary = self.evaluator.summary()
        self.assertEqual(summary["accuracy"], 100.0)
        self.assertEqual(summary["correct_samples"], 3)
        self.assertEqual(summary["total_samples"], 3)
        self.assertLess(summary["accuracy_ci_95"]["lower"], 100.0)
        self.assertEqual(summary["accuracy_ci_95"]["upper"], 100.0)

    def test_any_component_mismatch_makes_whole_scene_incorrect(self):
        wrong = dict(self.actual)
        wrong["direction"] = "South"
        for index, actual in enumerate((self.actual, self.actual, wrong)):
            self.record(actual=actual, scene_id=f"frame-{index}")
        summary = self.evaluator.summary()
        self.assertAlmostEqual(summary["accuracy"], 66.7)
        self.assertAlmostEqual(summary["component_accuracy"]["direction"], 66.7)
        self.assertEqual(summary["correct_samples"], 2)

    def test_invalid_ground_truth_is_rejected(self):
        invalid = dict(self.actual)
        invalid["emergency_type"] = "police"
        with self.assertRaises(ValueError):
            self.record(actual=invalid, scene_id="frame-invalid")

    def test_same_camera_frame_cannot_be_submitted_twice(self):
        self.record(scene_id="session-1:frame-1")
        with self.assertRaises(ValueError):
            self.record(scene_id="session-1:frame-1")
        self.assertEqual(self.evaluator.summary()["total_samples"], 1)

    def test_frame_id_is_required(self):
        with self.assertRaises(ValueError):
            self.record(scene_id="")

    def test_new_scene_must_wait_for_minimum_validation_interval(self):
        evaluator = OperationalEvaluator(
            os.path.join(self.temp_dir.name, "interval.csv"),
            os.path.join(self.temp_dir.name, "interval.json"),
            minimum_samples=2,
            minimum_sample_interval_seconds=5,
        )
        evaluator.record(self.prediction, self.actual, "interval-1")
        with self.assertRaises(SampleIntervalError):
            evaluator.record(self.prediction, self.actual, "interval-2")

    def test_emergency_macro_recall_waits_for_all_three_classes(self):
        for index in range(2):
            self.record(scene_id=f"none-{index}")
        ambulance_actual = dict(self.actual, emergency_type="ambulance")
        ambulance_prediction = dict(self.prediction, emergency_type="ambulance")
        for index in range(2):
            self.record(
                prediction=ambulance_prediction,
                actual=ambulance_actual,
                scene_id=f"ambulance-{index}",
            )
        summary = self.evaluator.summary()
        self.assertIsNone(summary["emergency_macro_recall"])
        self.assertEqual(summary["emergency_class_counts"]["fire_truck"], 0)

    def test_emergency_macro_recall_weights_all_three_classes_equally(self):
        for index in range(2):
            self.record(scene_id=f"none-{index}")

        ambulance_actual = dict(self.actual, emergency_type="ambulance")
        ambulance_correct = dict(self.prediction, emergency_type="ambulance")
        ambulance_missed = dict(self.prediction, emergency_type="none")
        self.record(ambulance_correct, ambulance_actual, "ambulance-1")
        self.record(ambulance_missed, ambulance_actual, "ambulance-2")

        fire_actual = dict(self.actual, emergency_type="fire_truck")
        fire_prediction = dict(self.prediction, emergency_type="fire_truck")
        for index in range(2):
            self.record(fire_prediction, fire_actual, f"fire-{index}")

        summary = self.evaluator.summary()
        self.assertAlmostEqual(summary["emergency_macro_recall"], 83.3)
        self.assertEqual(summary["emergency_class_recall"]["none"], 100.0)
        self.assertEqual(summary["emergency_class_recall"]["ambulance"], 50.0)
        self.assertEqual(summary["emergency_class_recall"]["fire_truck"], 100.0)

    def test_incomplete_legacy_rows_are_excluded_from_metrics(self):
        with open(self.evaluator.csv_path, "a", encoding="utf-8") as handle:
            handle.write("2026-01-01T00:00:00+00:00,legacy-frame,,,,,,,,,1\n")
        summary = self.evaluator.summary()
        self.assertEqual(summary["total_samples"], 0)
        self.assertEqual(summary["excluded_incomplete_samples"], 1)


if __name__ == "__main__":
    unittest.main()
