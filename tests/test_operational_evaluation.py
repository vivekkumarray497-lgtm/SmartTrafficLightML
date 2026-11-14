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

    def test_accuracy_is_not_reported_before_minimum_sample_count(self):
        self.evaluator.record(self.prediction, self.actual, "frame-1")
        summary = self.evaluator.summary()
        self.assertIsNone(summary["accuracy"])
        self.assertEqual(summary["status"], "collecting")
        self.assertEqual(summary["remaining_samples"], 2)

    def test_exact_match_accuracy_is_reported_after_minimum_samples(self):
        for index in range(3):
            self.evaluator.record(self.prediction, self.actual, f"frame-{index}")
        summary = self.evaluator.summary()
        self.assertEqual(summary["accuracy"], 100.0)
        self.assertEqual(summary["correct_samples"], 3)
        self.assertEqual(summary["total_samples"], 3)

    def test_any_component_mismatch_makes_the_whole_sample_incorrect(self):
        wrong = dict(self.actual)
        wrong["direction"] = "South"
        for index, actual in enumerate((self.actual, self.actual, wrong)):
            self.evaluator.record(self.prediction, actual, f"frame-{index}")
        summary = self.evaluator.summary()
        self.assertAlmostEqual(summary["accuracy"], 66.7)
        self.assertEqual(summary["component_accuracy"]["direction"], 66.7)

    def test_invalid_ground_truth_is_rejected(self):
        invalid = dict(self.actual)
        invalid["emergency_type"] = "police"
        with self.assertRaises(ValueError):
            self.evaluator.record(self.prediction, invalid, "frame-invalid")

    def test_same_camera_frame_cannot_be_submitted_twice(self):
        self.evaluator.record(self.prediction, self.actual, "session-1:frame-1")
        with self.assertRaises(ValueError):
            self.evaluator.record(self.prediction, self.actual, "session-1:frame-1")
        self.assertEqual(self.evaluator.summary()["total_samples"], 1)

    def test_frame_id_is_required(self):
        with self.assertRaises(ValueError):
            self.evaluator.record(self.prediction, self.actual, "")

    def test_new_scene_must_wait_for_minimum_validation_interval(self):
        evaluator = OperationalEvaluator(
            os.path.join(self.temp_dir.name, "interval.csv"),
            os.path.join(self.temp_dir.name, "interval-report.json"),
            minimum_samples=2,
            minimum_sample_interval_seconds=5,
        )
        evaluator.record(self.prediction, self.actual, "session:frame-1")
        with self.assertRaises(SampleIntervalError):
            evaluator.record(self.prediction, self.actual, "session:frame-2")

    def test_emergency_accuracy_requires_positive_and_negative_examples(self):
        for index in range(3):
            self.evaluator.record(self.prediction, self.actual, f"frame-{index}")
        self.assertIsNone(self.evaluator.summary()["emergency_accuracy"])


if __name__ == "__main__":
    unittest.main()
