import os
import tempfile
import unittest

from src.operational_evaluation import OperationalEvaluator


class OperationalEvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.evaluator = OperationalEvaluator(
            os.path.join(self.temp_dir.name, "validation.csv"),
            os.path.join(self.temp_dir.name, "report.json"),
            minimum_samples=3,
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
        self.evaluator.record(self.prediction, self.actual)
        summary = self.evaluator.summary()
        self.assertIsNone(summary["accuracy"])
        self.assertEqual(summary["status"], "collecting")
        self.assertEqual(summary["remaining_samples"], 2)

    def test_exact_match_accuracy_is_reported_after_minimum_samples(self):
        for _ in range(3):
            self.evaluator.record(self.prediction, self.actual)
        summary = self.evaluator.summary()
        self.assertEqual(summary["accuracy"], 100.0)
        self.assertEqual(summary["correct_samples"], 3)
        self.assertEqual(summary["total_samples"], 3)

    def test_any_component_mismatch_makes_the_whole_sample_incorrect(self):
        wrong = dict(self.actual)
        wrong["direction"] = "South"
        for actual in (self.actual, self.actual, wrong):
            self.evaluator.record(self.prediction, actual)
        summary = self.evaluator.summary()
        self.assertAlmostEqual(summary["accuracy"], 66.7)
        self.assertEqual(summary["component_accuracy"]["direction"], 66.7)

    def test_invalid_ground_truth_is_rejected(self):
        invalid = dict(self.actual)
        invalid["emergency_type"] = "police"
        with self.assertRaises(ValueError):
            self.evaluator.record(self.prediction, invalid)

    def test_emergency_accuracy_requires_positive_and_negative_examples(self):
        for _ in range(3):
            self.evaluator.record(self.prediction, self.actual)
        self.assertIsNone(self.evaluator.summary()["emergency_accuracy"])


if __name__ == "__main__":
    unittest.main()
