import math
import unittest

from ml.evaluator import ModelEvaluator


class FixedPredictionModel:
    def __init__(self, values):
        self.values = values

    def predict(self, _features):
        return self.values


class TestEvaluationMetrics(unittest.TestCase):
    def test_fr43_classification_metrics_and_confusion_matrix(self):
        metrics, matrix, report = ModelEvaluator.evaluate(
            FixedPredictionModel(["cat", "dog", "cat", "cat"]),
            [0, 1, 2, 3],
            ["cat", "dog", "dog", "cat"],
            "classification",
        )
        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertIn("precision_weighted", metrics)
        self.assertIn("recall_weighted", metrics)
        self.assertIn("f1_weighted", metrics)
        self.assertEqual(matrix, [[2, 0], [1, 1]])
        self.assertIn("cat", report)
        self.assertIn("dog", report)

    def test_fr43_regression_metrics_match_known_values(self):
        metrics, matrix, report = ModelEvaluator.evaluate(
            FixedPredictionModel([1.0, 2.0, 3.0]),
            [0, 1, 2],
            [1.0, 3.0, 2.0],
            "regression",
        )
        self.assertAlmostEqual(metrics["mse"], 2 / 3)
        self.assertAlmostEqual(metrics["rmse"], math.sqrt(2 / 3))
        self.assertAlmostEqual(metrics["r2"], 0.0)
        self.assertIsNone(matrix)
        self.assertIsNone(report)

    def test_invalid_task_type_is_rejected(self):
        with self.assertRaises(ValueError):
            ModelEvaluator.evaluate(FixedPredictionModel([]), [], [], "unknown")


if __name__ == "__main__":
    unittest.main()
