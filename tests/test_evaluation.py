"""Focused checks for class-aware model evaluation metrics."""

import unittest

from evaluation import evaluate_model


class EvaluationTests(unittest.TestCase):
    def test_metrics_use_malignant_as_positive_class(self) -> None:
        metrics = evaluate_model([0, 0, 1, 1], [0, 1, 0, 1])

        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["f1_score"], 0.5)
        self.assertEqual(metrics["benign_precision"], 0.5)
        self.assertEqual(metrics["benign_recall"], 0.5)
        self.assertEqual(metrics["confusion_matrix"].tolist(), [[1, 1], [1, 1]])
        self.assertIn("malignant", metrics["classification_report"])
        self.assertIn("benign", metrics["classification_report"])

    def test_zero_division_predictions_are_supported(self) -> None:
        metrics = evaluate_model([0, 1], [1, 1])

        self.assertEqual(metrics["precision"], 0.0)
        self.assertEqual(metrics["recall"], 0.0)
        self.assertEqual(metrics["f1_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
