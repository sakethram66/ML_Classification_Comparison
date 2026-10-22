"""Lightweight checks for the reusable Part 1 data pipeline."""

import contextlib
import io
import unittest

import numpy as np
import pandas as pd

from preprocessing import load_dataset, preprocess_data


class PreprocessingTests(unittest.TestCase):
    def setUp(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            self.X, self.y = load_dataset()

    def test_dataset_and_training_only_scaling(self) -> None:
        self.assertEqual(self.X.shape, (569, 30))
        self.assertEqual(self.y.shape, (569,))
        self.assertEqual(set(self.y.unique()), {0, 1})
        self.assertEqual(int((self.y == 0).sum()), 212)
        self.assertEqual(int((self.y == 1).sum()), 357)

        prepared = preprocess_data(self.X, self.y)
        self.assertEqual(prepared["X_train"].shape, (455, 30))
        self.assertEqual(prepared["X_test"].shape, (114, 30))
        np.testing.assert_allclose(
            prepared["scaler"].mean_,
            prepared["X_train"].mean(axis=0).to_numpy(),
        )
        self.assertTrue(np.isfinite(prepared["X_train_scaled"].to_numpy()).all())
        self.assertTrue(np.isfinite(prepared["X_test_scaled"].to_numpy()).all())
        np.testing.assert_array_equal(
            prepared["X_train"].to_numpy(),
            self.X.loc[prepared["X_train"].index].to_numpy(),
        )

    def test_quality_issues_are_reported_without_dropping_data(self) -> None:
        missing = self.X.copy()
        missing.iloc[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "missing feature values"):
            preprocess_data(missing, self.y)

        non_finite = self.X.copy()
        non_finite.iloc[0, 0] = np.inf
        with self.assertRaisesRegex(ValueError, "non-finite feature values"):
            preprocess_data(non_finite, self.y)

        duplicated_X = pd.concat([self.X, self.X.iloc[[0]]], ignore_index=True)
        duplicated_y = pd.concat([self.y, self.y.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "duplicate complete rows"):
            preprocess_data(duplicated_X, duplicated_y)


if __name__ == "__main__":
    unittest.main()
