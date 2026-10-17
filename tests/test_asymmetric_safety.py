"""
tests/test_asymmetric_safety.py
Rigorously tests Outcome 2: Asymmetric Safety Penalty (20x for emergency under-triage).
Compares standard argmax Random Forest against Cost-Sensitive Bayes Decision Rule
on held-out test records.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from modules.ml_model import (
    load_models,
    prepare_features,
    predict_cost_sensitive,
    get_cost_matrix,
    COST_MATRIX,
)


class TestAsymmetricSafety(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        triage_art, _ = load_models()
        cls.rf_triage = triage_art["model"]
        cls.tfidf = triage_art["tfidf"]

        dev_path = "data/processed/dev_dataset.csv"
        df = pd.read_csv(dev_path)
        _, cls.val_df = train_test_split(
            df, test_size=0.20, random_state=42, stratify=df["y_triage_level"]
        )

        cls.X_val, _, _ = prepare_features(cls.val_df, tfidf_vectorizer=cls.tfidf, is_train=False)
        cls.probas = cls.rf_triage.predict_proba(cls.X_val)
        cls.y_true = cls.val_df["y_triage_level"].values
        cls.is_emergency = (cls.val_df["y_is_emergency"].values == 1)

    def test_cost_matrix_asymmetry(self):
        """Verifies that the cost matrix imposes a 20x penalty multiplier on under-triaged emergencies."""
        # Cost of predicting Level 4 when true is Level 1 (gap = 3)
        # Row index 3 (Level 4), Col index 0 (Level 1)
        cost_under_emerg = COST_MATRIX[3, 0]
        self.assertEqual(cost_under_emerg, 3 * 20.0) # 60.0

        # Cost of predicting Level 4 when true is Level 3 (gap = 1, non-emergency)
        # Row index 3 (Level 4), Col index 2 (Level 3)
        cost_under_non_emerg = COST_MATRIX[3, 2]
        self.assertEqual(cost_under_non_emerg, 1 * 4.0) # 4.0

        # Cost of over-triage: predicting Level 1 when true is Level 3 (gap = -2)
        cost_over_triage = COST_MATRIX[0, 2]
        self.assertEqual(cost_over_triage, 2 * 1.0) # 2.0

        # Confirms 20x vs 1x asymmetry
        self.assertEqual(COST_MATRIX[1, 0] / COST_MATRIX[0, 1], 20.0)

    def test_cost_sensitive_vs_standard_performance(self):
        """Proves that Cost-Sensitive Bayes decision rule prevents missed emergencies."""
        pred_standard = self.rf_triage.predict(self.X_val)
        pred_cost_sensitive = predict_cost_sensitive(self.probas)

        # Count emergencies under-triaged (> Level 2)
        missed_standard = ((pred_standard > 2) & self.is_emergency).sum()
        missed_cost = ((pred_cost_sensitive > 2) & self.is_emergency).sum()

        print(f"\n[Test Outcome 2] Validation Emergencies: {self.is_emergency.sum()}")
        print(f"  Standard Random Forest missed emergencies: {missed_standard}")
        print(f"  Cost-Sensitive Bayes missed emergencies:   {missed_cost}")

        # The cost-sensitive decision rule must achieve 0 missed emergencies!
        self.assertEqual(missed_cost, 0)
        self.assertGreater(missed_standard, 0) # Standard model fails without cost sensitivity


if __name__ == "__main__":
    unittest.main()
