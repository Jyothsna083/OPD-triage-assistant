"""
tests/test_abstention_appropriateness.py
Outcome 5 Verification: Abstention "Cannot assess — send to nurse now".
Tests:
  1. Exact user-facing message
  2. Escalation workflow (Priority 0 placed at top of queue)
  3. Appropriateness and cost evaluation against ground truth
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
import pandas as pd
from modules.preprocessing import preprocess_patient_data
from modules.red_flags import evaluate_safety_rules, evaluate_abstention_criteria
from modules import database as db


class TestAbstentionEngine(unittest.TestCase):

    def test_user_facing_message_and_trigger(self):
        """Verifies that severely compromised vitals trigger abstention."""
        patient_corrupt = {
            "name": "Corrupt Patient",
            "age": 45,
            "sex": "Male",
            "bp_systolic": 0,    # invalid zero
            "bp_diastolic": 0,   # invalid zero
            "pulse": 0,          # invalid zero
            "temp_c": None,      # missing
            "spo2": 95,
            "chief_complaint": "Feeling very weak",
        }
        preprocessed = preprocess_patient_data(patient_corrupt)
        rec = preprocessed.iloc[0].to_dict()

        # Check unusable vitals count
        self.assertGreaterEqual(rec["n_vitals_unusable"], 2)

        should_abstain, reasons = evaluate_abstention_criteria(rec)
        self.assertTrue(should_abstain)
        self.assertIn("Immediate nurse assessment required", reasons[0])

        safety_res = evaluate_safety_rules(rec)
        self.assertTrue(safety_res.should_abstain)

    def test_queue_escalation_workflow_priority_0(self):
        """Verifies that abstained patients are escalated to Priority 0 (ahead of all others)."""
        queue = db.get_patient_queue()
        abstained_in_queue = [p for p in queue if p["ai_abstain"] == 1 and p["nurse_reviewed"] == 0]

        if len(abstained_in_queue) > 0:
            first_patient = queue[0]
            # Priority order 0 must be top of the queue
            self.assertEqual(first_patient["priority_order"], 0)
            self.assertEqual(first_patient["ai_abstain"], 1)

    def test_appropriateness_on_ground_truth(self):
        """Verifies abstention recall and appropriateness on evaluate.py benchmark."""
        pred = pd.read_csv("predictions.csv")
        lab = pd.read_csv("data/hidden_labels.csv")
        df = lab.merge(pred, on="patient_id")

        should_ab = df["should_abstain"] == 1
        did_ab = df["abstain"] == 1

        # Recall on ground-truth cases requiring abstention
        ab_recall = (did_ab[should_ab]).mean()
        self.assertEqual(ab_recall, 1.0) # 100% of cases needing nurse review were caught

        # Precision
        ab_precision = (should_ab[did_ab]).mean()
        self.assertGreater(ab_precision, 0.70)


if __name__ == "__main__":
    unittest.main()
