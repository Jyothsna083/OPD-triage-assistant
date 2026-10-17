import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
from modules.red_flags import evaluate_safety_rules, apply_safety_overrides

class TestSafetyEngine(unittest.TestCase):

    def test_critical_hypoxia_level_1(self):
        record = {
            "age": 50,
            "spo2_clean": 82.0,
            "sbp_clean": 110.0,
            "pulse_clean": 95.0,
            "present_symptoms": ["cough"],
        }
        res = evaluate_safety_rules(record)
        self.assertEqual(res.override_level, 1)
        self.assertEqual(res.override_department, "Emergency")
        self.assertIn("CRITICAL_HYPOXIA_L1", res.triggered_rules)

    def test_profound_shock_level_1(self):
        record = {
            "age": 60,
            "sbp_clean": 65.0,
            "dbp_clean": 40.0,
            "pulse_clean": 115.0,
            "spo2_clean": 96.0,
            "present_symptoms": ["weakness"],
        }
        res = evaluate_safety_rules(record)
        self.assertEqual(res.override_level, 1)
        self.assertIn("PROFOUND_SHOCK_HYPOTENSION_L1", res.triggered_rules)

    def test_active_seizure_level_1(self):
        record = {
            "age": 28,
            "sbp_clean": 120.0,
            "pulse_clean": 100.0,
            "spo2_clean": 97.0,
            "present_symptoms": ["seizure"],
        }
        res = evaluate_safety_rules(record)
        self.assertEqual(res.override_level, 1)
        self.assertIn("ACTIVE_SEIZURE_L1", res.triggered_rules)

    def test_acs_chest_pain_level_2(self):
        record = {
            "age": 55,
            "sbp_clean": 130.0,
            "pulse_clean": 88.0,
            "spo2_clean": 98.0,
            "present_symptoms": ["chest_pain", "radiation_to_arm"],
            "detected_red_flags": ["acs_pattern"],
        }
        res = evaluate_safety_rules(record)
        self.assertEqual(res.override_level, 2)
        self.assertEqual(res.override_department, "Emergency")
        self.assertIn("ACS_CHEST_PAIN_L2", res.triggered_rules)

    def test_stroke_override_level_2(self):
        record = {
            "age": 68,
            "sbp_clean": 160.0,
            "pulse_clean": 82.0,
            "spo2_clean": 97.0,
            "present_symptoms": ["facial_droop", "slurred_speech"],
            "detected_red_flags": ["stroke_pattern"],
        }
        res = evaluate_safety_rules(record)
        self.assertEqual(res.override_level, 2)
        self.assertIn("ACUTE_STROKE_L2", res.triggered_rules)

    def test_poisoning_override_level_2(self):
        record = {
            "age": 30,
            "sbp_clean": 110.0,
            "pulse_clean": 90.0,
            "spo2_clean": 98.0,
            "present_symptoms": ["poisoning", "vomiting"],
        }
        res = evaluate_safety_rules(record)
        self.assertEqual(res.override_level, 2)
        self.assertIn("ACUTE_POISONING_L2", res.triggered_rules)

    def test_abstention_on_corrupt_vitals(self):
        record = {
            "age": 40,
            "n_vitals_unusable": 2,
            "temp_status": "missing",
            "bp_status": "invalid_zero",
            "present_symptoms": ["fever"],
        }
        res = evaluate_safety_rules(record)
        self.assertTrue(res.should_abstain)
        self.assertIn("DATA_QUALITY_ABSTENTION", res.triggered_rules)

    def test_mild_case_no_override(self):
        record = {
            "age": 25,
            "sbp_clean": 118.0,
            "dbp_clean": 78.0,
            "pulse_clean": 74.0,
            "temp_c_clean": 36.6,
            "spo2_clean": 99.0,
            "present_symptoms": ["cough"],
            "detected_red_flags": [],
            "n_vitals_unusable": 0,
        }
        res = evaluate_safety_rules(record)
        self.assertIsNone(res.override_level)
        self.assertFalse(res.should_abstain)

        # Applying override keeps ML prediction of Level 4 General Medicine
        lvl, dept, ab, exp = apply_safety_overrides(4, "General Medicine", res)
        self.assertEqual(lvl, 4)
        self.assertEqual(dept, "General Medicine")
        self.assertEqual(ab, 0)

    def test_override_only_escalates(self):
        # If ML predicts Level 1, an L2 rule shouldn't downgrade to Level 2
        record = {
            "age": 55,
            "sbp_clean": 130.0,
            "pulse_clean": 88.0,
            "spo2_clean": 98.0,
            "present_symptoms": ["chest_pain", "radiation_to_arm"],
            "detected_red_flags": ["acs_pattern"],
        }
        res = evaluate_safety_rules(record) # overrides to 2
        lvl, dept, ab, exp = apply_safety_overrides(1, "Emergency", res)
        self.assertEqual(lvl, 1) # preserved at Level 1

if __name__ == "__main__":
    unittest.main()
