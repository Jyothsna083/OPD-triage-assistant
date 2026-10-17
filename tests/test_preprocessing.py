import unittest
import numpy as np
import pandas as pd
from modules.preprocessing import (
    clean_temperature,
    clean_blood_pressure,
    clean_pulse,
    clean_spo2,
    preprocess_patient_data,
)

class TestPreprocessing(unittest.TestCase):

    def test_temperature_cleaning(self):
        # Normal C
        val, status = clean_temperature(37.2)
        self.assertEqual(val, 37.2)
        self.assertEqual(status, "ok")

        # Fahrenheit input (e.g. 98.6 F -> 37.0 C)
        val, status = clean_temperature(98.6)
        self.assertEqual(val, 37.0)
        self.assertEqual(status, "converted_from_F")

        # Fahrenheit 96 F -> 35.6 C
        val, status = clean_temperature(96.0)
        self.assertEqual(val, 35.6)
        self.assertEqual(status, "converted_from_F")

        # Missing / None
        val, status = clean_temperature(None)
        self.assertTrue(np.isnan(val))
        self.assertEqual(status, "missing")

        # Implausible
        val, status = clean_temperature(20.0)
        self.assertTrue(np.isnan(val))
        self.assertEqual(status, "invalid_out_of_range")

    def test_bp_cleaning(self):
        # Normal BP
        sbp, dbp, status = clean_blood_pressure(120, 80)
        self.assertEqual((sbp, dbp), (120.0, 80.0))
        self.assertEqual(status, "ok")

        # Swapped BP (80/120 -> 120/80)
        sbp, dbp, status = clean_blood_pressure(80, 120)
        self.assertEqual((sbp, dbp), (120.0, 80.0))
        self.assertEqual(status, "swapped_fixed")

        # Zero BP
        sbp, dbp, status = clean_blood_pressure(0, 0)
        self.assertTrue(np.isnan(sbp))
        self.assertEqual(status, "invalid_zero")

    def test_pulse_and_spo2(self):
        # Pulse
        p, s = clean_pulse(72)
        self.assertEqual((p, s), (72.0, "ok"))
        p, s = clean_pulse(0)
        self.assertTrue(np.isnan(p))
        self.assertEqual(s, "invalid_zero")

        # SpO2
        sp, s = clean_spo2(98)
        self.assertEqual((sp, s), (98.0, "ok"))
        sp, s = clean_spo2(0)
        self.assertTrue(np.isnan(sp))
        self.assertEqual(s, "invalid_zero")

    def test_single_record_pipeline(self):
        record = {
            "patient_id": "P_001",
            "age": 45,
            "sex": "Female",
            "bp_systolic": 75,
            "bp_diastolic": 115, # swapped
            "pulse": 110,
            "temp_c": 101.5, # Fahrenheit
            "spo2": 96,
            "chief_complaint": "Severe fever and body pain",
            "known_conditions": "Diabetes, Hypertension",
        }
        res = preprocess_patient_data(record)
        self.assertEqual(len(res), 1)
        self.assertEqual(res.iloc[0]["sbp_clean"], 115.0)
        self.assertEqual(res.iloc[0]["dbp_clean"], 75.0)
        self.assertEqual(res.iloc[0]["bp_status"], "swapped_fixed")
        self.assertEqual(res.iloc[0]["temp_status"], "converted_from_F")
        self.assertAlmostEqual(res.iloc[0]["temp_c_clean"], 38.6, places=1)
        self.assertEqual(res.iloc[0]["hist_diabetes"], 1)
        self.assertEqual(res.iloc[0]["hist_hypertension"], 1)
        self.assertEqual(res.iloc[0]["n_vitals_unusable"], 0)

    def test_batch_pipeline(self):
        batch = pd.DataFrame([
            {"patient_id": "P_001", "age": 70, "bp_systolic": 0, "bp_diastolic": 0, "pulse": 0, "temp_c": np.nan, "spo2": 95},
            {"patient_id": "P_002", "age": 5, "bp_systolic": 90, "bp_diastolic": 60, "pulse": 105, "temp_c": 36.8, "spo2": 99},
        ])
        res = preprocess_patient_data(batch)
        self.assertEqual(len(res), 2)
        # P_001 has 0 BP (invalid), 0 pulse (invalid), NaN temp (missing) -> 3 unusable vitals
        self.assertEqual(res.iloc[0]["n_vitals_unusable"], 3)
        self.assertEqual(res.iloc[0]["is_elderly"], 1)
        self.assertEqual(res.iloc[1]["n_vitals_unusable"], 0)
        self.assertEqual(res.iloc[1]["is_child"], 1)

if __name__ == "__main__":
    unittest.main()
