import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
from modules.nlp import parse_clinical_text, extract_duration_hours, extract_severity

class TestClinicalNLP(unittest.TestCase):

    def test_negation_english(self):
        res = parse_clinical_text("Patient presents with mild cough, denies fever, no chest pain for 2 days")
        self.assertIn("cough", res["present_symptoms"])
        self.assertIn("fever", res["negated_symptoms"])
        self.assertIn("chest_pain", res["negated_symptoms"])
        self.assertNotIn("chest_pain", res["present_symptoms"])
        self.assertNotIn("fever", res["present_symptoms"])
        self.assertNotIn("chest_pain", res["detected_red_flags"])

    def test_negation_hinglish(self):
        res = parse_clinical_text("khansi hai 3 din se, chhati me dard nahi hai aur bukhar nahi hai")
        self.assertIn("cough", res["present_symptoms"])
        self.assertIn("chest_pain", res["negated_symptoms"])
        self.assertIn("fever", res["negated_symptoms"])
        self.assertNotIn("chest_pain", res["present_symptoms"])
        self.assertNotIn("fever", res["present_symptoms"])
        self.assertNotIn("chest_pain", res["detected_red_flags"])

    def test_negation_telugu(self):
        res = parse_clinical_text("daggu undi 4 rojula nundi, gundelo noppi ledu, fever ledu")
        self.assertIn("cough", res["present_symptoms"])
        self.assertIn("chest_pain", res["negated_symptoms"])
        self.assertNotIn("chest_pain", res["present_symptoms"])
        self.assertNotIn("chest_pain", res["detected_red_flags"])

    def test_acs_chest_pain_red_flag(self):
        res1 = parse_clinical_text("Severe crushing chest pain radiating to left arm with profuse sweating for 1 hour")
        self.assertIn("chest_pain", res1["present_symptoms"])
        self.assertIn("radiation_to_arm", res1["present_symptoms"])
        self.assertIn("acs_pattern", res1["detected_red_flags"])

        res2 = parse_clinical_text("chhati mein bahut tez dard, baayein haath tak dard aur paseena 2 ghante se")
        self.assertIn("chest_pain", res2["present_symptoms"])
        self.assertIn("acs_pattern", res2["detected_red_flags"])

        res3 = parse_clinical_text("gundelo noppi, cheyi laagtundi chala chemata 1 ganta nundi")
        self.assertIn("chest_pain", res3["present_symptoms"])
        self.assertIn("acs_pattern", res3["detected_red_flags"])

    def test_stroke_red_flag(self):
        res1 = parse_clinical_text("Sudden onset facial droop and slurred speech, right arm weakness")
        self.assertIn("facial_droop", res1["present_symptoms"])
        self.assertIn("slurred_speech", res1["present_symptoms"])
        self.assertIn("stroke_pattern", res1["detected_red_flags"])

        res2 = parse_clinical_text("muh tedha ho gaya aur bol nahi pa raha, achanak hua")
        self.assertIn("facial_droop", res2["present_symptoms"])
        self.assertIn("stroke_pattern", res2["detected_red_flags"])

        res3 = parse_clinical_text("moothi vankara poyindi maata thadabaduthondi")
        self.assertIn("facial_droop", res3["present_symptoms"])
        self.assertIn("stroke_pattern", res3["detected_red_flags"])

    def test_duration_extraction(self):
        h, acute = extract_duration_hours("cough for 6 hours")
        self.assertEqual(h, 6.0)
        self.assertTrue(acute)

        h, acute = extract_duration_hours("fever for 3 days")
        self.assertEqual(h, 72.0)
        self.assertFalse(acute)

        h, acute = extract_duration_hours("bukhar 4 din se")
        self.assertEqual(h, 96.0)

        h, acute = extract_duration_hours("noppi 5 rojula nundi")
        self.assertEqual(h, 120.0)

        h, acute = extract_duration_hours("sudden onset severe headache")
        self.assertEqual(h, 0.5)
        self.assertTrue(acute)

    def test_severity_extraction(self):
        self.assertEqual(extract_severity("halka sar dard"), "mild")
        self.assertEqual(extract_severity("chinna noppi anthe"), "mild")
        self.assertEqual(extract_severity("bahut zyada tez dard, bardasht nahi"), "severe")
        self.assertEqual(extract_severity("chala ekkuva bhaaralenu"), "severe")
        self.assertEqual(extract_severity("runny nose and sneezing"), "moderate")

if __name__ == "__main__":
    unittest.main()
