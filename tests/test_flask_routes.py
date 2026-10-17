import sys
import os
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
import json
from app import app
from modules import database as db

class TestFlaskEndpoints(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
        # Set authenticated session
        with self.app.session_transaction() as sess:
            sess["user"] = {
                "user_id": 1,
                "username": "nurse.priya",
                "full_name": "Priya Sharma, RN",
                "role": "Triage Nurse",
                "staff_code": "RN-OGH-104",
                "department": "OPD Triage Desk",
            }
            sess["hospital_id"] = "HOSP_01"

    def test_login_page_get(self):
        # Test without session
        unauth_client = app.test_client()
        response = unauth_client.get("/login")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Hospital Staff Login", response.data)
        self.assertIn(b"King George Hospital", response.data)

    def test_outcomes_route(self):
        response = self.app.get("/outcomes")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Clinical Verification", response.data)
        self.assertIn(b"100.0%", response.data)

    def test_login_post_success(self):
        unauth_client = app.test_client()
        response = unauth_client.post("/login", data={
            "username": "nurse.priya",
            "password": "nurse123",
            "hospital_id": "HOSP_01"
        }, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Priya Sharma", response.data)

    def test_landing_page_get(self):
        unauth_client = app.test_client()
        response = unauth_client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Safe, Intelligent Outpatient Triage", response.data)
        self.assertIn(b"Connected AP Hospitals", response.data)

    def test_dashboard_route_authenticated(self):
        response = self.app.get("/dashboard")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Live Triage Queue", response.data)
        self.assertIn(b"PulseGuard", response.data)

    def test_intake_page_get(self):
        response = self.app.get("/intake")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Rapid Patient Intake", response.data)

    def test_simulation_route(self):
        response = self.app.get("/simulation")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Queue Simulation", response.data)

    def test_audit_route(self):
        response = self.app.get("/audit")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Audit Trail", response.data)

    def test_api_triage_emergency(self):
        payload = {
            "name": "Test Emergency",
            "age": 60,
            "sex": "Male",
            "bp_systolic": 140,
            "bp_diastolic": 90,
            "pulse": 105,
            "temp_c": 37.0,
            "spo2": 95,
            "chief_complaint": "Severe crushing chest pain radiating to left arm with sweating",
        }
        response = self.app.post("/api/triage", data=json.dumps(payload), content_type="application/json")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["status"], "success")
        self.assertIn(data["triage_level"], [1, 2])
        self.assertEqual(data["department"], "Emergency")
        self.assertIn("ACS_CHEST_PAIN_L2", data["detected_red_flags"])

    def test_api_triage_abstain(self):
        payload = {
            "name": "Corrupt Vitals",
            "age": 40,
            "sex": "Female",
            "bp_systolic": 0,
            "bp_diastolic": 0,
            "pulse": 0,
            "temp_c": None,
            "spo2": 96,
            "chief_complaint": "Weakness since morning",
        }
        response = self.app.post("/api/triage", data=json.dumps(payload), content_type="application/json")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["should_abstain"], 1)

    def test_nurse_override_flow(self):
        unique_name = f"MultiHosp Patient {int(time.time() * 1000)}"
        post_data = {
            "name": unique_name,
            "age": "30",
            "sex": "Male",
            "bp_systolic": "120",
            "bp_diastolic": "80",
            "pulse": "75",
            "temp_c": "36.8",
            "spo2": "99",
            "chief_complaint": "Mild sore throat and cough",
        }
        resp = self.app.post("/intake", data=post_data, follow_redirects=True)
        self.assertEqual(resp.status_code, 200)

        q = db.get_patient_queue()
        matching = [x for x in q if x["name"] == unique_name]
        self.assertTrue(len(matching) > 0)
        p = matching[0]

        override_data = {
            "patient_id": p["patient_id"],
            "assessment_id": p["assessment_id"],
            "nurse_name": "Priya Sharma, RN",
            "new_level": "2",
            "new_dept": "Emergency",
            "override_reason": "Patient developing acute respiratory distress on re-observation",
        }
        resp_ov = self.app.post("/override", data=override_data, follow_redirects=True)
        self.assertEqual(resp_ov.status_code, 200)

        detail = db.get_patient_detail(p["patient_id"])
        self.assertEqual(detail["assessment"]["final_triage_level"], 2)
        self.assertEqual(detail["assessment"]["final_department"], "Emergency")
        self.assertEqual(len(detail["overrides"]), 1)
        self.assertEqual(detail["overrides"][0]["nurse_name"], "Priya Sharma, RN")

if __name__ == "__main__":
    unittest.main()
