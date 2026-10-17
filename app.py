"""
app.py
PulseGuard AI: Multi-Facility Outpatient Triage & Emergency Safeguard.
Features:
  - Multi-hospital network portal & authorized staff authentication
  - Live patient queue dashboard prioritized by clinical urgency
  - Interactive Review & Override Modal directly on the queue
  - Rapid walk-in intake form (<60s per patient) with presets
  - Assessment detail page with transparent ML & safety override explanation
  - Nurse override workflow with mandatory clinical rationale logging
  - Queue simulation and wait time analytics
  - REST API endpoints for external integration
"""

import os
import sys
from functools import wraps
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules import database as db
from modules.preprocessing import preprocess_patient_data
from modules.nlp import parse_clinical_text
from modules.red_flags import evaluate_safety_rules, apply_safety_overrides
from modules.ml_model import load_models, prepare_features, predict_cost_sensitive
from modules.queue_simulation import run_and_plot_queue_simulation

app = Flask(__name__)
app.secret_key = "pulseguard_ai_enterprise_secret_2026"

# Cache model artifacts
TRIAGE_ARTIFACT = None
DEPT_ARTIFACT = None

def get_models():
    global TRIAGE_ARTIFACT, DEPT_ARTIFACT
    if TRIAGE_ARTIFACT is None or DEPT_ARTIFACT is None:
        TRIAGE_ARTIFACT, DEPT_ARTIFACT = load_models()
    return TRIAGE_ARTIFACT, DEPT_ARTIFACT


# =========================================================
# Multi-Hospital Context & Auth Helpers
# =========================================================
@app.context_processor
def inject_hospital_and_user_context():
    """Injects current facility and authenticated staff context into all templates."""
    user = session.get("user")
    hospital_id = session.get("hospital_id", "HOSP_01")
    current_hospital = db.get_hospital_by_id(hospital_id)
    all_hospitals = db.get_all_hospitals()
    return {
        "current_user": user,
        "current_hospital": current_hospital,
        "all_hospitals": all_hospitals,
    }


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login", next=request.url))
        return f(*args, **kwargs)
    return decorated_function


def process_single_patient(raw_dict):
    """
    Executes the end-to-end clinical pipeline for a single patient:
    Preprocessing -> NLP -> ML Model -> Safety Engine -> Override Resolution.
    """
    triage_art, dept_art = get_models()
    rf_triage = triage_art["model"]
    rf_dept = dept_art["model"]
    tfidf = triage_art["tfidf"]

    # 1. Preprocessing
    preprocessed_df = preprocess_patient_data(raw_dict)
    patient_clean = preprocessed_df.iloc[0].to_dict()

    # 2. NLP Extraction
    parsed_nlp = parse_clinical_text(patient_clean.get("chief_complaint", ""))
    patient_clean["present_symptoms"] = parsed_nlp["present_symptoms"]
    patient_clean["negated_symptoms"] = parsed_nlp["negated_symptoms"]
    patient_clean["detected_red_flags"] = parsed_nlp["detected_red_flags"]

    # 3. ML Model Prediction
    X, _, _ = prepare_features(preprocessed_df, tfidf_vectorizer=tfidf, is_train=False)
    probas = rf_triage.predict_proba(X)[0]
    ml_level = int(predict_cost_sensitive(np.array([probas]))[0])
    ml_dept = str(rf_dept.predict(X)[0])

    prob_dict = {f"Level_{c}": round(float(p), 4) for c, p in zip(triage_art["classes"], probas)}

    # 4. Hard-coded Safety Engine
    safety_res = evaluate_safety_rules(patient_clean)
    fin_lvl, fin_dept, ab, explanations = apply_safety_overrides(ml_level, ml_dept, safety_res)

    return {
        "cleaned_patient": patient_clean,
        "ai_result": {
            "ai_triage_level": ml_level,
            "ai_department": ml_dept,
            "ai_abstain": ab,
            "final_triage_level": fin_lvl,
            "final_department": fin_dept,
            "detected_red_flags": safety_res.triggered_rules,
            "explanations": explanations,
            "level_probabilities": prob_dict,
        }
    }


# =========================================================
# Authentication & Multi-Hospital Routes
# =========================================================
@app.route("/login", methods=["GET", "POST"])
def login():
    """Multi-Hospital Staff Authentication Portal."""
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        hospital_id = request.form.get("hospital_id")

        user = db.authenticate_staff(username, password, hospital_id=hospital_id)
        if user:
            session["user"] = {
                "user_id": user["user_id"],
                "username": user["username"],
                "full_name": user["full_name"],
                "role": user["role"],
                "staff_code": user["staff_code"],
                "department": user["department"],
            }
            session["hospital_id"] = user["hospital_id"]
            flash(f"Welcome, {user['full_name']}! Signed into {user['hospital_name']}.", "info")
            next_url = request.args.get("next") or url_for("index")
            return redirect(next_url)
        else:
            flash("Authentication Failed: Invalid staff credentials or facility selection.", "danger")

    hospitals = db.get_all_hospitals()
    return render_template("login.html", hospitals=hospitals)


@app.route("/logout")
def logout():
    """Terminates staff session."""
    session.clear()
    flash("You have been signed out of PulseGuard AI.", "info")
    return redirect(url_for("login"))


@app.route("/switch-hospital/<hospital_id>")
@login_required
def switch_hospital(hospital_id):
    """Switches active hospital facility context."""
    h = db.get_hospital_by_id(hospital_id)
    if h:
        session["hospital_id"] = hospital_id
        flash(f"Switched facility context to {h['name']} ({h['code']}).", "info")
    return redirect(request.referrer or url_for("index"))


# =========================================================
# Public Landing & Clinical Workflow Routes
# =========================================================
@app.route("/")
def landing():
    """Public landing page for PulseGuard AI across 37 AP Government Hospitals."""
    hospitals = db.get_all_hospitals()
    return render_template("landing.html", hospitals=hospitals)


@app.route("/dashboard")
@app.route("/queue")
@login_required
def index():
    """Live Triage Queue Dashboard."""
    dept_filter = request.args.get("dept", "All")
    hosp_filter = session.get("hospital_id", "HOSP_01")
    queue = db.get_patient_queue(filter_dept=dept_filter, filter_hospital_id=hosp_filter)

    # If active hospital queue is empty, fallback to showing all waiting
    if len(queue) == 0:
        queue = db.get_patient_queue(filter_dept=dept_filter)

    critical_count = sum(1 for p in queue if p["final_triage_level"] in [1, 2] and p["ai_abstain"] == 0)
    abstain_count = sum(1 for p in queue if p["ai_abstain"] == 1 and p["nurse_reviewed"] == 0)

    return render_template(
        "dashboard.html",
        queue=queue,
        critical_count=critical_count,
        abstain_count=abstain_count,
    )


@app.route("/intake", methods=["GET", "POST"])
@login_required
def intake():
    """Rapid patient intake page."""
    if request.method == "POST":
        def parse_float(val):
            try:
                return float(val) if val and str(val).strip() else None
            except (ValueError, TypeError):
                return None

        hospital_id = session.get("hospital_id", "HOSP_01")

        # Flexible Blood Pressure parsing (supports single '120/80' input or separated fields)
        bp_raw = request.form.get("bp", "").strip()
        bp_sys = parse_float(request.form.get("bp_systolic"))
        bp_dia = parse_float(request.form.get("bp_diastolic"))

        if (bp_sys is None or bp_dia is None) and bp_raw:
            import re
            m = re.match(r"^(\d{2,3})\s*[/,\-\s]\s*(\d{2,3})$", bp_raw)
            if m:
                bp_sys = float(m.group(1))
                bp_dia = float(m.group(2))
            else:
                digits = re.findall(r"\d+", bp_raw)
                if len(digits) >= 2:
                    bp_sys = float(digits[0])
                    bp_dia = float(digits[1])
                elif len(digits) == 1 and bp_sys is None:
                    bp_sys = float(digits[0])

        raw_patient = {
            "patient_id": f"P_WALKIN_{int(datetime.now().timestamp()*1000) % 100000}",
            "name": request.form.get("name", "Walk-in Patient").strip() or "Walk-in Patient",
            "age": parse_float(request.form.get("age")),
            "sex": request.form.get("sex", "Unknown"),
            "bp_systolic": bp_sys,
            "bp_diastolic": bp_dia,
            "pulse": parse_float(request.form.get("pulse")),
            "temp_c": parse_float(request.form.get("temp_c")),
            "spo2": parse_float(request.form.get("spo2")),
            "chief_complaint": request.form.get("chief_complaint", "").strip(),
            "known_conditions": request.form.get("known_conditions", "").strip(),
            "medication_list": request.form.get("medication_list", "").strip(),
            "language_style": "user_input",
        }

        outcome = process_single_patient(raw_patient)
        db.save_patient_and_assessment(raw_patient, outcome["ai_result"], hospital_id=hospital_id)

        fin_lvl = outcome["ai_result"]["final_triage_level"]
        fin_dept = outcome["ai_result"]["final_department"]
        ab = outcome["ai_result"]["ai_abstain"]

        if ab == 1:
            flash(f"Immediate Nurse Assessment Required: Vitals uninterpretable. Assigned Priority 0.", "warning")
        elif fin_lvl in [1, 2]:
            flash(f"Critical Safety Escalation: Assigned Level {fin_lvl} ({fin_dept}) based on clinical red flags.", "danger")
        else:
            flash(f"Patient Triaged: Assigned Level {fin_lvl} ({fin_dept}). Added to live queue.", "info")

        return redirect(url_for("patient_detail", patient_id=raw_patient["patient_id"]))

    return render_template("intake.html")


@app.route("/patient/<patient_id>")
@login_required
def patient_detail(patient_id):
    """Assessment detail page and nurse override interface."""
    patient = db.get_patient_detail(patient_id)
    if not patient:
        flash("Patient record not found.", "warning")
        return redirect(url_for("index"))

    return render_template("patient_detail.html", patient=patient)


@app.route("/override", methods=["POST"])
@login_required
def override():
    """Handles nurse override of AI decision with mandatory reason."""
    default_nurse = session.get("user", {}).get("full_name", "Staff Nurse")
    hospital_id = session.get("hospital_id", "HOSP_01")

    # Check if request is JSON (e.g. from Dashboard Modal)
    if request.is_json:
        data = request.get_json(silent=True) or {}
        patient_id = data.get("patient_id")
        assessment_id = data.get("assessment_id")
        nurse_name = data.get("nurse_name", default_nurse).strip()
        new_level = int(data.get("new_level", 3))
        new_dept = data.get("new_dept", "General Medicine")
        reason = data.get("override_reason", "").strip()
        is_ajax = True
    else:
        patient_id = request.form.get("patient_id")
        assessment_id = request.form.get("assessment_id")
        nurse_name = request.form.get("nurse_name", default_nurse).strip()
        new_level = int(request.form.get("new_level", 3))
        new_dept = request.form.get("new_dept", "General Medicine")
        reason = request.form.get("override_reason", "").strip()
        is_ajax = False

    if not reason or len(reason) < 5:
        msg = "Clinical Policy Error: Mandatory reason (minimum 5 characters) must be provided for override."
        if is_ajax:
            return jsonify({"status": "error", "message": msg}), 400
        flash(msg, "danger")
        return redirect(url_for("patient_detail", patient_id=patient_id))

    # Fallback lookup of assessment_id if missing
    try:
        assessment_id = int(assessment_id)
    except (ValueError, TypeError):
        assessment_id = None

    if not assessment_id:
        conn = db.get_connection()
        cur = conn.cursor()
        cur.execute("SELECT assessment_id FROM triage_assessments WHERE patient_id = ? ORDER BY created_at DESC LIMIT 1", (patient_id,))
        row = cur.fetchone()
        conn.close()
        if row:
            assessment_id = row["assessment_id"]

    success = False
    if assessment_id:
        success = db.record_nurse_override(
            assessment_id=assessment_id,
            patient_id=patient_id,
            nurse_name=nurse_name,
            new_level=new_level,
            new_dept=new_dept,
            reason=reason,
            hospital_id=hospital_id,
        )

    if success:
        msg = f"Nurse Override Saved: Patient updated to Level {new_level} ({new_dept}). Action recorded in audit history."
        if is_ajax:
            return jsonify({
                "status": "success",
                "message": msg,
                "new_level": new_level,
                "new_dept": new_dept
            })
        flash(msg, "info")
    else:
        msg = "Failed to record override."
        if is_ajax:
            return jsonify({"status": "error", "message": msg}), 500
        flash(msg, "danger")

    next_url = request.form.get("next", url_for("patient_detail", patient_id=patient_id))
    return redirect(next_url)


@app.route("/simulation")
@login_required
def simulation():
    """Queue simulation and wait time impact page."""
    chart_path = os.path.join("static", "img", "queue_simulation_chart.png")
    if not os.path.exists(chart_path):
        run_and_plot_queue_simulation()
    return render_template("simulation.html")


@app.route("/outcomes")
def outcomes():
    """5 Required Outcomes verification page and clinical benchmark matrix."""
    return render_template("outcomes.html")


@app.route("/audit")
@login_required
def audit():
    """System-wide audit trail."""
    hospital_id = session.get("hospital_id")
    audit_logs = db.get_audit_trail(limit=100, hospital_id=hospital_id)
    return render_template("audit.html", audit_logs=audit_logs)


# =========================================================
# REST API Endpoints (Stateless & Direct)
# =========================================================
@app.route("/api/triage", methods=["POST"])
def api_triage():
    """
    REST API endpoint for real-time triage prediction.
    Accepts JSON body with patient intake fields.
    Returns JSON with cleaned vitals, NLP concepts, and triage decision.
    """
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON payload provided"}), 400

    outcome = process_single_patient(data)
    return jsonify({
        "status": "success",
        "triage_level": outcome["ai_result"]["final_triage_level"],
        "department": outcome["ai_result"]["final_department"],
        "should_abstain": outcome["ai_result"]["ai_abstain"],
        "detected_red_flags": outcome["ai_result"]["detected_red_flags"],
        "clinical_rationale": outcome["ai_result"]["explanations"],
        "probabilities": outcome["ai_result"]["level_probabilities"],
    })


@app.route("/api/queue", methods=["GET"])
def api_queue():
    """Returns active queue in JSON format."""
    dept = request.args.get("dept")
    hosp = request.args.get("hospital_id")
    queue = db.get_patient_queue(filter_dept=dept, filter_hospital_id=hosp)
    return jsonify({"queue": queue, "count": len(queue)})


@app.route("/api/override", methods=["POST"])
def api_override():
    """Dedicated REST endpoint for nurse overrides."""
    return override()


if __name__ == "__main__":
    db.init_db(reset=False)
    q = db.get_patient_queue()
    if len(q) == 0:
        from scripts.seed_database import seed
        seed(40)
    print("Starting PulseGuard AI Multi-Hospital platform on http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
