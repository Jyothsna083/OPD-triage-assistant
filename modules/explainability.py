"""
modules/explainability.py
Clinical Explainability & Feature Contribution Engine for OPD Triage.
Translates model outputs, physiological risk factors, and NLP concepts
into human-interpretable clinical rationales for nurses and attending physicians.
"""

from typing import Dict, List, Any
import numpy as np
import pandas as pd


def generate_clinical_explanation(
    patient_dict: Dict[str, Any],
    triage_level: int,
    department: str,
    triggered_rules: List[str],
    level_probabilities: Dict[str, float],
) -> Dict[str, Any]:
    """
    Generates a structured clinical explanation breakdown for the dashboard:
      - Primary driving factors (vitals, symptoms, age)
      - Red flag escalations
      - Risk contribution scores
    """
    factors = []
    age = float(patient_dict.get("age", 35) if pd.notna(patient_dict.get("age")) else 35)
    sbp = patient_dict.get("sbp_clean") or patient_dict.get("bp_systolic")
    dbp = patient_dict.get("dbp_clean") or patient_dict.get("bp_diastolic")
    pulse = patient_dict.get("pulse_clean") or patient_dict.get("pulse")
    temp = patient_dict.get("temp_c_clean") or patient_dict.get("temp_c")
    spo2 = patient_dict.get("spo2_clean") or patient_dict.get("spo2")

    # Oxygenation
    if spo2 is not None and pd.notna(spo2):
        if spo2 < 90:
            factors.append({"factor": "Severe Hypoxemia", "detail": f"SpO2 {spo2}% indicates severe respiratory impairment", "risk": "Critical"})
        elif spo2 < 95:
            factors.append({"factor": "Mild-Moderate Hypoxia", "detail": f"SpO2 {spo2}% below normal room-air baseline", "risk": "Moderate"})

    # Hemodynamics
    if sbp is not None and pd.notna(sbp) and sbp > 0:
        if sbp < 85:
            factors.append({"factor": "Hypotension", "detail": f"Systolic BP {sbp} mmHg indicates potential circulatory shock", "risk": "High"})
        elif sbp >= 180 or (dbp and dbp >= 110):
            factors.append({"factor": "Severe Hypertension", "detail": f"BP {sbp}/{dbp} mmHg in hypertensive emergency range", "risk": "High"})

    # Heart rate
    if pulse is not None and pd.notna(pulse) and pulse > 0:
        if pulse > 130:
            factors.append({"factor": "Marked Tachycardia", "detail": f"Heart rate {pulse} bpm suggests severe physiological stress", "risk": "High"})
        elif pulse < 45:
            factors.append({"factor": "Severe Bradycardia", "detail": f"Heart rate {pulse} bpm requires urgent cardiac evaluation", "risk": "High"})

    # Temperature
    if temp is not None and pd.notna(temp):
        if temp >= 39.0:
            factors.append({"factor": "High Pyrexia", "detail": f"Body temperature {temp:.1f}°C indicates severe infection or systemic response", "risk": "Moderate"})
        elif temp < 35.5:
            factors.append({"factor": "Hypothermia", "detail": f"Core temp {temp:.1f}°C indicates metabolic crisis or severe sepsis", "risk": "High"})

    # Pediatric age factor
    if age <= 1:
        factors.append({"factor": "Infant Vulnerability", "detail": "Age under 1 year requires heightened clinical vigilance", "risk": "Advisory"})
    elif age >= 75:
        factors.append({"factor": "Geriatric Complexity", "detail": "Age 75+ increases risk of atypical disease presentation", "risk": "Advisory"})

    return {
        "triage_level": triage_level,
        "department": department,
        "key_clinical_factors": factors,
        "hard_coded_rules": triggered_rules,
        "confidence": level_probabilities.get(f"Level_{triage_level}", 0.0),
    }
