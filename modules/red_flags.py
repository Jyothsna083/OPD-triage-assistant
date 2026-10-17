"""
modules/red_flags.py
Hard-coded Safety Escalation Engine and Rule-Based Clinical Overrides.
Based on the Emergency Severity Index (ESI) clinical rubric.
Guarantees asymmetric safety by overriding ML predictions for any life-threatening
vitals or high-risk symptom presentations, and triggers abstention when vitals
are severely compromised.
"""

from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd


class SafetyRuleResult:
    def __init__(
        self,
        override_level: Optional[int] = None,
        override_department: Optional[str] = None,
        should_abstain: bool = False,
        triggered_rules: Optional[List[str]] = None,
        explanations: Optional[List[str]] = None,
    ):
        self.override_level = override_level
        self.override_department = override_department
        self.should_abstain = should_abstain
        self.triggered_rules = triggered_rules or []
        self.explanations = explanations or []

    @property
    def has_override(self) -> bool:
        return self.override_level is not None or self.should_abstain

    def to_dict(self) -> Dict[str, Any]:
        return {
            "override_level": self.override_level,
            "override_department": self.override_department,
            "should_abstain": int(self.should_abstain),
            "triggered_rules": self.triggered_rules,
            "explanations": self.explanations,
        }


def evaluate_abstention_criteria(patient_record: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Evaluates whether an intake record has too many missing or corrupt vitals
    to be safely triaged by an automated model, requiring immediate nurse evaluation.
    Rule: 2 or more unusable vitals (or missing essential vitals during acute illness).
    """
    reasons = []
    n_unusable = patient_record.get("n_vitals_unusable", 0)

    # Check vitals statuses
    statuses = [
        patient_record.get("temp_status", "ok"),
        patient_record.get("bp_status", "ok"),
        patient_record.get("pulse_status", "ok"),
        patient_record.get("spo2_status", "ok"),
    ]
    invalid_or_missing_count = sum(1 for s in statuses if s in ["missing", "invalid_zero", "invalid", "invalid_out_of_range"])

    if n_unusable >= 2 or invalid_or_missing_count >= 2:
        reasons.append(f"Multiple unusable/missing vital signs ({max(n_unusable, invalid_or_missing_count)} abnormal/missing). Immediate nurse assessment required.")

    return len(reasons) > 0, reasons


def evaluate_safety_rules(patient_record: Dict[str, Any]) -> SafetyRuleResult:
    """
    Evaluates physiological vital signs and clinical NLP flags against hard-coded
    clinical escalation rules.
    Returns SafetyRuleResult with override recommendations and explanations.
    """
    triggered_rules = []
    explanations = []
    target_level = None
    target_dept = None

    age = float(patient_record.get("age", 35) if pd.notna(patient_record.get("age")) else 35)
    sbp = patient_record.get("sbp_clean")
    dbp = patient_record.get("dbp_clean")
    pulse = patient_record.get("pulse_clean")
    temp = patient_record.get("temp_c_clean")
    spo2 = patient_record.get("spo2_clean")

    # NLP concepts / red flags
    nlp_rfs = patient_record.get("detected_red_flags", [])
    if isinstance(nlp_rfs, str):
        nlp_rfs = [r.strip() for r in nlp_rfs.split(",") if r.strip()]
    present_symptoms = patient_record.get("present_symptoms", [])

    # -------------------------------------------------------------
    # 1. LEVEL 1: IMMEDIATE RESUSCITATION / LIFE THREAT
    # -------------------------------------------------------------
    level_1_triggered = False

    # Profound hypoxia (SpO2 < 85%)
    if pd.notna(spo2) and spo2 < 85.0:
        triggered_rules.append("CRITICAL_HYPOXIA_L1")
        explanations.append(f"Severe life-threatening hypoxia (SpO2 {spo2}% < 85%). Immediate airway/oxygen resuscitation required.")
        level_1_triggered = True

    # Profound shock: SBP < 70 or Shock Index > 1.3
    if pd.notna(sbp) and sbp > 0:
        if sbp < 70.0:
            triggered_rules.append("PROFOUND_SHOCK_HYPOTENSION_L1")
            explanations.append(f"Profound circulatory shock (SBP {sbp} mmHg < 70). Immediate resuscitation.")
            level_1_triggered = True
        elif pd.notna(pulse) and pulse > 0 and (pulse / sbp) >= 1.3:
            triggered_rules.append("ELEVATED_SHOCK_INDEX_L1")
            explanations.append(f"Critical shock index (HR/SBP = {pulse/sbp:.2f} >= 1.3). Impending circulatory collapse.")
            level_1_triggered = True

    # Extreme bradycardia (< 35 bpm)
    if pd.notna(pulse) and pulse < 35.0:
        triggered_rules.append("EXTREME_BRADYCARDIA_L1")
        explanations.append(f"Extreme life-threatening bradycardia (Pulse {pulse} bpm < 35).")
        level_1_triggered = True

    # Coma / unresponsiveness / active seizure
    if "seizure" in present_symptoms or "seizure" in nlp_rfs:
        triggered_rules.append("ACTIVE_SEIZURE_L1")
        explanations.append("Active or repeated seizures reported. Immediate resuscitation protocol.")
        level_1_triggered = True

    if "unconscious" in present_symptoms or ("altered_mentation" in present_symptoms and "cyanosis" in present_symptoms):
        triggered_rules.append("UNRESPONSIVE_COMA_L1")
        explanations.append("Severe unresponsiveness / coma with hypoxia.")
        level_1_triggered = True

    if level_1_triggered:
        target_level = 1
        target_dept = "Emergency"

    # -------------------------------------------------------------
    # 2. LEVEL 2: EMERGENT / HIGH RISK SITUATION
    # -------------------------------------------------------------
    level_2_triggered = False

    # Respiratory distress: SpO2 < 92% or severe dyspnea
    if pd.notna(spo2) and 85.0 <= spo2 < 92.0:
        triggered_rules.append("HYPOXIA_L2")
        explanations.append(f"Significant hypoxia (SpO2 {spo2}% < 92%). Urgent respiratory stabilization required.")
        level_2_triggered = True

    if "severe_dyspnea" in present_symptoms or "severe_respiratory_pattern" in nlp_rfs or "cyanosis" in present_symptoms:
        triggered_rules.append("RESPIRATORY_DISTRESS_L2")
        explanations.append("Severe respiratory distress / dyspnea / cyanosis reported.")
        level_2_triggered = True

    # Acute Coronary Syndrome (ACS)
    if "acs_pattern" in nlp_rfs or ("chest_pain" in present_symptoms and ("radiation_to_arm" in present_symptoms or "diaphoresis" in present_symptoms)):
        triggered_rules.append("ACS_CHEST_PAIN_L2")
        explanations.append("Suspected Acute Coronary Syndrome (chest pain with radiation/sweating). Immediate ECG and cardiac evaluation.")
        level_2_triggered = True

    # Acute Stroke
    if "stroke_pattern" in nlp_rfs or any(s in present_symptoms for s in ["facial_droop", "slurred_speech", "unilateral_weakness"]):
        triggered_rules.append("ACUTE_STROKE_L2")
        explanations.append("Acute neurological deficit (facial droop / speech difficulty / unilateral weakness). Code Stroke protocol.")
        level_2_triggered = True

    # Sepsis shock risk
    if "sepsis_pattern" in nlp_rfs or (pd.notna(temp) and (temp >= 38.5 or temp < 35.5) and ("altered_mentation" in present_symptoms or (pd.notna(sbp) and sbp < 90))):
        triggered_rules.append("SUSPECTED_SEPSIS_L2")
        explanations.append("Suspected severe sepsis (fever/hypothermia with altered mentation or hypotension).")
        level_2_triggered = True

    # Major bleeding / GI hemorrhage / Trauma
    if any(s in present_symptoms for s in ["hematemesis", "melena", "major_bleeding"]):
        triggered_rules.append("ACUTE_HEMORRHAGE_L2")
        explanations.append("Active or significant hemorrhage (hematemesis, melena, or profuse bleeding).")
        level_2_triggered = True

    # Poisoning / Chemical ingestion
    if "poisoning" in present_symptoms or "poisoning" in nlp_rfs:
        triggered_rules.append("ACUTE_POISONING_L2")
        explanations.append("Acute toxic ingestion / pesticide / chemical poisoning.")
        level_2_triggered = True

    # Anaphylaxis
    if "anaphylaxis" in present_symptoms or "anaphylaxis" in nlp_rfs:
        triggered_rules.append("ANAPHYLAXIS_L2")
        explanations.append("Severe allergic reaction / suspected anaphylaxis with respiratory/mucosal involvement.")
        level_2_triggered = True

    # Pediatric high risk: Infant <= 1 year with fever and poor feeding or lethargy
    if age <= 1.0 and pd.notna(temp) and temp >= 38.0 and any(s in present_symptoms for s in ["poor_feeding", "lethargic_child", "altered_mentation"]):
        triggered_rules.append("PEDIATRIC_FEBRILE_LETHARGY_L2")
        explanations.append(f"Febrile infant (age {age} yr, temp {temp} C) with lethargy or poor feeding.")
        level_2_triggered = True

    # Hemodynamic instability: Adult SBP < 85 or Pulse > 135
    if pd.notna(sbp) and 70.0 <= sbp < 85.0 and age >= 18:
        triggered_rules.append("HYPOTENSION_L2")
        explanations.append(f"Significant hypotension (SBP {sbp} mmHg < 85).")
        level_2_triggered = True

    if pd.notna(pulse) and pulse > 135.0:
        triggered_rules.append("MARKED_TACHYCARDIA_L2")
        explanations.append(f"Marked tachycardia (Pulse {pulse} bpm > 135).")
        level_2_triggered = True

    # Hypertensive Emergency: SBP >= 210 or DBP >= 125 with symptoms
    if (pd.notna(sbp) and sbp >= 210) or (pd.notna(dbp) and dbp >= 125):
        if any(s in present_symptoms for s in ["headache", "chest_pain", "dyspnea"]):
            triggered_rules.append("HYPERTENSIVE_EMERGENCY_L2")
            explanations.append(f"Hypertensive crisis (BP {sbp}/{dbp} mmHg) with end-organ symptoms.")
            level_2_triggered = True

    if level_2_triggered and target_level != 1:
        target_level = 2
        target_dept = "Emergency"

    # -------------------------------------------------------------
    # 3. ABSTENTION CHECK
    # -------------------------------------------------------------
    should_abstain, abstain_reasons = evaluate_abstention_criteria(patient_record)
    if should_abstain:
        triggered_rules.extend(["DATA_QUALITY_ABSTENTION"])
        explanations.extend(abstain_reasons)

    return SafetyRuleResult(
        override_level=target_level,
        override_department=target_dept,
        should_abstain=should_abstain,
        triggered_rules=triggered_rules,
        explanations=explanations,
    )


def apply_safety_overrides(
    pred_level: int,
    pred_dept: str,
    safety_result: SafetyRuleResult,
) -> Tuple[int, str, int, List[str]]:
    """
    Combines ML predictions with hard-coded safety overrides.
    Returns: (final_level, final_dept, abstain_flag, explanations)
    """
    final_level = pred_level
    final_dept = pred_dept
    abstain_flag = 1 if safety_result.should_abstain else 0

    # Overrides can only ESCALATE urgency (lower level number = higher urgency)
    if safety_result.override_level is not None:
        if safety_result.override_level < final_level:
            final_level = safety_result.override_level
        if safety_result.override_department:
            final_dept = safety_result.override_department

    return final_level, final_dept, abstain_flag, safety_result.explanations
