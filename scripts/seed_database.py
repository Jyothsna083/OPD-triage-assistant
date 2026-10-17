"""
scripts/seed_database.py
Seeds the SQLite database with initial active OPD walk-in patients.
Runs them through Preprocessing, Clinical NLP, ML Model, and Safety Overrides.
"""

import sys
import os
import json
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.database import init_db, save_patient_and_assessment
from modules.preprocessing import preprocess_patient_data
from modules.nlp import parse_clinical_text
from modules.red_flags import evaluate_safety_rules, apply_safety_overrides
from modules.ml_model import load_models, prepare_features, predict_cost_sensitive

def seed(n_patients: int = 50):
    print("Initializing SQLite database...")
    init_db(reset=True)

    print("Loading models and intake sample...")
    triage_artifact, dept_artifact = load_models()
    rf_triage = triage_artifact["model"]
    rf_dept = dept_artifact["model"]
    tfidf = triage_artifact["tfidf"]

    intake_df = pd.read_csv("data/intake_records.csv").head(n_patients)
    preprocessed_df = preprocess_patient_data(intake_df)

    X, _, _ = prepare_features(preprocessed_df, tfidf_vectorizer=tfidf, is_train=False)
    probas = rf_triage.predict_proba(X)
    ml_levels = predict_cost_sensitive(probas)
    ml_depts = rf_dept.predict(X)

    records = preprocessed_df.to_dict(orient="records")

    for i, rec in enumerate(records):
        ml_lvl = int(ml_levels[i])
        ml_dept = str(ml_depts[i])

        parsed_complaint = parse_clinical_text(rec.get("chief_complaint", ""))
        rec["present_symptoms"] = parsed_complaint["present_symptoms"]
        rec["detected_red_flags"] = parsed_complaint["detected_red_flags"]

        safety_res = evaluate_safety_rules(rec)
        fin_lvl, fin_dept, ab, exp = apply_safety_overrides(ml_lvl, ml_dept, safety_res)

        prob_dict = {f"Level_{c}": round(float(p), 4) for c, p in zip(triage_artifact["classes"], probas[i])}

        ai_result = {
            "ai_triage_level": ml_lvl,
            "ai_department": ml_dept,
            "ai_abstain": ab,
            "final_triage_level": fin_lvl,
            "final_department": fin_dept,
            "detected_red_flags": safety_res.triggered_rules,
            "explanations": exp,
            "level_probabilities": prob_dict,
        }

        # Original raw record for database
        raw_row = intake_df.iloc[i].to_dict()
        raw_row["name"] = f"Walk-in Patient #{i+1:03d} ({raw_row.get('sex', 'U')}, {int(raw_row.get('age', 35))})"
        save_patient_and_assessment(raw_row, ai_result)

    print(f"Successfully seeded {len(intake_df)} patients into data/opd_triage.db!")

if __name__ == "__main__":
    seed(50)
