"""
modules/evaluation.py
Comprehensive Evaluation and Prediction Pipeline for OPD Triage.
Combines:
  1. Preprocessing (physiological validation & cleaning)
  2. Clinical NLP (symptom extraction & negation verification)
  3. ML Model (Random Forest with cost-sensitive decision rule)
  4. Hard-coded Safety Engine (mandatory clinical escalation & abstention)
Generates predictions.csv and benchmarking reports using evaluate.py.
"""

import sys
import os
import subprocess
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

from modules.preprocessing import preprocess_patient_data
from modules.nlp import batch_extract_nlp_features, parse_clinical_text
from modules.red_flags import evaluate_safety_rules, apply_safety_overrides
from modules.ml_model import load_models, prepare_features, predict_cost_sensitive


def generate_predictions_for_dataset(
    intake_csv_path: str = "data/intake_records.csv",
    output_predictions_path: str = "predictions.csv",
    batch_size: int = 5000,
) -> pd.DataFrame:
    """
    Processes all patient records through preprocessing, NLP, ML, and Safety Overrides.
    Outputs predictions.csv formatted for evaluate.py:
      patient_id, triage_level, department, abstain
    """
    print(f"Loading intake data from {intake_csv_path}...")
    intake_df = pd.read_csv(intake_csv_path)

    print("Loading trained models...")
    triage_artifact, dept_artifact = load_models()
    rf_triage = triage_artifact["model"]
    rf_dept = dept_artifact["model"]
    tfidf = triage_artifact["tfidf"]
    classes = triage_artifact["classes"]

    print("Step 1: Preprocessing physiological vitals...")
    preprocessed_df = preprocess_patient_data(intake_df)

    print("Step 2: Extracting clinical NLP features and symptom indicators...")
    nlp_df = batch_extract_nlp_features(preprocessed_df, text_column="chief_complaint")

    print("Step 3: Generating ML probability predictions...")
    X, _, _ = prepare_features(nlp_df, tfidf_vectorizer=tfidf, is_train=False)

    probas_triage = rf_triage.predict_proba(X)
    ml_levels = predict_cost_sensitive(probas_triage)
    ml_depts = rf_dept.predict(X)

    print("Step 4: Applying Safety Engine overrides and Abstention logic...")
    final_levels = []
    final_depts = []
    abstains = []
    override_counts = {"level_1": 0, "level_2": 0, "abstain": 0}

    # Process each patient through the safety engine
    # Extract records as list of dicts for rapid safety rule inspection
    records = nlp_df.to_dict(orient="records")

    for i, rec in enumerate(records):
        ml_lvl = int(ml_levels[i])
        ml_dept = str(ml_depts[i])

        # Parse complaint details for red flag safety rules
        parsed_complaint = parse_clinical_text(rec.get("chief_complaint", ""))
        rec["present_symptoms"] = parsed_complaint["present_symptoms"]
        rec["detected_red_flags"] = parsed_complaint["detected_red_flags"]

        # Evaluate safety rules
        safety_res = evaluate_safety_rules(rec)

        # Apply overrides
        fin_lvl, fin_dept, ab, exp = apply_safety_overrides(ml_lvl, ml_dept, safety_res)

        if ab == 1:
            override_counts["abstain"] += 1
        if fin_lvl < ml_lvl:
            if fin_lvl == 1:
                override_counts["level_1"] += 1
            elif fin_lvl == 2:
                override_counts["level_2"] += 1

        final_levels.append(fin_lvl)
        final_depts.append(fin_dept)
        abstains.append(ab)

    predictions_df = pd.DataFrame({
        "patient_id": intake_df["patient_id"],
        "triage_level": final_levels,
        "department": final_depts,
        "abstain": abstains,
    })

    predictions_df.to_csv(output_predictions_path, index=False)
    print(f"Predictions saved to {output_predictions_path} ({len(predictions_df)} records)")
    print(f"Safety Overrides applied: Level 1 overrides = {override_counts['level_1']}, Level 2 overrides = {override_counts['level_2']}, Abstentions = {override_counts['abstain']}")

    return predictions_df


def run_benchmark_evaluation(
    predictions_path: str = "predictions.csv",
    labels_path: str = "data/hidden_labels.csv",
    arrival_log_path: str = "data/arrival_log.csv",
    report_output_path: str = "reports/evaluation_report.md",
) -> str:
    """
    Executes evaluate.py against predictions and generates a markdown report.
    """
    cmd = [sys.executable, "evaluate.py", predictions_path, labels_path, arrival_log_path]
    print(f"Running evaluation benchmark: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)

    eval_output = proc.stdout
    print(eval_output)
    if proc.stderr:
        print("STDERR:", proc.stderr)

    os.makedirs(os.path.dirname(report_output_path), exist_ok=True)
    with open(report_output_path, "w", encoding="utf-8") as f:
        f.write("# Phase 6: System Evaluation & Benchmark Report\n\n")
        f.write("Evaluated with stand-in benchmark `evaluate.py`.\n\n")
        f.write("```text\n")
        f.write(eval_output)
        f.write("\n```\n")

    return eval_output
