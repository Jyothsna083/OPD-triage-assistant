"""
modules/ml_model.py
Machine Learning Pipeline for OPD Triage and Department Routing.
Includes:
  - Feature extraction combining physiological vitals, demographics, NLP symptom indicators, and TF-IDF
  - Cost-sensitive Bayes risk decision rule (20x penalty for under-triaging emergencies)
  - Random Forest classifiers for triage level and department
  - Leakage protection (never uses true hidden metadata at inference)
"""

import os
from typing import Dict, Any, Tuple, List, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from modules.preprocessing import preprocess_patient_data, derive_physiological_features
from modules.nlp import batch_extract_nlp_features, ClinicalTextVectorizer


TRIAGE_MODEL_PATH = os.path.join("models", "triage_model.joblib")
DEPT_MODEL_PATH = os.path.join("models", "department_model.joblib")

# Cost matrix C[pred, true]
# Row = pred (1..5), Col = true (1..5)
def get_cost_matrix() -> np.ndarray:
    C = np.zeros((5, 5))
    for pred_idx in range(5):
        pred_lvl = pred_idx + 1
        for true_idx in range(5):
            true_lvl = true_idx + 1
            gap = pred_lvl - true_lvl
            is_emerg = (true_lvl <= 2)
            if gap > 0: # under-triage
                multiplier = 20.0 if is_emerg else 4.0
                C[pred_idx, true_idx] = gap * multiplier
            elif gap < 0: # over-triage
                C[pred_idx, true_idx] = -gap * 1.0
            else:
                C[pred_idx, true_idx] = 0.0
    return C

COST_MATRIX = get_cost_matrix()


def predict_cost_sensitive(proba: np.ndarray, cost_matrix: np.ndarray = COST_MATRIX) -> np.ndarray:
    """
    Given class probabilities for classes [1, 2, 3, 4, 5],
    selects the class that minimizes expected cost:
    k* = argmin_k sum_j ( P(j) * C[k, j] )
    """
    expected_costs = np.dot(proba, cost_matrix.T)
    best_class_idx = np.argmin(expected_costs, axis=1)
    return best_class_idx + 1 # 1-indexed levels


def prepare_features(
    df: pd.DataFrame,
    tfidf_vectorizer: Optional[ClinicalTextVectorizer] = None,
    is_train: bool = False,
) -> Tuple[np.ndarray, List[str], Optional[ClinicalTextVectorizer]]:
    """
    Extracts tabular features + NLP symptom flags + TF-IDF embeddings.
    Strictly excludes all ground truth leakage columns.
    """
    # 1. Preprocess physiological vitals
    if "temp_c_clean" not in df.columns:
        work_df = preprocess_patient_data(df)
    else:
        work_df = derive_physiological_features(df.copy())

    # 2. Extract NLP symptoms & flags if not present
    if "symptom_chest_pain" not in work_df.columns:
        work_df = batch_extract_nlp_features(work_df, text_column="chief_complaint")

    # 3. Numeric tabular columns
    feature_cols = [
        "age", "is_infant", "is_pediatric", "is_child", "is_elderly",
        "temp_c_clean", "sbp_clean", "dbp_clean", "pulse_clean", "spo2_clean",
        "shock_index", "map", "pulse_pressure", "n_vitals_unusable",
        "flag_severe_hypoxia", "flag_moderate_hypoxia", "flag_hypotension",
        "flag_ped_hypotension", "flag_hypertensive_crisis", "flag_tachycardia",
        "flag_extreme_tachycardia", "flag_bradycardia", "flag_severe_bradycardia",
        "flag_high_fever", "flag_fever", "flag_hypothermia",
        "hist_hypertension", "hist_diabetes", "hist_cardiac", "hist_asthma_copd",
        "hist_ckd", "hist_stroke", "hist_pregnancy",
        "nlp_is_acute", "nlp_n_red_flags",
        "symptom_chest_pain", "symptom_severe_dyspnea", "symptom_dyspnea",
        "symptom_wheeze", "symptom_cyanosis", "symptom_facial_droop",
        "symptom_slurred_speech", "symptom_unilateral_weakness", "symptom_seizure",
        "symptom_altered_mentation", "symptom_syncope", "symptom_hematemesis",
        "symptom_melena", "symptom_major_bleeding", "symptom_head_injury",
        "symptom_poisoning", "symptom_anaphylaxis", "symptom_fever",
        "symptom_abdominal_pain", "symptom_fracture_suspected", "symptom_poor_feeding"
    ]

    # Ensure all columns exist in work_df
    for col in feature_cols:
        if col not in work_df.columns:
            work_df[col] = 0

    # Impute missing values for tabular columns
    X_num = work_df[feature_cols].copy()
    med_vals = {
        "temp_c_clean": 37.0,
        "sbp_clean": 120.0,
        "dbp_clean": 80.0,
        "pulse_clean": 78.0,
        "spo2_clean": 98.0,
        "shock_index": 0.65,
        "map": 93.3,
        "pulse_pressure": 40.0,
        "nlp_duration_hours": 24.0,
    }
    for col, v in med_vals.items():
        if col in X_num.columns:
            X_num[col] = X_num[col].fillna(v)
    X_num = X_num.fillna(0)

    # 4. Text features via TF-IDF
    texts = work_df["chief_complaint"].fillna("").astype(str).tolist()
    if is_train:
        tfidf = ClinicalTextVectorizer(max_features=120, ngram_range=(1, 2))
        X_tfidf = tfidf.fit_transform(texts)
    else:
        if tfidf_vectorizer is not None:
            X_tfidf = tfidf_vectorizer.transform(texts)
        else:
            X_tfidf = np.zeros((len(df), 120))
        tfidf = tfidf_vectorizer

    X_all = np.hstack([X_num.values, X_tfidf])
    all_col_names = feature_cols + [f"tfidf_{i}" for i in range(X_tfidf.shape[1])]

    return X_all, all_col_names, tfidf


def train_models(
    dataset_path: str = "data/processed/dev_dataset.csv",
    save_models: bool = True
) -> Dict[str, Any]:
    """
    Trains Triage Level and Department classifiers.
    Compares Random Forest with Baseline (Logistic Regression).
    Evaluates cost-sensitive Bayes decision rule on held-out 20% validation split.
    Saves models to models/ directory.
    """
    os.makedirs("models", exist_ok=True)
    df = pd.read_csv(dataset_path)

    # Split into train (80%) and validation (20%)
    train_df, val_df = train_test_split(
        df, test_size=0.20, random_state=42, stratify=df["y_triage_level"]
    )

    # Feature extraction
    print("Extracting features on training split...")
    X_train, feature_names, tfidf = prepare_features(train_df, is_train=True)
    print("Extracting features on validation split...")
    X_val, _, _ = prepare_features(val_df, tfidf_vectorizer=tfidf, is_train=False)

    y_train_triage = train_df["y_triage_level"].values
    y_val_triage = val_df["y_triage_level"].values

    y_train_dept = train_df["y_department"].values
    y_val_dept = val_df["y_department"].values

    # 1. Baseline: Logistic Regression
    print("Training Baseline Logistic Regression for Triage Level...")
    baseline_lr = LogisticRegression(max_iter=500, random_state=42)
    baseline_lr.fit(X_train, y_train_triage)
    val_pred_lr_standard = baseline_lr.predict(X_val)
    val_acc_lr = accuracy_score(y_val_triage, val_pred_lr_standard)

    # 2. Random Forest Triage Model
    print("Training Random Forest Classifier for Triage Level...")
    rf_triage = RandomForestClassifier(
        n_estimators=100,
        max_depth=16,
        min_samples_split=4,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    rf_triage.fit(X_train, y_train_triage)

    val_proba_rf = rf_triage.predict_proba(X_val)
    val_pred_rf_standard = rf_triage.predict(X_val)
    val_acc_rf_standard = accuracy_score(y_val_triage, val_pred_rf_standard)

    # Cost-sensitive prediction
    val_pred_rf_cost = predict_cost_sensitive(val_proba_rf)
    val_acc_rf_cost = accuracy_score(y_val_triage, val_pred_rf_cost)

    emerg_mask = (y_val_triage <= 2)

    print("\n--- Validation Performance Comparison ---")
    print(f"Baseline Logistic Regression Acc: {val_acc_lr:.4f}")
    print(f"Random Forest (Standard Argmax) Acc: {val_acc_rf_standard:.4f}")
    print(f"Random Forest (Cost-Sensitive Bayes) Acc: {val_acc_rf_cost:.4f}")
    print(f"Emergencies missed under standard RF: {((val_pred_rf_standard > 2) & emerg_mask).sum()} of {emerg_mask.sum()}")
    print(f"Emergencies missed under cost-sensitive RF: {((val_pred_rf_cost > 2) & emerg_mask).sum()} of {emerg_mask.sum()}")

    # 3. Department Classifier
    print("\nTraining Department Classifier...")
    rf_dept = RandomForestClassifier(
        n_estimators=100,
        max_depth=18,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1,
    )
    rf_dept.fit(X_train, y_train_dept)
    val_pred_dept = rf_dept.predict(X_val)
    val_dept_acc = accuracy_score(y_val_dept, val_pred_dept)
    print(f"Department Classifier Validation Accuracy: {val_dept_acc:.4f}")

    # Package artifacts
    triage_artifact = {
        "model": rf_triage,
        "baseline_model": baseline_lr,
        "tfidf": tfidf,
        "feature_names": feature_names,
        "classes": rf_triage.classes_,
        "cost_matrix": COST_MATRIX,
    }

    dept_artifact = {
        "model": rf_dept,
        "classes": rf_dept.classes_,
    }

    if save_models:
        joblib.dump(triage_artifact, TRIAGE_MODEL_PATH)
        joblib.dump(dept_artifact, DEPT_MODEL_PATH)
        print(f"Saved models to {TRIAGE_MODEL_PATH} and {DEPT_MODEL_PATH}")

    return {
        "triage_artifact": triage_artifact,
        "dept_artifact": dept_artifact,
        "val_metrics": {
            "baseline_lr_acc": float(val_acc_lr),
            "rf_standard_acc": float(val_acc_rf_standard),
            "rf_cost_acc": float(val_acc_rf_cost),
            "dept_acc": float(val_dept_acc),
        }
    }


def load_models() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Loads saved models from disk."""
    triage_artifact = joblib.load(TRIAGE_MODEL_PATH)
    dept_artifact = joblib.load(DEPT_MODEL_PATH)
    return triage_artifact, dept_artifact


def predict_patient(
    patient_data: Dict[str, Any],
    triage_artifact: Optional[Dict[str, Any]] = None,
    dept_artifact: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Generates model predictions for a single patient dictionary.
    Returns:
      - raw_pred_level: int
      - raw_pred_dept: str
      - probas: dict of level probabilities
    """
    if triage_artifact is None or dept_artifact is None:
        triage_artifact, dept_artifact = load_models()

    df = pd.DataFrame([patient_data])
    X, _, _ = prepare_features(df, tfidf_vectorizer=triage_artifact["tfidf"], is_train=False)

    rf_triage = triage_artifact["model"]
    rf_dept = dept_artifact["model"]

    probas = rf_triage.predict_proba(X)[0]
    classes = triage_artifact["classes"]
    prob_dict = {f"Level_{c}": round(float(p), 4) for c, p in zip(classes, probas)}

    # Cost-sensitive prediction
    pred_level = int(predict_cost_sensitive(np.array([probas]))[0])
    pred_dept = str(rf_dept.predict(X)[0])

    return {
        "raw_pred_level": pred_level,
        "raw_pred_dept": pred_dept,
        "level_probabilities": prob_dict,
    }
