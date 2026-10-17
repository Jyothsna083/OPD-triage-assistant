"""
modules/preprocessing.py
Robust Preprocessing Pipeline for OPD Triage Assistant.
Handles single patient records (dict) or batches (pd.DataFrame).
Validates vitals, detects unit errors (e.g. Fahrenheit in Celsius),
corrects swapped blood pressures, tracks unusable vitals,
and derives age-adjusted physiological markers.
"""

from typing import Union, Dict, Any
import numpy as np
import pandas as pd


def clean_temperature(val):
    """
    Cleans temperature value.
    Detects Fahrenheit input (>= 90) and converts to Celsius.
    Checks plausible human temperature range [30.0, 45.0] C.
    Returns (cleaned_temp_c, status).
    """
    if val is None or pd.isna(val):
        return np.nan, "missing"
    try:
        f_val = float(val)
    except (ValueError, TypeError):
        return np.nan, "invalid"

    if f_val >= 90.0:
        c_val = round((f_val - 32.0) * 5.0 / 9.0, 1)
        if 30.0 <= c_val <= 45.0:
            return c_val, "converted_from_F"
        return np.nan, "invalid_out_of_range"
    elif 30.0 <= f_val <= 45.0:
        return round(f_val, 1), "ok"
    else:
        return np.nan, "invalid_out_of_range"


def clean_blood_pressure(sbp, dbp):
    """
    Cleans systolic and diastolic blood pressure.
    Detects swapped values (dbp > sbp), zero BP, and out-of-range values.
    Returns (sbp_clean, dbp_clean, status).
    """
    if pd.isna(sbp) or pd.isna(dbp) or sbp is None or dbp is None:
        return np.nan, np.nan, "missing"
    try:
        s = float(sbp)
        d = float(dbp)
    except (ValueError, TypeError):
        return np.nan, np.nan, "invalid"

    if s <= 0 or d <= 0:
        return np.nan, np.nan, "invalid_zero"

    # Swapped detection
    if d > s:
        s, d = d, s
        status = "swapped_fixed"
    else:
        status = "ok"

    # Check physiological plausibility
    if s < 40 or s > 300 or d < 20 or d > 200:
        return np.nan, np.nan, "invalid_out_of_range"

    return round(s, 1), round(d, 1), status


def clean_pulse(val):
    """
    Cleans heart rate / pulse.
    Plausible range: 20 - 250 bpm.
    Returns (pulse_clean, status).
    """
    if val is None or pd.isna(val):
        return np.nan, "missing"
    try:
        p = float(val)
    except (ValueError, TypeError):
        return np.nan, "invalid"

    if p <= 0:
        return np.nan, "invalid_zero"
    if 20 <= p <= 250:
        return round(p, 1), "ok"
    return np.nan, "invalid_out_of_range"


def clean_spo2(val):
    """
    Cleans pulse oximetry SpO2.
    Plausible range: 50% - 100%.
    Returns (spo2_clean, status).
    """
    if val is None or pd.isna(val):
        return np.nan, "missing"
    try:
        s = float(val)
    except (ValueError, TypeError):
        return np.nan, "invalid"

    if s <= 0:
        return np.nan, "invalid_zero"
    if 50 <= s <= 100:
        return round(s, 1), "ok"
    return np.nan, "invalid_out_of_range"


def derive_physiological_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes derived physiological metrics, age groups, and abnormal vital flags.
    """
    # Age brackets
    age = df["age"].fillna(35).astype(float)
    df["is_infant"] = (age <= 1).astype(int)
    df["is_pediatric"] = (age < 18).astype(int)
    df["is_child"] = ((age > 1) & (age <= 12)).astype(int)
    df["is_elderly"] = (age >= 65).astype(int)

    # Shock index: HR / SBP
    df["shock_index"] = np.where(
        (df["sbp_clean"] > 0) & (df["pulse_clean"] > 0),
        (df["pulse_clean"] / df["sbp_clean"]).round(2),
        np.nan
    )
    df["elevated_shock_index"] = (df["shock_index"] >= 0.9).astype(int)

    # Mean Arterial Pressure (MAP) = (2 * DBP + SBP) / 3
    df["map"] = np.where(
        (df["sbp_clean"] > 0) & (df["dbp_clean"] > 0),
        ((2 * df["dbp_clean"] + df["sbp_clean"]) / 3.0).round(1),
        np.nan
    )

    # Pulse pressure = SBP - DBP
    df["pulse_pressure"] = np.where(
        (df["sbp_clean"] > 0) & (df["dbp_clean"] > 0),
        (df["sbp_clean"] - df["dbp_clean"]).round(1),
        np.nan
    )

    # Critical vital flags
    df["flag_severe_hypoxia"] = (df["spo2_clean"] < 90).astype(int)
    df["flag_moderate_hypoxia"] = ((df["spo2_clean"] >= 90) & (df["spo2_clean"] < 95)).astype(int)

    # Adult hypotension
    df["flag_hypotension"] = ((age >= 18) & (df["sbp_clean"] < 90)).astype(int)
    # Pediatric hypotension (age adjusted)
    ped_sbp_thresh = np.where(age < 1, 65, 70 + 2 * age)
    df["flag_ped_hypotension"] = ((age < 18) & (df["sbp_clean"] < ped_sbp_thresh)).astype(int)

    # Severe hypertension
    df["flag_hypertensive_crisis"] = ((df["sbp_clean"] >= 180) | (df["dbp_clean"] >= 110)).astype(int)

    # Heart rate abnormalities (age-adjusted)
    is_tachy = np.where(
        age <= 1,
        df["pulse_clean"] > 160,
        np.where(age <= 12, df["pulse_clean"] > 130, df["pulse_clean"] > 100)
    )
    df["flag_tachycardia"] = np.nan_to_num(is_tachy, nan=0).astype(int)
    df["flag_extreme_tachycardia"] = (df["pulse_clean"] >= 140).astype(int)
    df["flag_bradycardia"] = (df["pulse_clean"] < 50).astype(int)
    df["flag_severe_bradycardia"] = (df["pulse_clean"] < 40).astype(int)

    # Temperature flags
    df["flag_high_fever"] = (df["temp_c_clean"] >= 39.0).astype(int)
    df["flag_fever"] = ((df["temp_c_clean"] >= 38.0) & (df["temp_c_clean"] < 39.0)).astype(int)
    df["flag_hypothermia"] = (df["temp_c_clean"] < 35.5).astype(int)

    # Medical history parsing
    cond_str = df["known_conditions"].fillna("").astype(str).str.lower()
    df["hist_hypertension"] = cond_str.str.contains(r"hypertension|htn|high bp").astype(int)
    df["hist_diabetes"] = cond_str.str.contains(r"diabetes|dm|sugar").astype(int)
    df["hist_cardiac"] = cond_str.str.contains(r"cad|cardiac|heart|angina|infarction|stent").astype(int)
    df["hist_asthma_copd"] = cond_str.str.contains(r"asthma|copd|bronchitis|respiratory").astype(int)
    df["hist_ckd"] = cond_str.str.contains(r"kidney|renal|ckd|dialysis").astype(int)
    df["hist_stroke"] = cond_str.str.contains(r"stroke|cva|paralysis|tia").astype(int)
    df["hist_pregnancy"] = cond_str.str.contains(r"pregnant|pregnancy|gravida|trimester").astype(int)

    return df


def preprocess_patient_data(data: Union[Dict[str, Any], pd.DataFrame]) -> pd.DataFrame:
    """
    Main entry point for preprocessing patient records.
    Accepts single record (dict) or batch DataFrame.
    Preserves all original input fields, produces cleaned vitals,
    status flags, and derived physiological features.
    """
    is_single = isinstance(data, dict)
    if is_single:
        df = pd.DataFrame([data])
    else:
        df = data.copy()

    # Ensure required columns exist
    defaults = {
        "patient_id": "P_UNKNOWN",
        "age": 35,
        "sex": "Unknown",
        "bp_systolic": np.nan,
        "bp_diastolic": np.nan,
        "pulse": np.nan,
        "temp_c": np.nan,
        "spo2": np.nan,
        "chief_complaint": "",
        "known_conditions": "",
        "medication_list": "",
        "language_style": "english",
    }
    for col, default_val in defaults.items():
        if col not in df.columns:
            df[col] = default_val

    # Clean temperature
    temp_clean = []
    temp_stat = []
    for val in df["temp_c"]:
        c, s = clean_temperature(val)
        temp_clean.append(c)
        temp_stat.append(s)
    df["temp_c_clean"] = temp_clean
    df["temp_status"] = temp_stat

    # Clean BP
    sbp_clean = []
    dbp_clean = []
    bp_stat = []
    for s, d in zip(df["bp_systolic"], df["bp_diastolic"]):
        sc, dc, st = clean_blood_pressure(s, d)
        sbp_clean.append(sc)
        dbp_clean.append(dc)
        bp_stat.append(st)
    df["sbp_clean"] = sbp_clean
    df["dbp_clean"] = dbp_clean
    df["bp_status"] = bp_stat

    # Clean Pulse
    p_clean = []
    p_stat = []
    for val in df["pulse"]:
        c, s = clean_pulse(val)
        p_clean.append(c)
        p_stat.append(s)
    df["pulse_clean"] = p_clean
    df["pulse_status"] = p_stat

    # Clean SpO2
    sp_clean = []
    sp_stat = []
    for val in df["spo2"]:
        c, s = clean_spo2(val)
        sp_clean.append(c)
        sp_stat.append(s)
    df["spo2_clean"] = sp_clean
    df["spo2_status"] = sp_stat

    # Number of unusable vitals (missing or invalid)
    vitals_unusable = (
        df["temp_c_clean"].isna().astype(int) +
        df["sbp_clean"].isna().astype(int) +
        df["pulse_clean"].isna().astype(int) +
        df["spo2_clean"].isna().astype(int)
    )
    df["n_vitals_unusable"] = vitals_unusable

    # Derive physiological metrics
    df = derive_physiological_features(df)

    return df
