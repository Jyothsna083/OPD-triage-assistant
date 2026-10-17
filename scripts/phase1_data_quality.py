"""Phase 1: data-quality report + clean development dataset.
Originals in data/ are never modified.
Outputs: reports/data_quality_report.md, data/processed/dev_dataset.csv
Run from the project root:  python scripts/phase1_data_quality.py
"""
import os
import numpy as np
import pandas as pd

D = "data"
os.makedirs(f"{D}/processed", exist_ok=True)
os.makedirs("reports", exist_ok=True)
i = pd.read_csv(f"{D}/intake_records.csv")
h = pd.read_csv(f"{D}/hidden_labels.csv")
a = pd.read_csv(f"{D}/arrival_log.csv")
R = []


def sec(t):
    R.append(f"\n## {t}\n")


def line(s):
    R.append(s)


# ---- 1. structure
sec("1. Structure, types, missing values, duplicates")
for n, d in [("intake_records", i), ("hidden_labels", h), ("arrival_log", a)]:
    line(f"- **{n}**: {d.shape[0]} rows x {d.shape[1]} cols; duplicate rows={d.duplicated().sum()}; "
         f"duplicate patient_id={d.patient_id.duplicated().sum()}")
miss = i.isna().sum()
miss = miss[miss > 0]
line("- Missing in intake: " + ", ".join(f"{k}={v} ({v/len(i):.1%})" for k, v in miss.items()))
line(f"- Distinct complaint strings: {i.chief_complaint.nunique()} of {len(i)} (templated text; repeats are not duplicate patients)")

# ---- 2. ids
sec("2. ID consistency")
s1, s2, s3 = set(i.patient_id), set(h.patient_id), set(a.patient_id)
same_order = (i.patient_id == h.patient_id).all() and (i.patient_id == a.patient_id).all()
line(f"- ID sets identical across all 3 files: **{s1 == s2 == s3}**; same row order: {same_order}")

# ---- 3. vitals
sec("3. Vital-sign validity (raw)")
rows = []
for c, lo, hi in [("bp_systolic", 50, 260), ("bp_diastolic", 20, 160), ("pulse", 20, 220), ("spo2", 50, 100), ("temp_c", 30, 45)]:
    s = i[c]
    rows.append(f"| {c} | {s.isna().sum()} | {(s == 0).sum()} | {((s > 0) & ((s < lo) | (s > hi))).sum()} | {s[s > 0].min()} | {s.max()} |")
line("| vital | missing | exactly 0 | outside plausible range | min(>0) | max |\n|---|---|---|---|---|---|\n" + "\n".join(rows))
line(f"\n- temp_c >= 90 (Fahrenheit typed into a Celsius field): **{(i.temp_c >= 90).sum()}**; "
     f"values between 43 and 90: {((i.temp_c > 43) & (i.temp_c < 90)).sum()}, so the split is clean.")
line(f"- diastolic > systolic (swapped): {((i.bp_diastolic > i.bp_systolic) & (i.bp_systolic > 0)).sum()}; "
     f"equal and nonzero: {((i.bp_diastolic == i.bp_systolic) & (i.bp_systolic > 0)).sum()}")
line(f"- Age {i.age.min()}-{i.age.max()}; {(i.age < 1).sum()} records aged 0; {(i.age < 18).sum()} under 18. "
     "Pulse/BP thresholds must be age-adjusted.")

# ---- 4. labels
sec("4. Labels")
line("Triage level counts: " + ", ".join(f"L{k}={v}" for k, v in h.triage_level.value_counts().sort_index().items()))
line(f"\nEmergencies: {h.is_emergency.sum()} ({h.is_emergency.mean():.2%}); "
     f"should_abstain: {h.should_abstain.sum()} ({h.should_abstain.mean():.2%})")
line("\nTriage level x is_emergency:\n\n" + pd.crosstab(h.triage_level, h.is_emergency).to_markdown())
line("\nDepartments: " + ", ".join(f"{k}={v}" for k, v in h.department.value_counts().items()))
line("\nDepartment==Emergency x is_emergency:\n\n" + pd.crosstab(h.department == "Emergency", h.is_emergency).to_markdown())
line(f"\n- All Level-1 are emergencies: {(h[h.triage_level == 1].is_emergency == 1).all()}; "
     f"no Level 3-5 is an emergency: {(h[h.triage_level >= 3].is_emergency == 0).all()}.")
line(f"- Level 2 splits into emergency={((h.triage_level == 2) & (h.is_emergency == 1)).sum()} and "
     f"non-emergency={((h.triage_level == 2) & (h.is_emergency == 0)).sum()}.")
line("- Flag rates: under_reported={:.1%}, over_reported={:.1%}, flags_visible_in_text={:.1%}".format(
    h.under_reported.mean(), h.over_reported.mean(), h.flags_visible_in_text.mean()))
line("- Emergency archetypes: " + ", ".join(f"{k}={v}" for k, v in h[h.is_emergency == 1].archetype.value_counts().items()))

# ---- 5. label-vs-metadata consistency
sec("5. Label / metadata consistency")
vi = h.vitals_issues.fillna("")


def has(k):
    return vi.str.contains(k)


m = i.merge(h, on="patient_id")
swapped_raw = (i.bp_diastolic > i.bp_systolic) & (i.bp_systolic > 0)
chk = {
    "fahrenheit/temp_96 vs temp_c>=90": ((has("fahrenheit") | has("temp_96")) == (i.temp_c >= 90)).mean(),
    "missing_temp vs NaN temp": (has("missing_temp") == i.temp_c.isna()).mean(),
    "missing_bp vs NaN bp": (has("missing_bp") == i.bp_systolic.isna()).mean(),
    "missing_pulse vs NaN pulse": (has("missing_pulse") == i.pulse.isna()).mean(),
    "missing_spo2 vs NaN spo2": (has("missing_spo2") == i.spo2.isna()).mean(),
    "zero_bp vs bp==0": (has("zero_bp") == (i.bp_systolic == 0)).mean(),
    "zero_pulse vs pulse==0": (has("zero_pulse") == (i.pulse == 0)).mean(),
    "zero_spo2 vs spo2==0": (has("zero_spo2") == (i.spo2 == 0)).mean(),
    "bp_swapped vs dbp>sbp": (has("swapped") == swapped_raw).mean(),
}
line("| check | agreement |\n|---|---|\n" + "\n".join(f"| {k} | {v:.4f} |" for k, v in chk.items()))
line(f"\n- should_abstain=1 always has a vitals issue: {h[h.should_abstain == 1].vitals_issues.notna().all()} "
     f"({h.should_abstain.sum()} abstain cases of {h.vitals_issues.notna().sum()} vitals-issue cases)")
ab = m[m.should_abstain == 1]
line(f"- Abstain cases: {ab.is_emergency.mean():.1%} are emergencies; by level: " +
     ", ".join(f"L{k}={v}" for k, v in ab.triage_level.value_counts().sort_index().items()))
line("- Pediatrics share among age<=12: {:.1%}".format((m[m.age <= 12].department == "Pediatrics").mean()))
line("- Language styles: " + ", ".join(f"{k}={v}" for k, v in i.language_style.value_counts().items()))

# ---- 6. arrival
sec("6. Arrival log")
g = a.groupby("date").size()
line(f"- {a.date.nunique()} days, median {g.median():.0f} arrivals/day (min {g.min()}, max {g.max()}); "
     f"consult_minutes mean {a.consult_minutes.mean():.2f}, max {a.consult_minutes.max()}")
line(f"- Sorted by time within each day: {a.groupby('date').arrival_sec_of_day.apply(lambda s: s.is_monotonic_increasing).all()}")
line("- Mean consult_minutes by true level: " + ", ".join(
    f"L{k}={v:.2f}" for k, v in a.merge(h, on="patient_id").groupby("triage_level").consult_minutes.mean().items()))

# ---- 7. leakage
sec("7. Leakage warning")
line("In `hidden_labels.csv`, only `triage_level`, `department`, `is_emergency` and `should_abstain` may be used as "
     "training/evaluation **targets**. `red_flags`, `archetype`, `vitals_issues`, `under_reported`, `over_reported` and "
     "`flags_visible_in_text` are answer keys and must never be inference-time inputs. Derive vitals problems from the raw values.")

# ---- clean dev dataset (raw columns preserved)
c = i.copy()
c["temp_status"] = np.where(c.temp_c.isna(), "missing", np.where(c.temp_c >= 90, "converted_from_F", "ok"))
c["temp_c_clean"] = np.where(c.temp_c >= 90, (c.temp_c - 32) * 5 / 9, c.temp_c).round(1)
c.loc[(c.temp_c_clean < 30) | (c.temp_c_clean > 45), "temp_c_clean"] = np.nan
sw = (c.bp_diastolic > c.bp_systolic) & (c.bp_systolic > 0)
c["bp_status"] = np.where(c.bp_systolic.isna(), "missing",
                          np.where(c.bp_systolic == 0, "invalid_zero", np.where(sw, "swapped_fixed", "ok")))
c["sbp_clean"] = np.where(sw, c.bp_diastolic, c.bp_systolic)
c["dbp_clean"] = np.where(sw, c.bp_systolic, c.bp_diastolic)
c.loc[c.bp_status == "invalid_zero", ["sbp_clean", "dbp_clean"]] = np.nan
for col, new, lo in [("pulse", "pulse_clean", 20), ("spo2", "spo2_clean", 50)]:
    c[f"{col}_status"] = np.where(c[col].isna(), "missing", np.where(c[col] < lo, "invalid_zero", "ok"))
    c[new] = np.where(c[col] < lo, np.nan, c[col])
c["n_vitals_unusable"] = c[["temp_c_clean", "sbp_clean", "pulse_clean", "spo2_clean"]].isna().sum(axis=1)
y = h[["patient_id", "triage_level", "department", "is_emergency", "should_abstain"]].copy()
y.columns = ["patient_id"] + ["y_" + k for k in y.columns[1:]]
dev = c.merge(y, on="patient_id")
dev.to_csv(f"{D}/processed/dev_dataset.csv", index=False)

sec("8. Clean dataset summary")
line("Written: `data/processed/dev_dataset.csv` (raw columns kept + *_clean + *_status + y_* targets; leaky metadata excluded).")
for col in ["temp_status", "bp_status", "pulse_status", "spo2_status"]:
    line(f"- {col}: " + ", ".join(f"{k}={v}" for k, v in c[col].value_counts().items()))
line(f"- Records with >=1 unusable vital: {(c.n_vitals_unusable > 0).sum()} ({(c.n_vitals_unusable > 0).mean():.1%}); "
     f"clean temp range {c.temp_c_clean.min()}-{c.temp_c_clean.max()}")

open("reports/data_quality_report.md", "w").write("# Phase 1: Data-quality report\n" + "\n".join(R) + "\n")
print("\n".join(R))
