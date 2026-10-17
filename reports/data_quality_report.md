# Phase 1: Data-quality report

## 1. Structure, types, missing values, duplicates

- **intake_records**: 30000 rows x 12 cols; duplicate rows=0; duplicate patient_id=0
- **hidden_labels**: 30000 rows x 11 cols; duplicate rows=0; duplicate patient_id=0
- **arrival_log**: 30000 rows x 5 cols; duplicate rows=0; duplicate patient_id=0
- Missing in intake: bp_systolic=1138 (3.8%), bp_diastolic=1138 (3.8%), pulse=1155 (3.9%), temp_c=1196 (4.0%), spo2=873 (2.9%)
- Distinct complaint strings: 6567 of 30000 (templated text; repeats are not duplicate patients)

## 2. ID consistency

- ID sets identical across all 3 files: **True**; same row order: True

## 3. Vital-sign validity (raw)

| vital | missing | exactly 0 | outside plausible range | min(>0) | max |
|---|---|---|---|---|---|
| bp_systolic | 1138 | 447 | 1 | 49.0 | 226.0 |
| bp_diastolic | 1138 | 447 | 5 | 33.0 | 185.0 |
| pulse | 1155 | 132 | 0 | 38.0 | 180.0 |
| spo2 | 873 | 234 | 0 | 75.0 | 100.0 |
| temp_c | 1196 | 0 | 677 | 35.8 | 102.9 |

- temp_c >= 90 (Fahrenheit typed into a Celsius field): **677**; values between 43 and 90: 0, so the split is clean.
- diastolic > systolic (swapped): 147; equal and nonzero: 0
- Age 0-90; 123 records aged 0; 5979 under 18. Pulse/BP thresholds must be age-adjusted.

## 4. Labels

Triage level counts: L1=534, L2=2715, L3=7268, L4=9819, L5=9664

Emergencies: 1213 (4.04%); should_abstain: 553 (1.84%)

Triage level x is_emergency:

|   triage_level |    0 |   1 |
|---------------:|-----:|----:|
|              1 |    0 | 534 |
|              2 | 2036 | 679 |
|              3 | 7268 |   0 |
|              4 | 9819 |   0 |
|              5 | 9664 |   0 |

Departments: General Medicine=11310, Orthopedics=3772, Pediatrics=2792, Surgery=2788, Gastroenterology=2320, Dermatology=1726, Pulmonology=1557, Emergency=1213, ENT=738, Ophthalmology=730, Dental=600, Obstetrics & Gynecology=454

Department==Emergency x is_emergency:

| department   |     0 |    1 |
|:-------------|------:|-----:|
| False        | 28787 |    0 |
| True         |     0 | 1213 |

- All Level-1 are emergencies: True; no Level 3-5 is an emergency: True.
- Level 2 splits into emergency=679 and non-emergency=2036.
- Flag rates: under_reported=12.8%, over_reported=4.5%, flags_visible_in_text=87.2%
- Emergency archetypes: acs_chest_pain=322, severe_resp_distress=157, stroke=151, sepsis_shock=135, major_trauma=120, gi_bleed=84, seizure=80, anaphylaxis=65, eclampsia=57, poisoning=42

## 5. Label / metadata consistency

| check | agreement |
|---|---|
| fahrenheit/temp_96 vs temp_c>=90 | 0.9998 |
| missing_temp vs NaN temp | 1.0000 |
| missing_bp vs NaN bp | 1.0000 |
| missing_pulse vs NaN pulse | 1.0000 |
| missing_spo2 vs NaN spo2 | 1.0000 |
| zero_bp vs bp==0 | 0.9999 |
| zero_pulse vs pulse==0 | 0.9999 |
| zero_spo2 vs spo2==0 | 0.9999 |
| bp_swapped vs dbp>sbp | 1.0000 |

- should_abstain=1 always has a vitals issue: True (553 abstain cases of 5115 vitals-issue cases)
- Abstain cases: 2.9% are emergencies; by level: L1=5, L2=41, L3=138, L4=187, L5=182
- Pediatrics share among age<=12: 62.3%
- Language styles: hinglish=9026, english=8979, telugu_english=6646, hindi_translit=5349

## 6. Arrival log

- 38 days, median 800 arrivals/day (min 400, max 800); consult_minutes mean 1.43, max 14.4
- Sorted by time within each day: True
- Mean consult_minutes by true level: L1=3.91, L2=2.98, L3=1.80, L4=1.11, L5=0.92

## 7. Leakage warning

In `hidden_labels.csv`, only `triage_level`, `department`, `is_emergency` and `should_abstain` may be used as training/evaluation **targets**. `red_flags`, `archetype`, `vitals_issues`, `under_reported`, `over_reported` and `flags_visible_in_text` are answer keys and must never be inference-time inputs. Derive vitals problems from the raw values.

## 8. Clean dataset summary

Written: `data/processed/dev_dataset.csv` (raw columns kept + *_clean + *_status + y_* targets; leaky metadata excluded).
- temp_status: ok=28127, missing=1196, converted_from_F=677
- bp_status: ok=28268, missing=1138, invalid_zero=447, swapped_fixed=147
- pulse_status: ok=28713, missing=1155, invalid_zero=132
- spo2_status: ok=28893, missing=873, invalid_zero=234
- Records with >=1 unusable vital: 4387 (14.6%); clean temp range 35.6-41.2
