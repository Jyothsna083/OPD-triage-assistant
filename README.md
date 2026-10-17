# PulseGuard AI - Intelligent Outpatient Triage & Emergency Safeguard for Overloaded Government Hospitals

An asymmetric, safety-first clinical triage and queue management system designed for overburdened public hospital Outpatient Departments (OPDs) handling ~800 walk-ins a day with only 2 doctors.

---

## 1. Problem Statement & Real-World Pain

In high-volume public hospitals, patients are traditionally seen on a **First-Come, First-Served (FCFS)** basis. A patient experiencing acute myocardial infarction (chest pain) may wait behind 40 individuals presenting with mild upper respiratory infections. Triage decisions are often made by overworked triage nurses in under 30 seconds with no lab or imaging support.

**Core Objective:**
From a brief intake record and multilingual free-text complaint, accurately assign an Emergency Severity Index (ESI 1–5) triage urgency level and route to the correct clinical specialty department—safely, instantly, and with zero missed emergencies.

---

## 2. Key Benchmark Performance (`evaluate.py`)

Evaluated on the synthetic dataset of **30,000 patients** against `evaluate.py`:

| Metric | Baseline (FCFS / Level 3) | Our System | Clinical Target |
| :--- | :---: | :---: | :---: |
| **Emergency Recall (n=1,213)** | 27.4% | **100.0% (1.0000)** | **&ge; 98.0%** |
| **Missed Emergencies** | 880 | **0** | **0** |
| **Under-Triage Rate** | 42.1% | **0.0000 (0.0%)** | **&le; 2.0%** |
| **Safety Score** | 0.0000 | **0.4592** | Higher is better |
| **Mean Cost per Patient** | 3.112 | **1.683** | Lower is better |
| **Abstention Recall** | 0.0% | **1.000 (100.0%)** | **&ge; 95.0%** |
| **Emergency Wait Time (2 Doctors)** | 62.1 min | **6.4 min (-89.6%)** | Under 15 min |

---

## 3. Core Architectural Highlights

```
Raw Patient Intake (<60s Entry)
          │
          ▼
┌───────────────────────────────────────────────┐
│ Phase 02: Physiological Preprocessing Pipeline │
│  - Detects Fahrenheit in Celsius (e.g. 96°F)  │
│  - Corrects swapped BP (diastolic > systolic) │
│  - Flags unusable vitals (0 BP, 0 Pulse, NaN) │
└──────────────────────┬────────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────────┐
│ Phase 03: Multilingual Clinical NLP Engine     │
│  - English, Hinglish, Hindi, Telugu-English   │
│  - Verified Negation Handling ("denies fever")│
│  - Duration & severity parsing                │
└──────────────────────┬────────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────────┐
│ Phase 05: Cost-Sensitive Bayes Risk ML Model  │
│  - Random Forest Classifier (100 estimators)  │
│  - 20x penalty multiplier on under-triage     │
│  - TF-IDF symptom embeddings                  │
└──────────────────────┬────────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────────┐
│ Phase 04: Mandatory Clinical Safety Escalation │
│  - SpO2 < 85%, SBP < 70, Shock Index >= 1.3   │
│  - Acute Coronary Syndrome (chest pain + arm) │
│  - Code Stroke Triad, Sepsis, Severe Hemorrhage│
│  - Hard overrides always escalate to Level 1/2│
│  - Safe Abstention ("send to nurse now")      │
└──────────────────────┬────────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────────┐
│ Phase 07 & 08: SQLite & Live Nurse Dashboard  │
│  - Live urgency-prioritized patient queue     │
│  - Assessment review & explainability         │
│  - Authorized override with mandatory reason  │
│  - Full immutable audit log                   │
└───────────────────────────────────────────────┘
```

### Why Asymmetric Safety?
In clinical medicine, errors are fundamentally asymmetric:
- **Under-triaging an emergency** (e.g. assigning Level 4 to a patient having an acute myocardial infarction) can lead to preventable death.
- **Over-triaging a mild cold** (e.g. assigning Level 3 instead of Level 5) only costs a few minutes of clinical time.
Our model uses a **Cost-Sensitive Bayes Decision Rule** where expected loss is minimized using the asymmetric cost matrix ($20\times$ penalty for under-triaging true emergencies).

### Safe Abstention Mechanism
When data is missing or corrupted (e.g. 0 BP, zero pulse, or multiple missing vitals), rather than guessing dangerously, the system flags **"NEEDS IMMEDIATE NURSE REVIEW"** (Priority 0). In `evaluate.py`, abstaining on uninterpretable records costs only **0.3**, and all emergencies in abstained records are counted as safely caught.

---

## 4. Project Folder Structure

```
opd-triage-assistant/
│
├── data/
│   ├── intake_records.csv
│   ├── hidden_labels.csv
│   ├── arrival_log.csv
│   ├── processed/
│   │   └── dev_dataset.csv
│   └── opd_triage.db
│
├── models/
│   ├── triage_model.joblib
│   └── department_model.joblib
│
├── modules/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── nlp.py
│   ├── red_flags.py
│   ├── ml_model.py
│   ├── explainability.py
│   ├── database.py
│   ├── evaluation.py
│   └── queue_simulation.py
│
├── templates/
│   ├── base.html
│   ├── intake.html
│   ├── dashboard.html
│   ├── patient_detail.html
│   ├── simulation.html
│   └── audit.html
│
├── static/
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   └── app.js
│   └── img/
│       └── queue_simulation_chart.png
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_nlp.py
│   ├── test_red_flags.py
│   └── test_flask_routes.py
│
├── reports/
│   ├── data_quality_report.md
│   ├── evaluation_report.md
│   └── queue_simulation_chart.png
│
├── scripts/
│   ├── phase1_data_quality.py
│   └── seed_database.py
│
├── app.py
├── train.py
├── evaluate.py
├── predictions.csv
├── requirements.txt
└── README.md
```

---

## 5. Quickstart & Verification Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Data Quality Pipeline (Phase 01)
```bash
python scripts/phase1_data_quality.py
```
Outputs `reports/data_quality_report.md` and `data/processed/dev_dataset.csv`.

### 3. Run All Unit & Integration Tests
```bash
python -m unittest discover -s tests
```
Runs 28 automated tests covering physiological cleaning, negation verification, safety escalations, and Flask routes.

### 4. Train Models (Phase 05)
```bash
python train.py
```
Trains Baseline Logistic Regression and Random Forest classifiers, evaluates cost-sensitive Bayes decision rules, and saves models to `models/`.

### 5. Run Evaluation Benchmark (Phase 06)
```bash
python evaluate.py predictions.csv data/hidden_labels.csv data/arrival_log.csv
```

### 6. Launch the Live Nurse Dashboard (Phase 07 & 08)
```bash
python app.py
```
Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your web browser.

---

## 6. Nurse Dashboard Features

1. **Active Patient Queue (`/`)**: Real-time queue sorted by clinical priority (Priority 0 Abstention &rarr; Level 1 &rarr; Level 2 &rarr; Level 3 &rarr; Level 4 &rarr; Level 5). Department filters and instant search.
2. **Rapid Patient Intake (`/intake`)**: Rapid data entry interface (<60s target time). Includes 6 one-click demo presets (Critical ACS, Acute Stroke, Severe Respiratory Distress, Pediatric Febrile Lethargy, Corrupt Vitals Abstention, Mild URI).
3. **Clinical Assessment & Override (`/patient/<id>`)**: Displays side-by-side comparison of raw ML recommendations vs Hard-coded Safety Engine escalations. Authorized clinical staff can override triage levels and departments with a **mandatory written clinical rationale**.
4. **Queue Simulation & Impact (`/simulation`)**: High-resolution chart demonstrating the 89.6% reduction in emergency wait time under 2 doctors.
5. **Audit Trail (`/audit`)**: Complete chronological logging of all intake assessments and nurse overrides in SQLite.

---

## 7. Known Limitations & Clinical Governance

1. **Synthetic Data Calibration**: Model is trained on synthetic OPD walk-ins. Clinical vitals thresholds (SpO2 < 90, SBP < 80) follow international ESI guidelines, but local clinical governance protocols must approve thresholds prior to bedside deployment.
2. **Laboratory & Imaging Independence**: Triage operates strictly on intake presentation (vitals + complaint); definitive diagnosis occurs in the assigned clinical department.
3. **Nurse In-The-Loop**: The AI serves as clinical decision support. Nurses retain full authority to override any recommendation with logged justification.
