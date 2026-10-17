"""
modules/database.py
Multi-Hospital SQLite Database for PulseGuard AI.
Populates all 37 AP Government Hospitals (H01 - H37) across Teaching/General,
District, and Area Hospitals.
"""

import sqlite3
import json
import os
import csv
from datetime import datetime
from typing import Dict, List, Any, Optional

DB_PATH = os.path.join("data", "opd_triage.db")
HOSPITALS_CSV_PATH = os.path.join("data", "hospitals.csv")


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DB_PATH, reset: bool = False):
    """Initializes SQLite database schema with multi-hospital support."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    if reset and os.path.exists(db_path):
        os.remove(db_path)

    conn = get_connection(db_path)
    cur = conn.cursor()

    # Hospitals table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS hospitals (
            hospital_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            code TEXT NOT NULL,
            city TEXT,
            district TEXT,
            state TEXT DEFAULT 'Andhra Pradesh',
            facility_type TEXT,
            doctors_on_duty INTEGER DEFAULT 2,
            daily_walkins INTEGER DEFAULT 800,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Staff Users table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS staff_users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            full_name TEXT NOT NULL,
            role TEXT NOT NULL,
            staff_code TEXT,
            hospital_id TEXT,
            department TEXT DEFAULT 'Triage & Emergency',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (hospital_id) REFERENCES hospitals(hospital_id)
        )
    """)

    # Patients table (with hospital_id)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            patient_id TEXT PRIMARY KEY,
            hospital_id TEXT DEFAULT 'H07',
            name TEXT,
            age REAL,
            sex TEXT,
            bp_systolic REAL,
            bp_diastolic REAL,
            pulse REAL,
            temp_c REAL,
            spo2 REAL,
            chief_complaint TEXT,
            known_conditions TEXT,
            medication_list TEXT,
            language_style TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (hospital_id) REFERENCES hospitals(hospital_id)
        )
    """)

    # Triage assessments table (with hospital_id)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS triage_assessments (
            assessment_id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT,
            hospital_id TEXT DEFAULT 'H07',
            ai_triage_level INTEGER,
            ai_department TEXT,
            ai_abstain INTEGER,
            final_triage_level INTEGER,
            final_department TEXT,
            detected_red_flags TEXT,
            clinical_rationale TEXT,
            level_probabilities TEXT,
            status TEXT DEFAULT 'WAITING',
            nurse_reviewed INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (patient_id) REFERENCES patients(patient_id),
            FOREIGN KEY (hospital_id) REFERENCES hospitals(hospital_id)
        )
    """)

    # Nurse overrides audit table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS nurse_overrides (
            override_id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT,
            assessment_id INTEGER,
            hospital_id TEXT DEFAULT 'H07',
            nurse_name TEXT,
            previous_level INTEGER,
            new_level INTEGER,
            previous_dept TEXT,
            new_dept TEXT,
            reason TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (patient_id) REFERENCES patients(patient_id),
            FOREIGN KEY (assessment_id) REFERENCES triage_assessments(assessment_id)
        )
    """)

    # Audit events log
    cur.execute("""
        CREATE TABLE IF NOT EXISTS audit_logs (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            action TEXT,
            patient_id TEXT,
            hospital_id TEXT DEFAULT 'H07',
            details TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()

    # Load all 37 hospitals from CSV
    populate_all_hospitals(conn)

    conn.close()


def populate_all_hospitals(conn: sqlite3.Connection):
    """Loads all 37 hospitals from data/hospitals.csv."""
    cur = conn.cursor()

    if os.path.exists(HOSPITALS_CSV_PATH):
        with open(HOSPITALS_CSV_PATH, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                hid = row["hospital_id"].strip()
                name = row["name"].strip()
                town = row["town"].strip()
                dist = row["district"].strip()
                htype = row["type"].strip()

                docs = 3 if "Teaching" in htype else (2 if "District" in htype else 2)
                walkins = 950 if "Teaching" in htype else (650 if "District" in htype else 450)

                cur.execute("""
                    INSERT OR REPLACE INTO hospitals (
                        hospital_id, name, code, city, district, state, facility_type, doctors_on_duty, daily_walkins
                    ) VALUES (?, ?, ?, ?, ?, 'Andhra Pradesh', ?, ?, ?)
                """, (hid, name, hid, town, dist, htype, docs, walkins))

    # Seed Demo staff accounts across different hospitals
    demo_staff = [
        ("nurse.priya", "nurse123", "Nurse Priya Sharma, RN", "Triage Nurse", "RN-H07-104", "H07", "OPD Triage Desk"),
        ("dr.verma", "doctor123", "Dr. Rajesh Verma, MD", "Emergency Medical Officer", "EMO-H07-042", "H07", "Emergency"),
        ("nurse.srikakulam", "nurse123", "Nurse Sunita Rao, RN", "Triage Nurse", "RN-H01-018", "H01", "OPD Triage Desk"),
        ("dr.vizag", "doctor123", "Dr. Ravi Kumar, MD", "Chief Medical Officer", "CMO-H03-005", "H03", "Emergency"),
        ("nurse.guntur", "nurse123", "Nurse Anjali Devi, RN", "Staff Nurse", "RN-H09-082", "H09", "OPD Triage Desk"),
        ("dr.tirupati", "doctor123", "Dr. Suresh Reddy, MS", "Emergency Physician", "EP-H13-011", "H13", "Trauma & Emergency"),
        ("nurse.nandyal", "nurse123", "Nurse Swapna, B.Sc.", "District Triage Lead", "RN-H16-033", "H16", "OPD Triage Desk"),
        ("admin.ap", "admin123", "Dr. K. S. Rao", "AP Health Network Administrator", "ADM-AP-001", "H07", "Directorate of Medical Education"),
    ]

    cur.executemany("""
        INSERT OR REPLACE INTO staff_users (
            username, password, full_name, role, staff_code, hospital_id, department
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """, demo_staff)

    conn.commit()


def get_all_hospitals(db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    """Returns list of all 37 hospitals sorted by ID."""
    conn = get_connection(db_path)
    cur = conn.cursor()
    cur.execute("SELECT * FROM hospitals ORDER BY hospital_id ASC")
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_hospital_by_id(hospital_id: str, db_path: str = DB_PATH) -> Optional[Dict[str, Any]]:
    """Fetches details for a specific hospital facility."""
    conn = get_connection(db_path)
    cur = conn.cursor()
    cur.execute("SELECT * FROM hospitals WHERE hospital_id = ?", (hospital_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def authenticate_staff(username: str, password: str, hospital_id: Optional[str] = None, db_path: str = DB_PATH) -> Optional[Dict[str, Any]]:
    """Validates staff login credentials against database."""
    conn = get_connection(db_path)
    cur = conn.cursor()

    cur.execute("""
        SELECT u.*, h.name as hospital_name, h.code as hospital_code, h.city as hospital_city, h.district as hospital_district
        FROM staff_users u
        LEFT JOIN hospitals h ON u.hospital_id = h.hospital_id
        WHERE u.username = ? AND u.password = ?
    """, (username, password))

    user = cur.fetchone()
    conn.close()
    if user:
        u_dict = dict(user)
        # If staff logs into a specific hospital selection, update active session hospital
        if hospital_id and hospital_id != u_dict["hospital_id"]:
            selected_hosp = get_hospital_by_id(hospital_id)
            if selected_hosp:
                u_dict["hospital_id"] = hospital_id
                u_dict["hospital_name"] = selected_hosp["name"]
                u_dict["hospital_code"] = selected_hosp["code"]
                u_dict["hospital_city"] = selected_hosp["city"]
        return u_dict
    return None


def save_patient_and_assessment(
    patient_data: Dict[str, Any],
    ai_result: Dict[str, Any],
    hospital_id: str = "H07",
    db_path: str = DB_PATH
) -> int:
    """Saves new patient record and corresponding triage assessment with hospital tagging."""
    conn = get_connection(db_path)
    cur = conn.cursor()

    patient_id = patient_data.get("patient_id", f"P_{int(datetime.now().timestamp()*1000)}")

    cur.execute("""
        INSERT OR REPLACE INTO patients (
            patient_id, hospital_id, name, age, sex, bp_systolic, bp_diastolic,
            pulse, temp_c, spo2, chief_complaint, known_conditions,
            medication_list, language_style
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        patient_id,
        hospital_id,
        patient_data.get("name", f"Patient {patient_id}"),
        patient_data.get("age"),
        patient_data.get("sex"),
        patient_data.get("bp_systolic"),
        patient_data.get("bp_diastolic"),
        patient_data.get("pulse"),
        patient_data.get("temp_c"),
        patient_data.get("spo2"),
        patient_data.get("chief_complaint"),
        patient_data.get("known_conditions"),
        patient_data.get("medication_list"),
        patient_data.get("language_style", "english"),
    ))

    red_flags_json = json.dumps(ai_result.get("detected_red_flags", []))
    rationale_json = json.dumps(ai_result.get("explanations", []))
    probas_json = json.dumps(ai_result.get("level_probabilities", {}))

    cur.execute("""
        INSERT INTO triage_assessments (
            patient_id, hospital_id, ai_triage_level, ai_department, ai_abstain,
            final_triage_level, final_department, detected_red_flags,
            clinical_rationale, level_probabilities, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'WAITING')
    """, (
        patient_id,
        hospital_id,
        ai_result.get("ai_triage_level"),
        ai_result.get("ai_department"),
        ai_result.get("ai_abstain", 0),
        ai_result.get("final_triage_level", ai_result.get("ai_triage_level")),
        ai_result.get("final_department", ai_result.get("ai_department")),
        red_flags_json,
        rationale_json,
        probas_json,
    ))

    assessment_id = cur.lastrowid

    cur.execute("""
        INSERT INTO audit_logs (action, patient_id, hospital_id, details)
        VALUES (?, ?, ?, ?)
    """, ("NEW_INTAKE_ASSESSMENT", patient_id, hospital_id, f"Initial Triage Level {ai_result.get('final_triage_level')} -> {ai_result.get('final_department')}"))

    conn.commit()
    conn.close()
    return assessment_id


def record_nurse_override(
    assessment_id: int,
    patient_id: str,
    nurse_name: str,
    new_level: int,
    new_dept: str,
    reason: str,
    hospital_id: str = "H07",
    db_path: str = DB_PATH
) -> bool:
    """
    Records nurse override of AI decision.
    Preserves original AI assessment while updating final status and logging reason.
    """
    conn = get_connection(db_path)
    cur = conn.cursor()

    cur.execute("SELECT * FROM triage_assessments WHERE assessment_id = ?", (assessment_id,))
    row = cur.fetchone()
    if not row:
        conn.close()
        return False

    prev_level = row["final_triage_level"]
    prev_dept = row["final_department"]

    cur.execute("""
        INSERT INTO nurse_overrides (
            patient_id, assessment_id, hospital_id, nurse_name,
            previous_level, new_level, previous_dept, new_dept, reason
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        patient_id, assessment_id, hospital_id, nurse_name,
        prev_level, new_level, prev_dept, new_dept, reason
    ))

    cur.execute("""
        UPDATE triage_assessments
        SET final_triage_level = ?, final_department = ?, nurse_reviewed = 1
        WHERE assessment_id = ?
    """, (new_level, new_dept, assessment_id))

    cur.execute("""
        INSERT INTO audit_logs (action, patient_id, hospital_id, details)
        VALUES (?, ?, ?, ?)
    """, (
        "NURSE_OVERRIDE",
        patient_id,
        hospital_id,
        f"Staff {nurse_name} changed Level {prev_level} ({prev_dept}) -> Level {new_level} ({new_dept}). Reason: {reason}"
    ))

    conn.commit()
    conn.close()
    return True


def get_patient_queue(
    db_path: str = DB_PATH,
    filter_dept: Optional[str] = None,
    filter_hospital_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Returns active waiting queue sorted by clinical urgency:
    Priority 0: Abstain / Immediate Nurse Review Needed
    Priority 1: Level 1 (Resuscitation)
    Priority 2: Level 2 (Emergent)
    Priority 3: Level 3 (Urgent)
    Priority 4: Level 4 (Less Urgent)
    Priority 5: Level 5 (Non-Urgent)
    Tie breaker: arrival time (created_at ascending).
    """
    conn = get_connection(db_path)
    cur = conn.cursor()

    query = """
        SELECT
            p.patient_id, p.name, p.age, p.sex, p.chief_complaint,
            p.bp_systolic, p.bp_diastolic, p.pulse, p.temp_c, p.spo2,
            a.assessment_id, a.hospital_id, a.ai_triage_level, a.ai_department, a.ai_abstain,
            a.final_triage_level, a.final_department, a.detected_red_flags,
            a.clinical_rationale, a.level_probabilities, a.status,
            a.nurse_reviewed, a.created_at,
            h.name as hospital_name, h.code as hospital_code,
            CASE
                WHEN a.ai_abstain = 1 AND a.nurse_reviewed = 0 THEN 0
                ELSE a.final_triage_level
            END as priority_order
        FROM triage_assessments a
        JOIN patients p ON a.patient_id = p.patient_id
        LEFT JOIN hospitals h ON a.hospital_id = h.hospital_id
        WHERE a.status = 'WAITING'
    """
    params = []

    if filter_hospital_id and filter_hospital_id != "ALL":
        query += " AND a.hospital_id = ?"
        params.append(filter_hospital_id)

    if filter_dept and filter_dept != "All":
        query += " AND a.final_department = ?"
        params.append(filter_dept)

    query += " ORDER BY priority_order ASC, a.created_at ASC"

    cur.execute(query, params)
    rows = cur.fetchall()
    result = []
    for r in rows:
        d = dict(r)
        d["detected_red_flags"] = json.loads(d["detected_red_flags"]) if d["detected_red_flags"] else []
        d["clinical_rationale"] = json.loads(d["clinical_rationale"]) if d["clinical_rationale"] else []
        d["level_probabilities"] = json.loads(d["level_probabilities"]) if d["level_probabilities"] else {}
        result.append(d)

    conn.close()
    return result


def get_patient_detail(patient_id: str, db_path: str = DB_PATH) -> Optional[Dict[str, Any]]:
    """Retrieves full clinical intake, AI assessment, hospital, and override history for a patient."""
    conn = get_connection(db_path)
    cur = conn.cursor()

    cur.execute("""
        SELECT p.*, h.name as hospital_name, h.code as hospital_code, h.city as hospital_city, h.district as hospital_district
        FROM patients p
        LEFT JOIN hospitals h ON p.hospital_id = h.hospital_id
        WHERE p.patient_id = ?
    """, (patient_id,))
    patient = cur.fetchone()
    if not patient:
        conn.close()
        return None

    cur.execute("""
        SELECT * FROM triage_assessments
        WHERE patient_id = ?
        ORDER BY created_at DESC LIMIT 1
    """, (patient_id,))
    assessment = cur.fetchone()

    cur.execute("""
        SELECT * FROM nurse_overrides
        WHERE patient_id = ?
        ORDER BY created_at DESC
    """, (patient_id,))
    overrides = cur.fetchall()

    res = dict(patient)
    if assessment:
        res["assessment"] = dict(assessment)
        res["assessment"]["detected_red_flags"] = json.loads(assessment["detected_red_flags"]) if assessment["detected_red_flags"] else []
        res["assessment"]["clinical_rationale"] = json.loads(assessment["clinical_rationale"]) if assessment["clinical_rationale"] else []
        res["assessment"]["level_probabilities"] = json.loads(assessment["level_probabilities"]) if assessment["level_probabilities"] else {}
    else:
        res["assessment"] = None

    res["overrides"] = [dict(o) for o in overrides]
    conn.close()
    return res


def get_audit_trail(limit: int = 50, hospital_id: Optional[str] = None, db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    """Retrieves recent audit events."""
    conn = get_connection(db_path)
    cur = conn.cursor()
    if hospital_id and hospital_id != "ALL":
        cur.execute("""
            SELECT a.*, h.name as hospital_name
            FROM audit_logs a
            LEFT JOIN hospitals h ON a.hospital_id = h.hospital_id
            WHERE a.hospital_id = ?
            ORDER BY a.created_at DESC LIMIT ?
        """, (hospital_id, limit))
    else:
        cur.execute("""
            SELECT a.*, h.name as hospital_name
            FROM audit_logs a
            LEFT JOIN hospitals h ON a.hospital_id = h.hospital_id
            ORDER BY a.created_at DESC LIMIT ?
        """, (limit,))
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]
