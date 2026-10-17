import sqlite3
conn = sqlite3.connect("data/opd_triage.db")
cur = conn.cursor()
cur.execute("DELETE FROM hospitals WHERE hospital_id LIKE 'HOSP_%'")
cur.execute("UPDATE staff_users SET hospital_id = 'H07' WHERE hospital_id = 'HOSP_01'")
conn.commit()
cur.execute("SELECT COUNT(*) FROM hospitals")
print("Active AP Hospitals count:", cur.fetchone()[0])
conn.close()
