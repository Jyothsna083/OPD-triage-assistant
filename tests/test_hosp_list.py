import sqlite3

conn = sqlite3.connect("data/opd_triage.db")
cur = conn.cursor()
cur.execute("PRAGMA table_info(hospitals)")
cols = [c[1] for c in cur.fetchall()]
if "district" not in cols:
    cur.execute("ALTER TABLE hospitals ADD COLUMN district TEXT")
    conn.commit()
conn.close()

import sys
sys.path.insert(0, ".")
from modules import database as db
db.init_db(reset=False)
h = db.get_all_hospitals()
print(f"Total hospitals loaded: {len(h)}")
for item in h[:10]:
    print(f"  {item['hospital_id']}: {item['name']} ({item['city']}, {item['district']}) - {item['facility_type']}")
