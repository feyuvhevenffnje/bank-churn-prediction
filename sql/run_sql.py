import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
conn = sqlite3.connect(ROOT / "database" / "bank.db")
sql = (ROOT / "sql" / "analysis.sql").read_text(encoding="utf-8")

for i, stmt in enumerate([s.strip() for s in sql.split(";") if s.strip()], 1):
    cur = conn.execute(stmt)
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    print(f"\n--- Q{i} --- {cols}")
    for r in rows[:10]:
        print(" ", r)
    if len(rows) > 10:
        print(f"  ... 共 {len(rows)} 行")
conn.close()
