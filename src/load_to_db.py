import sqlite3
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
csv_path = ROOT / "data" / "raw" / "Churn_Modelling.csv"
db_path = ROOT / "database" / "bank.db"
db_path.parent.mkdir(exist_ok=True)

df = pd.read_csv(csv_path)
with sqlite3.connect(db_path) as conn:
    df.to_sql("customers", conn, if_exists="replace", index=False)

print(f"imported {len(df)} rows -> {db_path}")
