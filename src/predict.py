from pathlib import Path
import json
import joblib
import pandas as pd
from preprocessing import load_data, split_data, FEATURES, CAT_COLS

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "churn_model.joblib"
META_PATH = ROOT / "models" / "model_meta.json"

def load_model():
    pipe = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    return pipe, meta

def build_defaults():
    X, y = load_data()
    X_train, _, _, _ = split_data(X, y)
    defaults = {}
    for col in FEATURES:
        if col in CAT_COLS:
            defaults[col] = X_train[col].mode()[0]
        elif col in ("HasCrCard", "IsActiveMember", "NumOfProducts"):
            defaults[col] = int(X_train[col].mode()[0])
        else:
            defaults[col] = float(X_train[col].median())
    return defaults

DEFAULTS = build_defaults()

def predict_one(customer: dict, threshold: float = 0.5):
    pipe, meta = load_model()
    cols = meta["features"]
    unknown = set(customer) - set(cols)
    if unknown:
        raise ValueError(f"未知字段: {sorted(unknown)}")
    row = {c: customer.get(c, DEFAULTS[c]) for c in cols}
    X_new = pd.DataFrame([row], columns=cols)
    prob = float(pipe.predict_proba(X_new)[0, 1])
    return {
        "prediction": int(prob >= threshold),
        "churn_probability": round(prob, 4),
        "used_fields": row,
    }
