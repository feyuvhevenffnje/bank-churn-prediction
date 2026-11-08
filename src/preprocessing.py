from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "Churn_Modelling.csv"

NUM_COLS = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
            "HasCrCard", "IsActiveMember", "EstimatedSalary"]
CAT_COLS = ["Geography", "Gender"]
FEATURES = NUM_COLS + CAT_COLS
TARGET = "Exited"

def load_data():
    df = pd.read_csv(DATA)
    return df[FEATURES], df[TARGET]

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size,
                            random_state=random_state, stratify=y)

def build_preprocessor():
    num = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", num, NUM_COLS),
        ("cat", cat, CAT_COLS),
    ])

if __name__ == "__main__":
    X, y = load_data()
    X_tr, X_te, y_tr, y_te = split_data(X, y)
    print("train/test:", X_tr.shape, X_te.shape)
    print("churn rate:", round(y_tr.mean(), 4), round(y_te.mean(), 4))
    print("preprocessor:", type(build_preprocessor()).__name__)
