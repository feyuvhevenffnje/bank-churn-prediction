from pathlib import Path
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
from preprocessing import load_data, split_data, build_preprocessor

ROOT = Path(__file__).resolve().parents[1]
X, y = load_data()
X_train, X_test, y_train, y_test = split_data(X, y)

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
rows = []

for name, model in models.items():
    pipe = Pipeline([("pre", build_preprocessor()), ("model", model)])

    res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring)
    row = {"model": name}
    for s in scoring:
        row[s + "_cv"] = round(res["test_" + s].mean(), 4)

    pipe.fit(X_train, y_train)
    tr_pred = (pipe.predict_proba(X_train)[:, 1] >= 0.5).astype(int)
    prob = pipe.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    row["train_acc"] = round(accuracy_score(y_train, tr_pred), 4)
    row["test_acc"] = round(accuracy_score(y_test, pred), 4)
    row["test_precision"] = round(precision_score(y_test, pred), 4)
    row["test_recall"] = round(recall_score(y_test, pred), 4)
    row["test_f1"] = round(f1_score(y_test, pred), 4)
    row["test_roc_auc"] = round(roc_auc_score(y_test, prob), 4)
    row["overfit_gap"] = round(row["train_acc"] - row["test_acc"], 4)
    rows.append(row)

report = pd.DataFrame(rows)
print(report.to_string(index=False))
report.to_csv(ROOT / "reports" / "model_comparison.csv", index=False)
