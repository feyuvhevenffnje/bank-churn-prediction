from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve, roc_auc_score, precision_score,
                             recall_score, f1_score, accuracy_score,
                             average_precision_score)
from preprocessing import load_data, split_data, build_preprocessor, FEATURES

ROOT = Path(__file__).resolve().parents[1]
(ROOT / "models").mkdir(exist_ok=True)
(ROOT / "reports").mkdir(exist_ok=True)

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

X, y = load_data()
X_train, X_test, y_train, y_test = split_data(X, y)

pipe = Pipeline([
    ("pre", build_preprocessor()),
    ("model", RandomForestClassifier(
        n_estimators=200, max_depth=None, min_samples_leaf=5,
        random_state=42, n_jobs=-1)),
])
pipe.fit(X_train, y_train)

prob = pipe.predict_proba(X_test)[:, 1]
pred = (prob >= 0.5).astype(int)

print("=== 阈值 0.5 的五项指标 ===")
for name, val in [
    ("Accuracy",  accuracy_score(y_test, pred)),
    ("Precision", precision_score(y_test, pred)),
    ("Recall",    recall_score(y_test, pred)),
    ("F1",        f1_score(y_test, pred)),
    ("ROC-AUC",   roc_auc_score(y_test, prob)),
    ("PR-AUC",    average_precision_score(y_test, prob)),
]:
    print(f"{name:10s} {val:.4f}")

cm = confusion_matrix(y_test, pred)
print("\n混淆矩阵 [[TN, FP], [FN, TP]]:\n", cm)

fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay(cm, display_labels=["未流失", "流失"]).plot(cmap="Blues", ax=ax)
ax.set_title("混淆矩阵（阈值 0.5）")
plt.tight_layout()
plt.savefig(ROOT / "reports" / "confusion_matrix.png", dpi=150)
plt.close()

fpr, tpr, _ = roc_curve(y_test, prob)
auc = roc_auc_score(y_test, prob)
plt.figure(figsize=(5, 4))
plt.plot(fpr, tpr, label=f"RandomForest (AUC={auc:.4f})")
plt.plot([0, 1], [0, 1], "--", color="gray", label="随机猜测")
plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC 曲线")
plt.legend(); plt.tight_layout()
plt.savefig(ROOT / "reports" / "roc_curve.png", dpi=150)
plt.close()

rows = []
for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
    p = (prob >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, p).ravel()
    rows.append({
        "threshold": t,
        "precision": round(precision_score(y_test, p, zero_division=0), 4),
        "recall":    round(recall_score(y_test, p), 4),
        "f1":        round(f1_score(y_test, p), 4),
        "flagged":   int(p.sum()),
        "TP": int(tp), "FP": int(fp), "FN": int(fn),
    })
th = pd.DataFrame(rows)
print("\n=== 阈值实验 ===\n", th.to_string(index=False))
th.to_csv(ROOT / "reports" / "threshold_table.csv", index=False)

y_arr = y_test.to_numpy()
top100 = np.argsort(-prob)[:100]
print(f"\n按概率取前 100 人: 命中 {y_arr[top100].sum()} 人, 命中率 {y_arr[top100].mean():.4f}")

joblib.dump(pipe, ROOT / "models" / "churn_model.joblib")
meta = {
    "features": FEATURES,
    "target": "Exited",
    "model": "RandomForestClassifier",
    "best_params": {"n_estimators": 200, "max_depth": None,
                    "min_samples_leaf": 5, "random_state": 42},
    "default_threshold": 0.5,
    "test_roc_auc": round(auc, 4),
}
(ROOT / "models" / "model_meta.json").write_text(
    json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
print("\n模型与元数据已保存到 models/")
