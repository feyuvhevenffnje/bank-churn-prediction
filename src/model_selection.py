import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from preprocessing import load_data, split_data, build_preprocessor

X, y = load_data()
X_train, X_test, y_train, y_test = split_data(X, y)

pipe = Pipeline([
    ("pre", build_preprocessor()),
    ("model", RandomForestClassifier(random_state=42, n_jobs=-1)),
])

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 8, 12],
    "model__min_samples_leaf": [1, 5],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
gs = GridSearchCV(pipe, param_grid, scoring="roc_auc", cv=cv, verbose=1)
gs.fit(X_train, y_train)

print("best params:", gs.best_params_)
print("best CV roc_auc:", round(gs.best_score_, 4))
cols = ["params", "mean_test_score", "std_test_score"]
print(pd.DataFrame(gs.cv_results_)[cols]
        .sort_values("mean_test_score", ascending=False)
        .head(5).to_string(index=False))
