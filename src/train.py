"""Train, tune, compare and save the academic performance classifier.

Run from the project root:  python -m src.train

Method
------
* 80/20 stratified split (fixed seed). The test set is NOT used for tuning or selection.
* Each model is a Pipeline(preprocessor -> classifier) tuned with GridSearchCV
  (5-fold stratified CV on the training set, scoring = macro-F1).
* Selection score (fixed before looking at results):
      0.5 * CV macro-F1  +  0.5 * CV recall of the Low (L) class
  computed from cross-validated predictions on the training set, so missing
  at-risk students is penalised explicitly rather than selecting on accuracy.
* The winner is evaluated once on the held-out test set and saved with Joblib.
"""
from __future__ import annotations

import json
import platform
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .evaluate import (FIG_DIR, REPORT_DIR, compute_metrics, permutation_feature_importance,
                       plot_confusion, plot_importance)
from .preprocessing import (CATEGORICAL_FEATURES, CLASS_NAMES, CLASS_ORDER, FEATURES,
                            NUMERIC_FEATURES, NUMERIC_RANGE, PROJECT_ROOT, RANDOM_STATE, TARGET,
                            build_preprocessor, load_data, train_test_split_stratified)

MODEL_PATH = PROJECT_ROOT / "models" / "academic_performance_model.pkl"
META_PATH = PROJECT_ROOT / "models" / "model_metadata.json"
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def candidate_models() -> dict:
    """name -> (pipeline, param_grid). XGBoost is included only if installed."""
    cands = {
        "Logistic Regression": (
            Pipeline([("prep", build_preprocessor(scale_numeric=True)),
                      ("clf", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE))]),
            {"clf__C": [0.1, 1, 10], "clf__class_weight": [None, "balanced"]}),
        "Decision Tree": (
            Pipeline([("prep", build_preprocessor(False)),
                      ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE))]),
            {"clf__max_depth": [3, 5, 7, None], "clf__min_samples_leaf": [1, 3, 5],
             "clf__class_weight": [None, "balanced"]}),
        "Random Forest": (
            Pipeline([("prep", build_preprocessor(False)),
                      ("clf", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1))]),
            {"clf__n_estimators": [200, 400], "clf__max_depth": [None, 8],
             "clf__min_samples_leaf": [1, 2], "clf__class_weight": [None, "balanced"]}),
        "Gradient Boosting": (
            Pipeline([("prep", build_preprocessor(False)),
                      ("clf", GradientBoostingClassifier(random_state=RANDOM_STATE))]),
            {"clf__n_estimators": [100, 200], "clf__learning_rate": [0.05, 0.1],
             "clf__max_depth": [2, 3]}),
    }
    try:
        from xgboost import XGBClassifier
        from sklearn.preprocessing import LabelEncoder  # noqa: F401  (XGB needs integer labels)
        cands["XGBoost"] = (
            Pipeline([("prep", build_preprocessor(False)),
                      ("clf", XGBClassifier(random_state=RANDOM_STATE, eval_metric="mlogloss",
                                            n_jobs=1))]),
            {"clf__n_estimators": [100, 200], "clf__learning_rate": [0.05, 0.1],
             "clf__max_depth": [2, 3]})
    except ImportError:
        print("[info] xgboost is not installed - XGBoost skipped.")
    return cands


class _EncodedLabels:
    """Not used: XGBoost integer-label handling lives in _fit_one when needed."""


def _fit_one(name, pipe, grid, X_tr, y_tr):
    y_fit = y_tr
    if name == "XGBoost":  # XGBoost needs integer labels
        y_fit = y_tr.map({c: i for i, c in enumerate(CLASS_ORDER)})
    gs = GridSearchCV(pipe, grid, scoring="f1_macro", cv=CV, n_jobs=-1, refit=True)
    t0 = time.time()
    gs.fit(X_tr, y_fit)
    best = gs.best_estimator_
    cv_pred = cross_val_predict(best, X_tr, y_fit, cv=CV)
    if name == "XGBoost":
        cv_pred = np.array(CLASS_ORDER)[cv_pred]
    cv_macro_f1 = f1_score(y_tr, cv_pred, average="macro", labels=CLASS_ORDER)
    cv_low_recall = recall_score(y_tr, cv_pred, labels=["L"], average=None)[0]
    return {
        "estimator": best, "best_params": {k: (v if v is not None else "None") for k, v in gs.best_params_.items()},
        "grid_search_cv_macro_f1": float(gs.best_score_),
        "cv_macro_f1": float(cv_macro_f1), "cv_low_recall": float(cv_low_recall),
        "selection_score": float(0.5 * cv_macro_f1 + 0.5 * cv_low_recall),
        "fit_seconds": round(time.time() - t0, 1),
    }


class _LabelDecoder:
    """Wraps a fitted XGBoost pipeline so it predicts string labels (L/M/H)."""

    def __init__(self, pipe):
        self.pipe = pipe
        self.classes_ = np.array(CLASS_ORDER)

    def predict(self, X):
        return self.classes_[self.pipe.predict(X)]

    def predict_proba(self, X):
        return self.pipe.predict_proba(X)


def main() -> None:
    df = load_data()
    X_tr, X_te, y_tr, y_te = train_test_split_stratified(df)
    print(f"Rows after de-duplication: {len(df)} | train {len(X_tr)} | test {len(X_te)}")

    results, fitted = {}, {}
    for name, (pipe, grid) in candidate_models().items():
        print(f"Tuning {name} ...")
        r = _fit_one(name, pipe, grid, X_tr, y_tr)
        est = r.pop("estimator")
        fitted[name] = est
        eval_est = _LabelDecoder(est) if name == "XGBoost" else est
        classes = eval_est.classes_
        r["test"] = compute_metrics(y_te, eval_est.predict(X_te), eval_est.predict_proba(X_te), classes)
        results[name] = r
        plot_confusion(r["test"]["confusion_matrix"], f"{name} (test)",
                       FIG_DIR / f"confusion_{name.lower().replace(' ', '_')}.png")
        print(f"  CV macro-F1 {r['cv_macro_f1']:.3f} | CV Low recall {r['cv_low_recall']:.3f} | "
              f"score {r['selection_score']:.3f} | test acc {r['test']['accuracy']:.3f}")

    best_name = max(results, key=lambda n: results[n]["selection_score"])
    best = fitted[best_name]
    print(f"\nSelected: {best_name} (selection score {results[best_name]['selection_score']:.3f})")
    if best_name == "XGBoost":
        raise SystemExit("XGBoost selected: serialisation wrapper not wired for serving; handle explicitly.")

    imp = permutation_feature_importance(best, X_te, y_te)
    plot_importance(imp, f"Permutation importance - {best_name}", FIG_DIR / "feature_importance_best.png")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    imp.to_csv(REPORT_DIR / "feature_importance_best.csv", index=False)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, MODEL_PATH)

    summary = {n: {k: v for k, v in r.items()} for n, r in results.items()}
    (REPORT_DIR / "evaluation_results.json").write_text(json.dumps(
        {"selected_model": best_name, "selection_rule": "0.5*CV macro-F1 + 0.5*CV Low-class recall (train set only)",
         "models": summary}, indent=2))
    meta = {
        "model_name": best_name,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": "Students' Academic Performance Dataset (xAPI-Edu-Data), Kaggle aljarah/xAPI-Edu-Data",
        "rows_used": int(len(df)), "train_rows": int(len(X_tr)), "test_rows": int(len(X_te)),
        "target": TARGET, "class_labels": CLASS_ORDER, "class_names": CLASS_NAMES,
        "probability_class_order": list(best.classes_),
        "feature_names": FEATURES, "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "categorical_values_seen": {c: sorted(df[c].unique().tolist()) for c in CATEGORICAL_FEATURES},
        "numeric_valid_range": list(NUMERIC_RANGE),
        "preprocessing": "ColumnTransformer: OneHotEncoder(handle_unknown=ignore) on categoricals; "
                         "numerics passed through unscaled (tree model) - fitted on training data only",
        "best_params": results[best_name]["best_params"],
        "selection_rule": "0.5*CV macro-F1 + 0.5*CV Low-class recall (train set only)",
        "cv_macro_f1": results[best_name]["cv_macro_f1"], "cv_low_recall": results[best_name]["cv_low_recall"],
        "test_metrics": results[best_name]["test"],
        "top_features_permutation": imp.head(8)[["feature", "importance_mean"]].round(4).to_dict("records"),
        "library_versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__},
        "random_state": RANDOM_STATE,
        "note": "Reconstructed Kaggle-based component. Not part of the original SIH 2025 submission.",
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    print(f"Saved model -> {MODEL_PATH}\nSaved metadata -> {META_PATH}")


if __name__ == "__main__":
    main()
