"""Evaluation helpers: metrics, confusion matrices, feature importance."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)

from .preprocessing import CLASS_ORDER, PROJECT_ROOT, RANDOM_STATE

REPORT_DIR = PROJECT_ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"


def compute_metrics(y_true, y_pred, y_proba=None, classes=None) -> dict:
    """Accuracy, per-class / macro / weighted P-R-F1, confusion matrix, OvR ROC-AUC.

    The confusion matrix uses CLASS_ORDER (rows = true, cols = predicted).
    ROC-AUC is one-vs-rest, macro and weighted averaged, using predicted
    probabilities (``classes`` gives the column order of ``y_proba``).
    """
    out = {"accuracy": float(accuracy_score(y_true, y_pred))}
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=CLASS_ORDER, zero_division=0)
    out["per_class"] = {
        c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
        for i, c in enumerate(CLASS_ORDER)
    }
    for avg in ("macro", "weighted"):
        pa, ra, fa, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=CLASS_ORDER, average=avg, zero_division=0)
        out[f"{avg}_precision"], out[f"{avg}_recall"], out[f"{avg}_f1"] = float(pa), float(ra), float(fa)
    out["low_recall"] = out["per_class"]["L"]["recall"]
    out["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=CLASS_ORDER).tolist()
    if y_proba is not None:
        labels_sorted = sorted(CLASS_ORDER)  # roc_auc_score requires sorted labels
        order = [list(classes).index(c) for c in labels_sorted]
        proba = np.asarray(y_proba)[:, order]
        for avg in ("macro", "weighted"):
            out[f"roc_auc_ovr_{avg}"] = float(roc_auc_score(
                y_true, proba, multi_class="ovr", average=avg, labels=labels_sorted))
    return out


def plot_confusion(cm, title: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    sns.heatmap(np.asarray(cm), annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def permutation_feature_importance(pipeline, X_test, y_test, n_repeats: int = 20) -> pd.DataFrame:
    """Permutation importance on the ORIGINAL input columns (macro-F1 drop).

    Computed on held-out data and on raw columns, so one-hot columns of the same
    feature are not split apart and the result is comparable across model types.
    """
    res = permutation_importance(pipeline, X_test, y_test, scoring="f1_macro",
                                 n_repeats=n_repeats, random_state=RANDOM_STATE)
    df = pd.DataFrame({"feature": X_test.columns, "importance_mean": res.importances_mean,
                       "importance_std": res.importances_std})
    return df.sort_values("importance_mean", ascending=False).reset_index(drop=True)


def plot_importance(df: pd.DataFrame, title: str, path: Path, top: int = 12) -> None:
    d = df.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.barh(d["feature"], d["importance_mean"], xerr=d["importance_std"], color="#4C72B0")
    ax.set_xlabel("Drop in macro-F1 when shuffled"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main() -> None:
    """Re-evaluate the SAVED model on the deterministic held-out test split."""
    import joblib
    from .preprocessing import load_data, train_test_split_stratified

    model = joblib.load(PROJECT_ROOT / "models" / "academic_performance_model.pkl")
    _, X_test, _, y_test = train_test_split_stratified(load_data())
    m = compute_metrics(y_test, model.predict(X_test), model.predict_proba(X_test), model.classes_)
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
