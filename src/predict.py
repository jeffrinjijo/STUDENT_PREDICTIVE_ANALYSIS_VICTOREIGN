"""Inference on the saved model - no retraining required.

    from src.predict import predict_one
    predict_one({...all 16 features...})

Inputs are validated against the saved metadata. Missing features, non-numeric or
out-of-range engagement counts, and categories never seen in training are reported
as errors; values are never guessed or defaulted.
"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Iterable

import joblib
import pandas as pd

from .preprocessing import PROJECT_ROOT

MODEL_PATH = PROJECT_ROOT / "models" / "academic_performance_model.pkl"
META_PATH = PROJECT_ROOT / "models" / "model_metadata.json"


class InvalidInputError(ValueError):
    """Raised with a list of human-readable problems with the input record."""

    def __init__(self, errors: list[str], missing: list[str] | None = None):
        super().__init__("; ".join(errors))
        self.errors = errors
        self.missing = missing or []


@lru_cache(maxsize=1)
def load_artifacts():
    """Load (pipeline, metadata) once per process."""
    if not MODEL_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError("Model artifacts not found. Run `python -m src.train` first.")
    return joblib.load(MODEL_PATH), json.loads(META_PATH.read_text())


def validate_record(rec: dict[str, Any], meta: dict) -> dict[str, Any]:
    errors, missing, clean = [], [], {}
    lo, hi = meta["numeric_valid_range"]
    for f in meta["feature_names"]:
        v = rec.get(f)
        if v is None or (isinstance(v, str) and not v.strip()):
            missing.append(f)
            continue
        if f in meta["numeric_features"]:
            try:
                x = float(v)
            except (TypeError, ValueError):
                errors.append(f"{f}: expected a number, got {v!r}")
                continue
            if not lo <= x <= hi:
                errors.append(f"{f}: {x} is outside the valid range {lo}-{hi}")
                continue
            clean[f] = x
        else:
            allowed = meta["categorical_values_seen"][f]
            if v not in allowed:
                errors.append(f"{f}: {v!r} is not a known value; allowed: {allowed}")
                continue
            clean[f] = v
    if missing:
        errors.insert(0, f"missing required features: {missing}")
    extra = sorted(set(rec) - set(meta["feature_names"]))
    if extra:
        errors.append(f"unknown fields (ignored by the model, rejected for clarity): {extra}")
    if errors:
        raise InvalidInputError(errors, missing)
    return clean


def predict_batch(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    model, meta = load_artifacts()
    cleaned = [validate_record(r, meta) for r in records]
    X = pd.DataFrame(cleaned, columns=meta["feature_names"])
    proba = model.predict_proba(X)
    out = []
    for row in proba:
        probs = {c: round(float(p), 4) for c, p in zip(model.classes_, row)}
        label = max(probs, key=probs.get)
        out.append({
            "predicted_class": label,
            "predicted_label": meta["class_names"][label],
            "probabilities": {c: probs[c] for c in meta["class_labels"]},
            "low_performance_risk": probs["L"],
            "model": meta["model_name"],
        })
    return out


def predict_one(record: dict[str, Any]) -> dict[str, Any]:
    return predict_batch([record])[0]


if __name__ == "__main__":
    demo = {"gender": "M", "NationalITy": "Jordan", "PlaceofBirth": "Jordan", "StageID": "MiddleSchool",
            "GradeID": "G-08", "SectionID": "A", "Topic": "Math", "Semester": "F", "Relation": "Father",
            "raisedhands": 20, "VisITedResources": 25, "AnnouncementsView": 10, "Discussion": 30,
            "ParentAnsweringSurvey": "No", "ParentschoolSatisfaction": "Bad", "StudentAbsenceDays": "Above-7"}
    print(json.dumps(predict_one(demo), indent=2))
