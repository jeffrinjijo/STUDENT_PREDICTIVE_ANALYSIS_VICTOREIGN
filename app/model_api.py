"""FastAPI microservice exposing the saved academic-performance model.

Run from the project root:  uvicorn app.model_api:app --port 8000
"""
from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from src.predict import InvalidInputError, load_artifacts, predict_one

app = FastAPI(title="Academic Performance Prediction API", version="1.0.0")


class StudentFeatures(BaseModel):
    """All 16 dataset features are required; nothing is defaulted.

    Value-level validation (ranges, known categories) is done by src.predict so the
    same rules apply to API and Python callers and errors list every problem at once.
    """
    model_config = ConfigDict(extra="allow")
    gender: str | None = None
    NationalITy: str | None = None
    PlaceofBirth: str | None = None
    StageID: str | None = None
    GradeID: str | None = None
    SectionID: str | None = None
    Topic: str | None = None
    Semester: str | None = None
    Relation: str | None = None
    raisedhands: float | str | None = None
    VisITedResources: float | str | None = None
    AnnouncementsView: float | str | None = None
    Discussion: float | str | None = None
    ParentAnsweringSurvey: str | None = None
    ParentschoolSatisfaction: str | None = None
    StudentAbsenceDays: str | None = None


class PredictionResponse(BaseModel):
    predicted_class: Literal["L", "M", "H"]
    predicted_label: str
    probabilities: dict[str, float]
    low_performance_risk: float
    model: str


@app.get("/health")
def health():
    try:
        load_artifacts()
        return {"status": "ok", "model_loaded": True}
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/model-info")
def model_info():
    try:
        _, meta = load_artifacts()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    keys = ["model_name", "trained_at_utc", "dataset", "rows_used", "target", "class_labels",
            "class_names", "feature_names", "numeric_features", "categorical_features",
            "categorical_values_seen", "numeric_valid_range", "cv_macro_f1", "cv_low_recall",
            "test_metrics", "note"]
    return {k: meta[k] for k in keys}


@app.post("/predict", response_model=PredictionResponse)
def predict(student: StudentFeatures):
    try:
        return predict_one(student.model_dump())
    except InvalidInputError as e:
        raise HTTPException(status_code=422, detail={"message": "Invalid student record",
                                                     "errors": e.errors, "missing_features": e.missing})
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
