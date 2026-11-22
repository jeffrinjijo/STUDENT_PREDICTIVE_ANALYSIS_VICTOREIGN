"""Data loading and preprocessing for the xAPI-Edu-Data academic performance model.

Dataset: Students' Academic Performance Dataset (xAPI-Edu-Data), Kaggle aljarah/xAPI-Edu-Data.
Column names are kept exactly as in the CSV (including its spelling, e.g. ``NationalITy``).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = PROJECT_ROOT / "dataset" / "raw" / "xAPI-Edu-Data.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET = "Class"
CLASS_ORDER = ["L", "M", "H"]  # Low < Middle < High
CLASS_NAMES = {"L": "Low", "M": "Middle", "H": "High"}

NUMERIC_FEATURES = ["raisedhands", "VisITedResources", "AnnouncementsView", "Discussion"]
CATEGORICAL_FEATURES = [
    "gender", "NationalITy", "PlaceofBirth", "StageID", "GradeID", "SectionID",
    "Topic", "Semester", "Relation", "ParentAnsweringSurvey",
    "ParentschoolSatisfaction", "StudentAbsenceDays",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
NUMERIC_RANGE = (0, 100)  # all four engagement counters are 0-100 in the data


def load_data(path: Path | str = RAW_CSV, drop_duplicates: bool = True) -> pd.DataFrame:
    """Read the raw CSV and validate that the expected columns are present.

    The raw file on disk is never modified. Exact duplicate rows (2 in the
    original data) are dropped in memory so identical records cannot land in
    both the train and test split (a leakage source).
    """
    df = pd.read_csv(path)
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")
    if drop_duplicates:
        df = df.drop_duplicates().reset_index(drop=True)
    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[FEATURES].copy(), df[TARGET].copy()


def train_test_split_stratified(df: pd.DataFrame):
    """Stratified 80/20 split, deterministic via RANDOM_STATE."""
    X, y = split_features_target(df)
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)


def build_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    """ColumnTransformer: one-hot for categoricals; optional scaling for numerics.

    Scaling is only enabled for scale-sensitive models (Logistic Regression);
    tree-based models receive numeric features unscaled. The transformer is
    always placed inside a Pipeline so it is fitted on training folds only.
    ``handle_unknown='ignore'`` lets the saved pipeline accept categories not
    seen in training (all-zero encoding) instead of crashing.
    """
    numeric = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )
