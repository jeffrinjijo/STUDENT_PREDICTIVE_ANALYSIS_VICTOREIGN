import json
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.predict import InvalidInputError, load_artifacts, predict_batch, predict_one, validate_record
from src.preprocessing import (CLASS_ORDER, FEATURES, NUMERIC_FEATURES, RAW_CSV, TARGET,
                               build_preprocessor, load_data, train_test_split_stratified)

VALID = {"gender": "M", "NationalITy": "Jordan", "PlaceofBirth": "Jordan", "StageID": "MiddleSchool",
         "GradeID": "G-08", "SectionID": "A", "Topic": "Math", "Semester": "F", "Relation": "Father",
         "raisedhands": 80, "VisITedResources": 90, "AnnouncementsView": 70, "Discussion": 60,
         "ParentAnsweringSurvey": "Yes", "ParentschoolSatisfaction": "Good", "StudentAbsenceDays": "Under-7"}


@pytest.fixture(scope="module")
def df():
    return load_data()


def test_dataset_loading(df):
    raw = pd.read_csv(RAW_CSV)
    assert raw.shape == (480, 17)
    assert set(FEATURES + [TARGET]) == set(raw.columns)
    assert df.isna().sum().sum() == 0
    assert len(df) == len(raw) - raw.duplicated().sum()
    assert set(df[TARGET]) == set(CLASS_ORDER)


def test_missing_column_rejected(tmp_path, df):
    p = tmp_path / "bad.csv"; df.drop(columns=["Topic"]).to_csv(p, index=False)
    with pytest.raises(ValueError):
        load_data(p)


def test_split_is_stratified_and_disjoint(df):
    X_tr, X_te, y_tr, y_te = train_test_split_stratified(df)
    assert len(X_tr) + len(X_te) == len(df) and set(X_tr.index).isdisjoint(X_te.index)
    for c in CLASS_ORDER:  # class shares preserved within one row of rounding
        assert abs((y_tr == c).mean() - (y_te == c).mean()) < 0.03


def test_preprocessing_fit_on_train_only(df):
    X_tr, X_te, _, _ = train_test_split_stratified(df)
    prep = build_preprocessor(scale_numeric=True).fit(X_tr)
    scaler = prep.named_transformers_["num"]
    assert np.allclose(scaler.mean_, X_tr[NUMERIC_FEATURES].mean().values)  # not influenced by test rows
    assert not np.allclose(scaler.mean_, df[NUMERIC_FEATURES].mean().values)
    assert prep.transform(X_te).shape[0] == len(X_te)


def test_tree_preprocessor_does_not_scale(df):
    X_tr, _, _, _ = train_test_split_stratified(df)
    out = build_preprocessor(scale_numeric=False).fit_transform(X_tr)
    out = out.toarray() if hasattr(out, 'toarray') else np.asarray(out)
    assert np.allclose(out[:, :4], X_tr[NUMERIC_FEATURES].values)


def test_training_and_serialization_roundtrip(tmp_path, df):
    X_tr, X_te, y_tr, _ = train_test_split_stratified(df)
    pipe = Pipeline([("prep", build_preprocessor(False)),
                     ("clf", RandomForestClassifier(n_estimators=30, random_state=0))]).fit(X_tr, y_tr)
    f = tmp_path / "m.pkl"; joblib.dump(pipe, f)
    assert np.array_equal(pipe.predict(X_te), joblib.load(f).predict(X_te))


def test_saved_model_loads_and_metadata_consistent():
    model, meta = load_artifacts()
    assert meta["feature_names"] == FEATURES
    assert list(model.classes_) == meta["probability_class_order"]
    assert meta["test_metrics"]["accuracy"] > 0.6  # sanity floor, not a target


def test_prediction_output_shape():
    r = predict_one(VALID)
    assert r["predicted_class"] in CLASS_ORDER
    assert set(r["probabilities"]) == set(CLASS_ORDER)
    assert abs(sum(r["probabilities"].values()) - 1) < 1e-3


def test_prediction_direction_makes_sense():
    low = dict(VALID, raisedhands=5, VisITedResources=5, AnnouncementsView=3, Discussion=10,
               StudentAbsenceDays="Above-7", ParentAnsweringSurvey="No", ParentschoolSatisfaction="Bad")
    assert predict_one(low)["probabilities"]["L"] > predict_one(VALID)["probabilities"]["L"]


def test_batch_matches_single():
    assert predict_batch([VALID, VALID])[0] == predict_one(VALID)


def test_saved_model_reproduces_recorded_test_metrics(df):
    model, meta = load_artifacts()
    _, X_te, _, y_te = train_test_split_stratified(df)
    assert abs((model.predict(X_te) == y_te).mean() - meta["test_metrics"]["accuracy"]) < 1e-9


@pytest.mark.parametrize("mutation,fragment", [
    ({"raisedhands": None}, "missing required features"),
    ({"raisedhands": "abc"}, "expected a number"),
    ({"Discussion": 150}, "outside the valid range"),
    ({"VisITedResources": -1}, "outside the valid range"),
    ({"gender": "X"}, "not a known value"),
    ({"extra_field": 1}, "unknown fields"),
])
def test_invalid_inputs_rejected(mutation, fragment):
    with pytest.raises(InvalidInputError) as e:
        predict_one({**VALID, **mutation})
    assert fragment in str(e.value)


def test_all_missing_reports_every_feature():
    with pytest.raises(InvalidInputError) as e:
        validate_record({}, load_artifacts()[1])
    assert len(e.value.missing) == 16
