import pytest

pytest.importorskip("fastapi", reason="fastapi not installed in this environment")
pytest.importorskip("httpx", reason="httpx (TestClient dependency) not installed")
from fastapi.testclient import TestClient  # noqa: E402

from app.model_api import app  # noqa: E402
from tests.test_ml_pipeline import VALID  # noqa: E402

client = TestClient(app)


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_model_info():
    j = client.get("/model-info").json()
    assert j["class_labels"] == ["L", "M", "H"] and len(j["feature_names"]) == 16


def test_predict_ok():
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200 and r.json()["predicted_class"] in ["L", "M", "H"]


def test_predict_missing_fields_422():
    r = client.post("/predict", json={"gender": "M"})
    assert r.status_code == 422 and len(r.json()["detail"]["missing_features"]) == 15


def test_predict_bad_value_422():
    assert client.post("/predict", json={**VALID, "Discussion": 500}).status_code == 422
