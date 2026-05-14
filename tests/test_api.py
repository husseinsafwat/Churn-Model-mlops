"""API tests: schema, helper, and FastAPI endpoints."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import main


def test_churn_features_parses_example_payload() -> None:
    payload = {
        "CreditScore": 700,
        "Geography": "France",
        "Gender": "Male",
        "Age": 50,
        "Tenure": 5,
        "Balance": 90000.0,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 0,
        "EstimatedSalary": 80000.0,
    }
    row = main.ChurnFeatures.model_validate(payload)
    assert row.model_dump() == payload


def test_churn_features_missing_field_raises() -> None:
    with pytest.raises(ValidationError) as exc:
        main.ChurnFeatures.model_validate({"CreditScore": 700})
    assert "Geography" in str(exc.value)


def test_count_records_is_function_unit() -> None:
    assert main.count_records({"a": 1}) == 1
    assert main.count_records([{"a": 1}, {"a": 2}]) == 2
    assert main.count_records([]) == 0
    assert main.count_records("bad") == 0


@pytest.fixture
def client() -> TestClient:
    return TestClient(main.app)


def test_home_endpoint(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["message"] == "Churn inference API"


def test_health_endpoint(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def _sample_row() -> dict:
    return {
        "CreditScore": 700,
        "Geography": "France",
        "Gender": "Male",
        "Age": 50,
        "Tenure": 5,
        "Balance": 90000.0,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 0,
        "EstimatedSalary": 80000.0,
    }


@patch.object(main, "run_inference")
def test_predict_single_returns_flat_prediction(mock_inf, client: TestClient) -> None:
    mock_inf.return_value = [{"prediction": 0, "probability": 0.25}]
    r = client.post("/predict", json=_sample_row())
    assert r.status_code == 200
    assert r.json() == {"prediction": 0, "probability": 0.25}
    mock_inf.assert_called_once()


@patch.object(main, "run_inference")
def test_predict_batch_returns_results_only(mock_inf, client: TestClient) -> None:
    mock_inf.return_value = [
        {"prediction": 0, "probability": 0.1},
        {"prediction": 1, "probability": 0.9},
    ]
    r = client.post("/predict", json=[_sample_row(), {**_sample_row(), "Age": 30}])
    assert r.status_code == 200
    assert r.json() == {
        "results": [
            {"prediction": 0, "probability": 0.1},
            {"prediction": 1, "probability": 0.9},
        ]
    }


def test_predict_empty_list_returns_400(client: TestClient) -> None:
    r = client.post("/predict", json=[])
    assert r.status_code == 400


@patch.object(main, "run_inference", side_effect=ValueError("bad input"))
def test_predict_value_error_returns_400(mock_inf, client: TestClient) -> None:
    r = client.post("/predict", json=_sample_row())
    assert r.status_code == 400
    assert "bad input" in r.json()["detail"]


@patch.object(main, "run_inference", side_effect=RuntimeError("server down"))
def test_predict_other_error_returns_500(mock_inf, client: TestClient) -> None:
    r = client.post("/predict", json=_sample_row())
    assert r.status_code == 500
    assert r.json()["detail"] == "inference failed"
