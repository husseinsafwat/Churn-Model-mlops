SAMPLE_PAYLOAD = {
    "CreditScore": 600,
    "Geography": "France",
    "Gender": "Male",
    "Age": 40,
    "Tenure": 3,
    "Balance": 60000.0,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 50000.0,
}

# ---------------------------------------------------------------------------
# Endpoint Tests
# ---------------------------------------------------------------------------

def test_predict_endpoint_valid(client):
    response = client.post("/predict", json=SAMPLE_PAYLOAD)
    assert response.status_code == 200, response.text
    body = response.json()
    assert "prediction" in body
    assert "probability" in body
    assert body["prediction"] in (0, 1)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_home_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_predict_endpoint_invalid_credit_score(client):
    payload = {**SAMPLE_PAYLOAD, "CreditScore": 999}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_endpoint_invalid_geography(client):
    payload = {**SAMPLE_PAYLOAD, "Geography": "Egypt"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_endpoint_invalid_gender(client):
    payload = {**SAMPLE_PAYLOAD, "Gender": "Unknown"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_endpoint_missing_fields(client):
    response = client.post("/predict", json={"CreditScore": 600})
    assert response.status_code == 422


def test_predict_endpoint_empty_body(client):
    response = client.post("/predict", json={})
    assert response.status_code == 422