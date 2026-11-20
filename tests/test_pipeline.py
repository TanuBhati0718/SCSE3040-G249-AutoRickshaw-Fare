
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_valid_input():
    payload = {
        "distance_km": 5.2,
        "wait_min": 4,
        "passengers": 2,
        "hour": 14,
        "luggage_bags": 1,
        "fuel_price_rs": 105,
        "booking": "app",
        "zone": "Market",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert "predicted_fare_rs" in response.json()
    assert isinstance(response.json()["predicted_fare_rs"], float)


def test_predict_rejects_negative_distance():
    payload = {
        "distance_km": -5,
        "wait_min": 4,
        "passengers": 2,
        "hour": 14,
        "luggage_bags": 1,
        "fuel_price_rs": 105,
        "booking": "app",
        "zone": "Market",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_rejects_high_fuel_price():
    payload = {
        "distance_km": 5.2,
        "wait_min": 4,
        "passengers": 2,
        "hour": 14,
        "luggage_bags": 1,
        "fuel_price_rs": 200,
        "booking": "app",
        "zone": "Market",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
