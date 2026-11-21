
from pathlib import Path
import json
import logging
import sys
from datetime import datetime, timezone

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"

model = joblib.load(MODEL_PATH)

# JSON logging setup
logger = logging.getLogger("fare_prediction")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(handler)
logger.propagate = False


def log_prediction(request_data: dict, prediction: float):
    """Log prediction as a single JSON line to stdout."""
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "endpoint": "/predict",
        "distance_km": request_data.get("distance_km"),
        "wait_min": request_data.get("wait_min"),
        "passengers": request_data.get("passengers"),
        "hour": request_data.get("hour"),
        "luggage_bags": request_data.get("luggage_bags"),
        "fuel_price_rs": request_data.get("fuel_price_rs"),
        "booking": request_data.get("booking"),
        "zone": request_data.get("zone"),
        "predicted_fare_rs": round(float(prediction), 2),
    }
    logger.info(json.dumps(log_entry))


app = FastAPI(
    title="Auto-Rickshaw Fare Prediction API",
    description="Predict auto-rickshaw fares using a trained ML model.",
    version="1.0.0",
)


class FareRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    distance_km: float = Field(ge=0)
    wait_min: float
    passengers: int
    hour: int
    luggage_bags: int
    fuel_price_rs: float = Field(le=150)
    booking: str
    zone: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict_fare(request: FareRequest):
    input_data = pd.DataFrame([request.model_dump()])
    prediction = model.predict(input_data)[0]

    log_prediction(request.model_dump(), prediction)

    return {
        "predicted_fare_rs": round(float(prediction), 2)
    }
