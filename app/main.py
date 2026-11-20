
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"

model = joblib.load(MODEL_PATH)

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

    return {
        "predicted_fare_rs": round(float(prediction), 2)
    }
