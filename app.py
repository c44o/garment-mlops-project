"""Serve the saved sklearn pipeline via FastAPI."""
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).resolve().parent / 'models' / 'model.joblib'
app = FastAPI(title='Garment productivity predictor', version='1.0.0')
model = joblib.load(MODEL_PATH)


class PredictionInput(BaseModel):
    targeted_productivity: float = Field(ge=0)
    smv: float
    wip: float | None = None
    over_time: float
    incentive: float
    idle_time: float
    idle_men: int
    no_of_style_change: int
    no_of_workers: float
    quarter: str
    department: str
    day: str
    team: int


@app.get('/health')
def health() -> dict[str, str]:
    return {'status': 'ok'}


@app.post('/predict')
def predict(payload: PredictionInput) -> dict[str, float]:
    row = payload.model_dump()
    row['department'] = row['department'].strip().lower().replace('sweing', 'sewing')
    predicted = float(model.predict(pd.DataFrame([row]))[0])
    return {'predicted_actual_productivity': predicted}
