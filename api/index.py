import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Literal
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

app = FastAPI(title="Australian Weather Rain Predictor")

pipeline = joblib.load(BASE_DIR / "model.pkl")

class WeatherInput(BaseModel):
    Location: Literal["Melbourne", "MelbourneAirport", "Watsonia"]
    MinTemp: float
    MaxTemp: float
    Rainfall: float
    Evaporation: float
    Sunshine: float
    WindGustDir: str
    WindGustSpeed: float
    WindDir9am: str
    WindDir3pm: str
    WindSpeed9am: float
    WindSpeed3pm: float
    Humidity9am: float
    Humidity3pm: float
    Pressure9am: float
    Pressure3pm: float
    Cloud9am: float
    Cloud3pm: float
    Temp9am: float
    Temp3pm: float
    RainToday: Literal["Yes", "No"]
    Season: Literal["Summer", "Autumn", "Winter", "Spring"]


@app.get("/", response_class=HTMLResponse)
@app.get("/api", response_class=HTMLResponse)
def index():
    html_path = BASE_DIR / "templates" / "index.html"
    return html_path.read_text(encoding="utf-8")


@app.post("/api/predict")
@app.post("/predict")
def predict(data: WeatherInput):
    try:
        df = pd.DataFrame([data.model_dump()])
        prediction = pipeline.predict(df)[0]
        probability = pipeline.predict_proba(df)[0].max()
        return {
            "rain_tomorrow": prediction,
            "confidence": round(float(probability), 4),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
@app.get("/health")
def health():
    return {"status": "ok"}
