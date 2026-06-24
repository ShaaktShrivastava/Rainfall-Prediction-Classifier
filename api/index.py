import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Literal
from pathlib import Path
import os

# For Vercel serverless, we need to find files relative to the project root
BASE_DIR = Path(__file__).parent.parent if os.path.exists(Path(__file__).parent.parent / "model.pkl") else Path("/var/task")

app = FastAPI(title="Australian Weather Rain Predictor")

# Load model with error handling
try:
    model_path = BASE_DIR / "model.pkl"
    pipeline = joblib.load(model_path)
except Exception as e:
    print(f"Error loading model from {BASE_DIR}: {e}")
    pipeline = None

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
    try:
        html_path = BASE_DIR / "templates" / "index.html"
        if not html_path.exists():
            return HTMLResponse(content=f"<h1>Error: Template not found at {html_path}</h1><p>BASE_DIR: {BASE_DIR}</p>", status_code=500)
        return html_path.read_text(encoding="utf-8")
    except Exception as e:
        return HTMLResponse(content=f"<h1>Error loading template</h1><p>{str(e)}</p><p>BASE_DIR: {BASE_DIR}</p>", status_code=500)


@app.post("/api/predict")
@app.post("/predict")
def predict(data: WeatherInput):
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
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
    return {
        "status": "ok",
        "model_loaded": pipeline is not None,
        "base_dir": str(BASE_DIR)
    }
