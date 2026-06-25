# Rainfall Prediction Classifier - Australian Weather Dataset

A machine learning project that predicts whether it will rain tomorrow using historical weather observations from Australia (2008–2017), served via a FastAPI web application.

## Dataset

Source: [Australian Bureau of Meteorology](http://www.bom.gov.au/climate/dwo/) via [Kaggle](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package/)

Filtered to the **Melbourne region** (Melbourne, MelbourneAirport, Watsonia). Dataset is fetched automatically from a public URL — no manual download needed.

LIVE PROJECT RUNNING :- rainfall-project-ten.vercel.app

## Project Structure

```
├── FinalProject_AUSWeather_Final.ipynb  # EDA, modeling, evaluation
├── train.py                             # Train and save model.pkl
├── app.py                               # FastAPI prediction server
├── templates/
│   └── index.html                       # Browser UI
├── requirements.txt
├── .gitignore
└── README.md
```

## Models

- **Random Forest Classifier** — sklearn `Pipeline` with `ColumnTransformer` preprocessing (~84% accuracy)
- **Logistic Regression** — comparison baseline (in notebook)

## Setup

```bash
pip install -r requirements.txt
```

## Usage

**1. Train the model** — downloads data, trains, and saves `model.pkl`
```bash
python train.py
```

**2. Start the API**
```bash
python -m uvicorn app:app --reload
```

**3. Open in browser**
```
http://127.0.0.1:8000
```

Fill in the weather observation form and get a prediction with confidence score.
Interactive API docs available at `http://127.0.0.1:8000/docs`.

## API

`POST /predict`

Request body (JSON):
```json
{
  "Location": "Melbourne",
  "MinTemp": 10.0,
  "MaxTemp": 22.0,
  "Rainfall": 0.0,
  "Evaporation": 5.0,
  "Sunshine": 8.0,
  "WindGustDir": "S",
  "WindGustSpeed": 40.0,
  "WindDir9am": "S",
  "WindDir3pm": "S",
  "WindSpeed9am": 15.0,
  "WindSpeed3pm": 20.0,
  "Humidity9am": 70.0,
  "Humidity3pm": 50.0,
  "Pressure9am": 1015.0,
  "Pressure3pm": 1012.0,
  "Cloud9am": 4.0,
  "Cloud3pm": 4.0,
  "Temp9am": 14.0,
  "Temp3pm": 20.0,
  "RainToday": "No",
  "Season": "Winter"
}
```

Response:
```json
{
  "rain_tomorrow": "No",
  "confidence": 0.87
}
```
