# Rainfall Prediction Classifier - Australian Weather Dataset

A machine learning project that predicts whether it will rain tomorrow using historical weather observations from Australia (2008–2017).

## Dataset

Source: [Australian Bureau of Meteorology](http://www.bom.gov.au/climate/dwo/) via [Kaggle](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package/)

The dataset contains daily weather observations across multiple Australian locations. This project focuses on the **Melbourne region** (Melbourne, Melbourne Airport, Watsonia).

## Project Overview

| Step | Description |
|------|-------------|
| Data Loading | Load dataset directly from URL |
| Preprocessing | Drop missing values, rename columns, filter to Melbourne region |
| Feature Engineering | Extract `Season` from `Date` column |
| Modeling | Sklearn `Pipeline` with `ColumnTransformer` for preprocessing |
| Optimization | `GridSearchCV` with `StratifiedKFold` cross-validation |
| Evaluation | Classification report, confusion matrix, feature importances |
| Comparison | Random Forest vs Logistic Regression |

## Models

- **Random Forest Classifier** — tuned via grid search over `n_estimators`, `max_depth`, `min_samples_split`
- **Logistic Regression** — tuned via grid search over `solver`, `penalty`, `class_weight`

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
```

Install with:

```bash
pip install -r requirements.txt
```

## Usage

Open and run the notebook top to bottom:

```bash
jupyter notebook FinalProject_AUSWeather_Final.ipynb
```

All data is fetched automatically from a public URL — no manual download needed.

## Results

Both models achieve ~84% accuracy on the test set. Random Forest generally outperforms Logistic Regression on the minority class (rain days), which are underrepresented (~22% of observations).
