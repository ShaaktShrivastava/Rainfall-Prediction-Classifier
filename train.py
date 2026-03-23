import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ── Load ──────────────────────────────────────────────────────────────────────
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/_0eYOqji3unP1tDNKWZMjg/weatherAUS-2.csv"
df = pd.read_csv(url)

# ── Preprocess ────────────────────────────────────────────────────────────────
df = df.dropna()
df = df[df["Location"].isin(["Melbourne", "MelbourneAirport", "Watsonia"])]

df["Date"] = pd.to_datetime(df["Date"])

def date_to_season(date):
    m = date.month
    if m in [12, 1, 2]:  return "Summer"
    if m in [3, 4, 5]:   return "Autumn"
    if m in [6, 7, 8]:   return "Winter"
    return "Spring"

df["Season"] = df["Date"].apply(date_to_season)
df = df.drop(columns=["Date"])

# Fix duplicate column issue from notebook
df = df.loc[:, ~df.columns.duplicated()]

X = df.drop(columns=["RainTomorrow"])
y = df["RainTomorrow"]

# ── Pipeline ──────────────────────────────────────────────────────────────────
num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(include="object").columns.tolist()

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(n_estimators=100, random_state=42)),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

pipeline.fit(X_train, y_train)

# ── Save ──────────────────────────────────────────────────────────────────────
joblib.dump(pipeline, "model.pkl")
print(f"Model saved. Test accuracy: {pipeline.score(X_test, y_test):.4f}")
