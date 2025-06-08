import json
import pandas as pd
import numpy as np
import joblib
from sklearn.tree import DecisionTreeRegressor

# Load data
with open("public_cases.json") as f:
    raw = json.load(f)

# Flatten
df = pd.json_normalize(raw)
df.columns = ['trip_duration_days', 'miles_traveled', 'total_receipts_amount', 'expected_output']

# Features
X = df[["trip_duration_days", "miles_traveled", "total_receipts_amount"]]
y = df["expected_output"]

# Baseline model for residual analysis
from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor(max_depth=4, n_estimators=300)
model.fit(X, y)
df["model_output"] = model.predict(X)
df["abs_error"] = np.abs(df["model_output"] - df["expected_output"])

# Flag anomalies
THRESHOLD = 10.0
df["is_anomaly"] = df["abs_error"] > THRESHOLD

# Save anomaly report
anomalies = df[df["is_anomaly"]]
print(f"Found {len(anomalies)} anomalies out of {len(df)} cases.")
anomalies[["trip_duration_days", "miles_traveled", "total_receipts_amount", "expected_output", "model_output", "abs_error"]].to_csv("anomalies.csv", index=False)

# Train clean model
clean_df = df[~df["is_anomaly"]]
X_clean = clean_df[["trip_duration_days", "miles_traveled", "total_receipts_amount"]]
y_clean = clean_df["expected_output"]

clean_model = DecisionTreeRegressor(max_depth=6)
clean_model.fit(X_clean, y_clean)

joblib.dump(clean_model, "clean_model.pkl")
print("Clean model saved to clean_model.pkl")