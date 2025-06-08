import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

# Load training data
df = pd.read_csv("train_data.csv")
X = df[["days", "miles", "receipts"]]
y = df["output"]

# Train model
model = XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.05)
model.fit(X, y)

# Evaluate
preds = model.predict(X)
rmse = np.sqrt(mean_squared_error(y, preds))
mae = mean_absolute_error(y, preds)
max_err = max(abs(preds - y))

print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"Max Error: {max_err:.4f}")

# Save model
joblib.dump(model, "reimbursement_model.pkl")