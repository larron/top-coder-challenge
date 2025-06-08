from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np
import joblib

df = pd.read_csv("train_data.csv")
X = df[["days", "miles", "receipts"]]
y = df["output"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=3,
    reg_alpha=1,
    reg_lambda=1,
    early_stopping_rounds=20,
    eval_metric="mae"
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

preds = model.predict(X_val)
print(f"Validation MAE: {mean_absolute_error(y_val, preds):.2f}")

joblib.dump(model, "reimbursement_model2.pkl")