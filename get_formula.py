import json
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load and prepare data
with open("public_cases.json") as f:
    data = json.load(f)

df = pd.DataFrame([{
    'days': row['input']['trip_duration_days'],
    'miles': row['input']['miles_traveled'],
    'receipts': row['input']['total_receipts_amount'],
    'reimbursement': row['expected_output']
} for row in data])

X = df[['receipts', 'days', 'miles']]  # ordered by feature importance
y = df['reimbursement']

# Fit the model
model = LinearRegression()
model.fit(X, y)

# Predict using the learned formula
preds = model.predict(X)

# Print formula
a, b, c = model.coef_
d = model.intercept_
print(f"Formula:\nreimbursement = {a:.4f} * receipts + {b:.4f} * days + {c:.4f} * miles + {d:.4f}")

# Evaluate
mae = mean_absolute_error(y, preds)
print(f"\nMean Absolute Error (MAE): {mae:.2f}")

# Optional: see examples of worst mismatches
df['predicted'] = preds
df['error'] = abs(df['reimbursement'] - df['predicted'])
print("\nTop 10 biggest mismatches:")
print(df.sort_values(by='error', ascending=False).head(10))