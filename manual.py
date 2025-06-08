import pandas as pd
from scipy.optimize import minimize
import numpy as np

# Your dataset
data = [
    {'days': 1, 'miles': 47.0, 'receipts': 17.97, 'reimbursement': 128.91},
    {'days': 1, 'miles': 55.0, 'receipts': 3.6,   'reimbursement': 126.06},
    {'days': 1, 'miles': 58.0, 'receipts': 5.86,  'reimbursement': 117.24},
]

df = pd.DataFrame(data)

# Step 1: Apply a cap to receipts (assume system only counts up to $25)
df['receipts_capped'] = df['receipts'].clip(upper=25)

# Step 2: Band mileage down to nearest 10
df['mileage_band'] = (df['miles'] // 10) * 10

# Step 3: Adjusted reimbursement
df['adjusted'] = df['reimbursement'] - df['receipts_capped']

# Step 4: Loss function using banded miles
def loss(params):
    b, m, d = params  # b: days, m: mileage band, d: intercept
    predicted_adjusted = df['days'] * b + df['mileage_band'] * m + d
    return np.mean(np.abs(predicted_adjusted - df['adjusted']))

# Step 5: Initial guess
initial = [25, 0.5, 100]

# Step 6: Optimize
result = minimize(loss, initial, method='Nelder-Mead')
b, m, d = result.x

# Step 7: Final predicted reimbursement
df['predicted'] = df['days'] * b + df['mileage_band'] * m + d + df['receipts_capped']
df['error'] = abs(df['predicted'] - df['reimbursement'])

# Output results
print(f"Model:")
print(f"adjusted = {b:.4f} * days + {m:.4f} * mileage_band + {d:.4f}\n")

print("Results:")
print(df[['reimbursement', 'predicted', 'error']])
print(f"\nMax error: {df['error'].max():.6f}")