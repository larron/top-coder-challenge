import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree, export_text
from sklearn.metrics import mean_absolute_error

# Load the data
with open("public_cases.json") as f:
    data = json.load(f)

df = pd.DataFrame([{
    'days': row['input']['trip_duration_days'],
    'miles': row['input']['miles_traveled'],
    'receipts': row['input']['total_receipts_amount'],
    'reimbursement': row['expected_output']
} for row in data])

# Prepare input and output
X = df[['days', 'miles', 'receipts']]
y = df['reimbursement']

# Fit a decision tree
model = DecisionTreeRegressor(random_state=42)
model.fit(X, y)

# Evaluate performance
preds = model.predict(X)
mae = mean_absolute_error(y, preds)
print(f"Training MAE: {mae:.2f}")

# Print feature importance
importances = dict(zip(X.columns, model.feature_importances_))
print("Feature Importances:", importances)

# Plot the tree
plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=X.columns, filled=True, rounded=True, fontsize=10)
plt.title("Decision Tree (max_depth=4)")
plt.show()

# Export tree logic to readable text
tree_rules = export_text(model, feature_names=['days', 'miles', 'receipts'])
print("\nDecision Tree Rules:\n")
print(tree_rules)