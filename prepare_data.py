import json
import pandas as pd

with open("public_cases.json", "r") as f:
    raw = json.load(f)

data = [{
    "days": x["input"]["trip_duration_days"],
    "miles": x["input"]["miles_traveled"],
    "receipts": x["input"]["total_receipts_amount"],
    "output": x["expected_output"]
} for x in raw]

df = pd.DataFrame(data)
df.to_csv("train_data.csv", index=False)