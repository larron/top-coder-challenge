import joblib
import sys

# Load the trained model (ensure reimbursement_model.pkl exists)
model = joblib.load("reimbursement_model.pkl")
# model = joblib.load("reimbursement_model2.pkl")
# model = joblib.load("clean_model.pkl")

def predict(days, miles, receipts):
    return model.predict([[float(days), float(miles), float(receipts)]])[0]

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: predict.py <days> <miles> <receipts>")
        sys.exit(1)

    d, m, r = sys.argv[1:4]
    output = predict(d, m, r)
    print(f"{output:.2f}")