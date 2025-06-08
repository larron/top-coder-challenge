import sys

def predict_reimbursement(days, miles, receipts):
    base = 34.77
    per_day = 50.67
    per_mile = 0.31
    receipt_pct = 0.71
    receipt_cap = 1478.57

    capped_receipts = min(receipts, receipt_cap)
    return round(
        base +
        per_day * days +
        per_mile * miles +
        receipt_pct * capped_receipts,
        2
    )

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python reimbursement.py <days> <miles> <receipts>")
        sys.exit(1)

    days = float(sys.argv[1])
    miles = float(sys.argv[2])
    receipts = float(sys.argv[3])

    result = predict_reimbursement(days, miles, receipts)
    print(result)