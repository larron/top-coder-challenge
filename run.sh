#!/bin/bash

# Black Box Challenge - Reimbursement Calculation
# Usage: ./run.sh <trip_duration_days> <miles_traveled> <total_receipts_amount>

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <trip_duration_days> <miles_traveled> <total_receipts_amount>"
  exit 1
fi

DAYS=$1
MILES=$2
RECEIPTS=$3

python3 predict.py "$DAYS" "$MILES" "$RECEIPTS"