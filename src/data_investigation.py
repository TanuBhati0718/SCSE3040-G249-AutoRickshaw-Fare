import pandas as pd
import numpy as np

train = pd.read_csv("data/raw/train.csv")
later = pd.read_csv("data/raw/later.csv")
holdout = pd.read_csv("data/raw/holdout_features.csv")

print("\n========== 1. LEAKING COLUMN ==========")
for col in train.columns:
    if col not in holdout.columns:
        print("Missing from holdout:", col)

if "fare_paid_receipt_rs" in train.columns:
    print("Receipt-target correlation:",
          train["fare_paid_receipt_rs"].corr(train["fare_rs"]))

print("\n========== 2. UNIT MIXUP ==========")
numeric = [
    "distance_km", "wait_min", "passengers",
    "hour", "luggage_bags", "fuel_price_rs"
]

for col in numeric:
    print(f"\nColumn: {col}")
    print(train.groupby("zone")[col].agg(
        ["count", "min", "median", "mean", "max"]
    ))

print("\n========== 3. SENTINEL VALUES ==========")
sentinels = [-1, -999, 999, 9999]

for col in train.select_dtypes(include="number").columns:
    for value in sentinels:
        count = (train[col] == value).sum()
        if count > 0:
            print(col, "Value:", value, "Count:", count)

print("\n========== 4. DUPLICATE ROWS ==========")
print("Extra duplicate rows:", train.duplicated().sum())
print("Duplicate record IDs:", train["record_id"].duplicated().sum())

print("\n========== 5. STUCK SENSOR ==========")
sorted_train = train.sort_values("date").reset_index(drop=True)

for col in numeric:
    values = sorted_train[col]
    groups = values.ne(values.shift()).cumsum()
    runs = sorted_train.groupby(groups).agg(
        value=(col, "first"),
        count=(col, "size"),
        start_date=("date", "first"),
        end_date=("date", "last")
    )
    print(f"\nLongest repeated runs for {col}:")
    print(runs.sort_values("count", ascending=False).head(5).to_string(index=False))

print("\n========== 6. TARGET TYPOS ==========")
target = train["fare_rs"].dropna().sort_values()
print("Target summary:")
print(target.describe())
print("\nLargest target values:")
print(target.tail(25).to_string(index=False))

print("\nPossible 10x target typos:")
for value in target.unique():
    smaller = target[(target - value / 10).abs() < 0.02]
    if len(smaller) > 0 and value > target.median() * 5:
        print("Possible typo:", value, "Possible correct value:", value / 10)

print("\n========== 7. SPELLING VARIANTS ==========")
text_columns = train.select_dtypes(include="object").columns

for col in text_columns:
    print(f"\nColumn: {col}")
    print("TRAIN:")
    print(train[col].value_counts(dropna=False).to_string())
    if col in later.columns:
        print("LATER:")
        print(later[col].value_counts(dropna=False).to_string())

print("\n========== INVESTIGATION COMPLETE ==========")
