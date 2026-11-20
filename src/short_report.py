import pandas as pd
import numpy as np
import re

train = pd.read_csv("data/train.csv")
later = pd.read_csv("data/later.csv")

numeric = [
    "distance_km", "wait_min", "passengers",
    "hour", "luggage_bags", "fuel_price_rs"
]

print("\n========== 1. LEAKING COLUMN ==========")
print("Columns missing from holdout:")
holdout = pd.read_csv("data/holdout_features.csv")
print([c for c in train.columns if c not in holdout.columns])
print("Receipt-target correlation:",
      train["fare_paid_receipt_rs"].corr(train["fare_rs"]))

print("\n========== 2. UNIT MIXUP ==========")
for col in numeric:
    print("\n", col)
    print(train.groupby("zone")[col].agg(
        ["min", "median", "mean", "max"]
    ).round(2))

print("\n========== 3. SENTINEL VALUES ==========")
for col in train.select_dtypes(include="number").columns:
    counts = train[col].value_counts()
    found = counts[counts.index.isin([-1, -999, 999, 9999])]
    if len(found):
        print(col, found.to_dict())

print("\n========== 4. DUPLICATES ==========")
print("Full duplicate rows:", train.duplicated().sum())
print("Duplicate record IDs:", train["record_id"].duplicated().sum())

print("\n========== 5. STUCK SENSOR ==========")
daily = train.copy()
daily["date"] = pd.to_datetime(daily["date"])
for col in numeric:
    day_values = daily.groupby("date")[col].agg(
        lambda x: x.iloc[0] if x.nunique() == 1 else np.nan
    ).dropna()

    runs = []
    dates = list(day_values.index)
    vals = list(day_values.values)

    start = 0
    for i in range(1, len(vals) + 1):
        if (i == len(vals) or vals[i] != vals[start]
                or (dates[i] - dates[i-1]).days != 1):
            if i - start >= 3:
                runs.append((i-start, vals[start],
                             dates[start].date(), dates[i-1].date()))
            start = i

    print(col, "Repeated daily runs:", sorted(
        runs, reverse=True
    )[:3])

print("\n========== 6. TARGET TYPOS ==========")
print("Target summary:")
print(train["fare_rs"].describe().round(2))
print("\nLargest 15 fares:")
print(train["fare_rs"].nlargest(15).to_list())

print("\nPossible 10x values:")
for value in sorted(train["fare_rs"].dropna().unique()):
    matches = train.loc[
        np.isclose(train["fare_rs"], value / 10, atol=0.01),
        "fare_rs"
    ]
    if value > train["fare_rs"].median() * 5 and len(matches):
        print("Large:", value, "Possible normal value:", value/10)

print("\n========== 7. SPELLING VARIANTS ==========")
for col in ["booking", "zone"]:
    print("\nColumn:", col)
    print("Raw training values:", train[col].value_counts(
        dropna=False
    ).to_dict())
    print("Later values:", later[col].value_counts(
        dropna=False
    ).to_dict())

    normalized = train[col].dropna().astype(str).str.strip().str.lower()
    print("Normalized categories:", sorted(normalized.unique()))

print("\n========== REPORT COMPLETE ==========")
