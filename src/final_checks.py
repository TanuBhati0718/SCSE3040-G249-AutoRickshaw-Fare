import pandas as pd
import numpy as np

df = pd.read_csv("data/raw/train.csv")
df["date"] = pd.to_datetime(df["date"])

numeric = [
    "distance_km", "wait_min", "passengers",
    "hour", "luggage_bags", "fuel_price_rs"
]

print("\n========== A. UNIT MIXUP: EXCLUDING 9999 ==========")
for col in numeric:
    clean = df[df[col] != 9999]
    print("\n", col)
    print(clean.groupby("zone")[col].agg(
        ["count", "min", "median", "mean", "max"]
    ).round(2))

print("\n========== B. STUCK SENSOR: CONSECUTIVE ROWS ==========")
sorted_df = df.sort_values("date").reset_index(drop=True)

for col in numeric:
    values = sorted_df[col]
    change = values.ne(values.shift()).cumsum()
    runs = sorted_df.groupby(change).agg(
        value=(col, "first"),
        count=(col, "size"),
        start=("date", "min"),
        end=("date", "max")
    )
    runs = runs[runs["count"] >= 5]
    print("\n", col)
    if len(runs):
        print(runs.sort_values("count", ascending=False).head(5).to_string())
    else:
        print("No repeated consecutive values of 5 or more rows")

print("\n========== C. TARGET TYPO CANDIDATES ==========")
print("All fares above 1000:")
print(df.loc[df["fare_rs"] > 1000,
             ["record_id", "date", "fare_rs"]].to_string(index=False))

print("\nCounts of possible 10x values:")
for _, row in df[df["fare_rs"] > 1000].iterrows():
    expected = row["fare_rs"] / 10
    count = np.isclose(df["fare_rs"], expected, atol=0.01).sum()
    print(row["record_id"], row["fare_rs"],
          "divided by 10:", expected,
          "matching normal values:", count)

print("\n========== D. SPELLING VARIANTS ==========")
for col in df.select_dtypes(include="object").columns:
    print("\nColumn:", col)
    print("Exact raw values:")
    for value, count in df[col].value_counts(dropna=False).items():
        print(repr(value), ":", count)

print("\n========== VERIFICATION COMPLETE ==========")
