import os
import glob
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

with open("params.yaml") as f:
    p = yaml.safe_load(f)

FEATURE_COLUMNS = list(p["features"])  # 18 inputs
TARGET_COLUMNS = list(p["targets"])    # 19 outputs
TEST_SIZE = float(p.get("test_size", 0.2))
VAL_SIZE = float(p.get("val_size", 0.2))  # fraction of the remaining train
RANDOM_STATE = int(p.get("random_state", 42))

os.makedirs("data/processed", exist_ok=True)

# Resolve input dataset: prefer data/raw/dataset.csv, otherwise use most recent CSV in data/raw
default_path = "data/raw/dataset.csv"
if os.path.exists(default_path):
    dataset_path = default_path
else:
    csvs = [p for p in glob.glob("data/raw/*.csv") if os.path.getsize(p) > 0]
    if not csvs:
        raise FileNotFoundError("No CSV found in data/raw/. Place a CSV or create data/raw/dataset.csv")
    dataset_path = max(csvs, key=os.path.getmtime)
print(f"Using dataset: {dataset_path}")

df = pd.read_csv(dataset_path)

# Keep only required columns and drop rows missing any of them
required_cols = FEATURE_COLUMNS + TARGET_COLUMNS
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise KeyError(f"Missing required columns in dataset: {missing}")

df = df[required_cols].dropna().reset_index(drop=True)

X = df[FEATURE_COLUMNS]
y = df[TARGET_COLUMNS]

# train+rest split, then split rest into val/test
X_train, X_rest, y_train, y_rest = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
X_val, X_test, y_val, y_test = train_test_split(
    X_rest, y_rest, test_size=0.5, random_state=RANDOM_STATE
)

X_train.to_csv("data/processed/X_train.csv", index=False)
X_val.to_csv("data/processed/X_val.csv", index=False)
X_test.to_csv("data/processed/X_test.csv", index=False)

# Save multi-target y splits (wide format with 19 columns)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_val.to_csv("data/processed/y_val.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)
print("Preprocessing complete with multi-output splits (18 inputs, 19 targets).")
