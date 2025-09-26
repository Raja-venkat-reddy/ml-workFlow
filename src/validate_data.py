import os
import sys
import yaml
import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema


def build_schema(feature_cols, target_cols):
    columns = {}
    for col in feature_cols:
        columns[col] = Column(float, nullable=False)
    for col in target_cols:
        columns[col] = Column(float, nullable=False)
    return DataFrameSchema(columns)


def main():
    with open("params.yaml") as f:
        p = yaml.safe_load(f)

    feature_cols = list(p["features"])
    target_cols = list(p["targets"])

    # Use same dataset selection logic as preprocess
    import glob
    default_path = "data/raw/dataset.csv"
    if os.path.exists(default_path):
        dataset_path = default_path
    else:
        csvs = [p for p in glob.glob("data/raw/*.csv") if os.path.getsize(p) > 0]
        if not csvs:
            raise FileNotFoundError("No CSV found in data/raw/. Place a CSV or create data/raw/dataset.csv")
        dataset_path = max(csvs, key=os.path.getmtime)

    df = pd.read_csv(dataset_path)

    required_cols = feature_cols + target_cols
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in dataset: {missing}")

    # Only validate required subset
    df = df[required_cols]
    schema = build_schema(feature_cols, target_cols)
    schema.validate(df, lazy=True)
    # Emit a simple artifact to let DVC track stage completion
    os.makedirs("metrics", exist_ok=True)
    with open("metrics/validation.ok", "w") as f:
        f.write("ok\n")
    print("Data validation passed.")


if __name__ == "__main__":
    main()



