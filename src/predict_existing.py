import os
import glob
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

os.makedirs("metrics", exist_ok=True)
X_val = pd.read_csv("data/processed/X_val.csv")
y_val_wide = pd.read_csv("data/processed/y_val.csv")

rows = []
preds_dir = "metrics/preds_val"
os.makedirs(preds_dir, exist_ok=True)

for path in glob.glob("models/*.joblib"):
    name = os.path.splitext(os.path.basename(path))[0]
    target = name  # models are saved as <target>.joblib
    try:
        if target not in y_val_wide.columns:
            continue
        model = joblib.load(path)
        yhat = model.predict(X_val)
        pd.DataFrame({"pred": yhat}).to_csv(f"{preds_dir}/{name}.csv", index=False)

        y_true = y_val_wide[target]
        r2 = float(r2_score(y_true, yhat))
        rmse = float(np.sqrt(mean_squared_error(y_true, yhat)))
        mae = float(mean_absolute_error(y_true, yhat))
        rows.append({"target": target, "model": name, "r2": r2, "rmse": rmse, "mae": mae})
    except Exception:
        rows.append({"target": target, "model": name, "r2": -1e9, "rmse": 1e9, "mae": 1e9})

# Leaderboard based on validation set
lb = pd.DataFrame(rows).sort_values(by=["target", "r2", "rmse"], ascending=[True, False, True])
lb.to_csv("metrics/leaderboard.csv", index=False)

summary = lb.groupby("target").first().reset_index()
with open("metrics/metrics.json", "w") as f:
    json.dump({"per_target": summary.to_dict(orient="records")}, f, indent=2)
print("Validation predictions complete for all targets.")
