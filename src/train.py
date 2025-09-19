import os
import json
import yaml
import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

os.makedirs("metrics", exist_ok=True)
os.makedirs("models", exist_ok=True)

with open("params.yaml") as f:
    params = yaml.safe_load(f)

X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")
y_train_wide = pd.read_csv("data/processed/y_train.csv")
y_test_wide = pd.read_csv("data/processed/y_test.csv")

target_columns = list(params["targets"])  # 19 targets

def make_models(cfg, seed):
    models = {
        "LinearRegression": LinearRegression(**cfg.get("LinearRegression", {})),
        "Ridge": Ridge(**cfg.get("Ridge", {})),
        "Lasso": Lasso(**cfg.get("Lasso", {})),
        "ElasticNet": ElasticNet(**cfg.get("ElasticNet", {})),
        "SVR": SVR(**cfg.get("SVR", {})),
        "RandomForestRegressor": RandomForestRegressor(random_state=seed, **cfg.get("RandomForestRegressor", {})),
        "ExtraTreesRegressor": ExtraTreesRegressor(random_state=seed, **cfg.get("ExtraTreesRegressor", {})),
        "GradientBoostingRegressor": GradientBoostingRegressor(random_state=seed, **cfg.get("GradientBoostingRegressor", {})),
        "KNeighborsRegressor": KNeighborsRegressor(**cfg.get("KNeighborsRegressor", {})),
    }
    if HAS_XGB and "XGBRegressor" in cfg:
        models["XGBRegressor"] = XGBRegressor(random_state=seed, n_jobs=-1, **cfg.get("XGBRegressor", {}))
    return models

base_models = make_models(params["models"], params["random_state"])

def train_one_target(target_name: str):
    y_tr = y_train_wide[target_name]
    y_te = y_test_wide[target_name]

    rows_local = []
    best_model = None
    best_row = None

    for mdl_name, mdl in base_models.items():
        model = joblib.clone(mdl) if hasattr(joblib, "clone") else mdl.__class__(**getattr(mdl, "get_params", lambda: {})())
        model.fit(X_train, y_tr)
        preds = model.predict(X_test)
        r2 = float(r2_score(y_te, preds))
        rmse = float(np.sqrt(mean_squared_error(y_te, preds)))
        mae = float(mean_absolute_error(y_te, preds))
        row = {"target": target_name, "model": mdl_name, "r2": r2, "rmse": rmse, "mae": mae}
        rows_local.append(row)

    # select best by r2 desc, rmse asc
    df_local = pd.DataFrame(rows_local).sort_values(by=["r2", "rmse"], ascending=[False, True])
    best_row = df_local.iloc[0].to_dict()
    # retrain best on full train for this target
    best_name = best_row["model"]
    best_model = base_models[best_name]
    best_model.fit(X_train, y_tr)
    joblib.dump(best_model, f"models/{target_name}.joblib")
    return df_local, best_row

results = Parallel(n_jobs=-1, prefer="processes")(delayed(train_one_target)(t) for t in target_columns)

# Aggregate
leaderboards = pd.concat([df for df, _ in results], ignore_index=True)
leaderboards.to_csv("metrics/leaderboard.csv", index=False)

summary = {r["target"]: {"model": r["model"], "r2": r["r2"], "rmse": r["rmse"], "mae": r["mae"]} for _, r in results}
with open("metrics/metrics.json", "w") as f:
    json.dump({"per_target": summary,
               "avg_r2": float(np.mean([v["r2"] for v in summary.values()])),
               "avg_rmse": float(np.mean([v["rmse"] for v in summary.values()])),
               "num_targets": len(summary)}, f, indent=2)

print("Saved per-target models in models/, metrics/leaderboard.csv and metrics/metrics.json")
