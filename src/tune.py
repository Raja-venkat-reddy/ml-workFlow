import json
import os
import yaml
import pandas as pd
import numpy as np

os.makedirs("metrics", exist_ok=True)

# If Optuna or complex multi-target tuning is not configured, write a passthrough tuned file
HAS_OPTUNA = False
try:
    import optuna  # noqa: F401
    HAS_OPTUNA = True
except Exception:
    HAS_OPTUNA = False

with open("params.yaml") as f:
    params = yaml.safe_load(f)

metrics_in = {}
if os.path.exists("metrics/metrics.json"):
    with open("metrics/metrics.json") as f:
        metrics_in = json.load(f)

per_target = metrics_in.get("per_target") or metrics_in.get("per_target_metrics")
# For this project, we skip heavy tuning for now and just pass metrics through
with open("metrics/tuned_metrics.json", "w") as f:
    json.dump({
        "note": "tuning skipped; passthrough of metrics",
        "has_optuna": HAS_OPTUNA,
        "per_target": per_target if per_target is not None else metrics_in,
    }, f, indent=2)

print("Wrote metrics/tuned_metrics.json (passthrough)")
