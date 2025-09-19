import os
import json
import yaml

with open("params.yaml") as f:
    p = yaml.safe_load(f)

min_r2 = float(p["min_r2"])
max_rmse = float(p["max_rmse"])

# Use per-target metrics; compute pass rate
metrics_path = "metrics/tuned_metrics.json" if os.path.exists("metrics/tuned_metrics.json") else "metrics/metrics.json"
with open(metrics_path, "r") as f:
    m = json.load(f)

per_target = m.get("per_target")
if isinstance(per_target, list):
    results = per_target
elif isinstance(per_target, dict):
    # from train.py summary dict
    results = [dict(target=k, **v) for k, v in per_target.items()]
else:
    results = []

passes = [ (float(r.get("r2", -1)) >= min_r2) and (float(r.get("rmse", 1e9)) <= max_rmse) for r in results ]
pass_rate = float(sum(passes)) / float(len(results) or 1)

decision = {
    "meets_threshold": bool(pass_rate >= 0.8),
    "pass_rate": pass_rate,
    "min_r2": min_r2,
    "max_rmse": max_rmse,
    "num_targets": len(results),
    "source": os.path.basename(metrics_path),
}
os.makedirs("metrics", exist_ok=True)
with open("metrics/decision.json", "w") as f:
    json.dump(decision, f, indent=2)
print("Decision written:", decision)
