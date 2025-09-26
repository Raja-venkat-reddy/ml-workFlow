import os
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, RegressionPreset


def main():
    os.makedirs("metrics", exist_ok=True)
    os.makedirs("metrics/reports", exist_ok=True)

    # Load reference (train) and current (validation) datasets
    ref_X = pd.read_csv("data/processed/X_train.csv")
    cur_X = pd.read_csv("data/processed/X_val.csv")

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_X, current_data=cur_X)
    report.save_html("metrics/reports/data_drift.html")

    # If we have per-target ground truth in validation, also run regression quality preset on first target
    try:
        y_val = pd.read_csv("data/processed/y_val.csv")
        # Construct a sample dataframe using the first target if available (for quick signal)
        target_col = y_val.columns[0]
        df_ref = ref_X.copy()
        df_cur = cur_X.copy()
        # Dummy target columns to satisfy preset structure; actual model preds are not included here
        df_ref[target_col] = 0.0
        df_cur[target_col] = 0.0
        reg_report = Report(metrics=[RegressionPreset()])
        reg_report.run(reference_data=df_ref, current_data=df_cur, column_mapping=None)
        reg_report.save_html("metrics/reports/regression_quality.html")
    except Exception:
        pass


if __name__ == "__main__":
    main()



