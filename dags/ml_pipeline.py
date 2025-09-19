from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator
from datetime import datetime
import json

REPO="/Users/bunnyreddy/automation/ml-workFlow"

def decide_next():
    with open(f"{REPO}/metrics/decision.json") as f:
        d = json.load(f)
    return "finish" if d["meets_threshold"] else "tune"

with DAG(
    dag_id="ml_pipeline_local",
    start_date=datetime(2025, 9, 1),
    schedule_interval=None,
    catchup=False,
    tags=["ml","dvc"]
) as dag:

    dvc_pull = BashOperator(
        task_id="dvc_pull",
        bash_command=f"cd {REPO} && dvc pull || true"
    )

    repro_preprocess = BashOperator(
        task_id="repro_preprocess",
        bash_command=f"cd {REPO} && dvc repro preprocess"
    )

    repro_train = BashOperator(
        task_id="repro_train",
        bash_command=f"cd {REPO} && dvc repro train"
    )

    repro_predict = BashOperator(
        task_id="repro_predict",
        bash_command=f"cd {REPO} && dvc repro predict_existing"
    )

    repro_evaluate = BashOperator(
        task_id="repro_evaluate",
        bash_command=f"cd {REPO} && dvc repro evaluate"
    )

    branch = BranchPythonOperator(
        task_id="branch",
        python_callable=decide_next
    )

    tune = BashOperator(
        task_id="tune",
        bash_command=f"cd {REPO} && dvc repro tune"
    )

    re_evaluate = BashOperator(
        task_id="re_evaluate",
        bash_command=f"cd {REPO} && dvc repro evaluate"
    )

    finish = BashOperator(
        task_id="finish",
        bash_command="echo 'Done'"
    )

    dvc_push = BashOperator(
        task_id="dvc_push",
        bash_command=f"cd {REPO} && dvc add models && dvc push || true"
    )

    dvc_pull >> repro_preprocess >> repro_train >> repro_predict >> repro_evaluate >> branch
    branch >> [finish, tune]
    tune >> re_evaluate >> finish
    finish >> dvc_push
