from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2025, 9, 18),
}

with DAG(
    dag_id="ml_workflow_pipeline",
    default_args=default_args,
    schedule_interval=None,  # or something like '0 1 * * *' for daily
    catchup=False,
) as dag:

    # Task to pull data via DVC
    pull_data = BashOperator(
        task_id="pull_data",
        bash_command="cd /Users/bunnyreddy/path/to/ml-workFlow && dvc pull"
    )

    # Task to run the training script
    train = BashOperator(
        task_id="train_model",
        bash_command="cd /Users/bunnyreddy/path/to/ml-workFlow && python src/train.py"
    )

    # You can add more tasks: evaluate, push model, etc.
    pull_data >> train