"""
DAG 4/5 : train_model
Entraînement XGBoost, validation walk-forward et enregistrement du modèle dans MLflow.

Démarre quand feature_engineering a publié DS_GOLD. Les paramètres, métriques (dont l'écart avec la baseline « le
poleman gagne ») et le modèle sont enregistrés sur le serveur MLflow (http://localhost:5001 depuis le navigateur).
"""
from airflow import DAG

from raceflow_common import DEFAULT_ARGS, DS_GOLD, DS_MODEL, START_DATE, python_task

with DAG(
    dag_id="train_model",
    description="Gold -> XGBoost + MLflow (walk-forward)",
    start_date=START_DATE,
    schedule=[DS_GOLD],
    catchup=False,
    default_args=DEFAULT_ARGS,
    max_active_runs=1,
    tags=["raceflow", "ml", "mlflow"],
) as dag:
    train = python_task("train", "ml/train.py", outlets=[DS_MODEL])
