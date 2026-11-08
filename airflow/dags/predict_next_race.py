"""
DAG 5/5 : predict_next_race
Charge le modèle enregistré dans MLflow et écrit le classement prédit (probabilité de victoire par pilote) dans
data/predictions/latest_race_predictions.parquet, lu par le dashboard.

Démarre quand train_model a publié DS_MODEL. Limite actuelle : la prédiction porte sur la course la plus récente de la
table Gold. Prédire une course à venir demande de construire ses features à partir de la grille réelle : prévu ensuite.
"""
from airflow import DAG

from raceflow_common import DEFAULT_ARGS, DS_MODEL, START_DATE, python_task

with DAG(
    dag_id="predict_next_race",
    description="Modèle MLflow -> prédictions de la dernière course",
    start_date=START_DATE,
    schedule=[DS_MODEL],
    catchup=False,
    default_args=DEFAULT_ARGS,
    max_active_runs=1,
    tags=["raceflow", "ml", "predict"],
) as dag:
    predict = python_task("predict", "ml/predict.py")
