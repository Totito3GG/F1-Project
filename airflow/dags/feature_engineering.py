"""
DAG 3/5 : feature_engineering
Silver -> Gold (features sans fuite de données), puis contrôle qualité de la table Gold.

Démarre quand consume_kafka a publié DS_SILVER. Si le contrôle qualité échoue, le Dataset DS_GOLD n'est pas publié :
l'entraînement ne démarre pas sur des données cassées.
"""
from airflow import DAG

from raceflow_common import DEFAULT_ARGS, DS_GOLD, DS_SILVER, START_DATE, python_task, spark_task

with DAG(
    dag_id="feature_engineering",
    description="Silver -> Gold (Spark) + contrôle qualité",
    start_date=START_DATE,
    schedule=[DS_SILVER],
    catchup=False,
    default_args=DEFAULT_ARGS,
    max_active_runs=1,
    tags=["raceflow", "spark", "gold"],
) as dag:
    silver_to_gold = spark_task("silver_to_gold", "streaming/silver_to_gold.py")
    # outlets sur la dernière tâche : le Dataset n'est publié que si le contrôle réussit
    check_gold = python_task("check_gold", "ml/check_gold.py", outlets=[DS_GOLD])

    silver_to_gold >> check_gold
