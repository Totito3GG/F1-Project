"""
DAG 2/5 : consume_kafka
Kafka -> Bronze (JSON brut) -> Silver (typé, dédupliqué), pour les résultats et les qualifications.

Démarre tout seul quand ingest_historical a publié le Dataset DS_KAFKA. Spark lit les topics en mode « availableNow »
avec un checkpoint : seuls les nouveaux messages sont traités à chaque exécution.
"""
from airflow import DAG
from airflow.operators.empty import EmptyOperator

from raceflow_common import DEFAULT_ARGS, DS_KAFKA, DS_SILVER, START_DATE, spark_task

with DAG(
    dag_id="consume_kafka",
    description="Kafka -> Bronze -> Silver (Spark)",
    start_date=START_DATE,
    schedule=[DS_KAFKA],
    catchup=False,
    default_args=DEFAULT_ARGS,
    max_active_runs=1,
    tags=["raceflow", "spark", "bronze", "silver"],
) as dag:
    bronze_results = spark_task("kafka_to_bronze_results", "streaming/kafka_to_bronze.py",
                                "f1.results results", kafka=True)
    bronze_qualifying = spark_task("kafka_to_bronze_qualifying", "streaming/kafka_to_bronze.py",
                                   "f1.qualifying qualifying", kafka=True)
    silver_results = spark_task("bronze_to_silver_results", "streaming/bronze_to_silver.py")
    silver_qualifying = spark_task("qualifying_to_silver", "streaming/qualifying_to_silver.py")
    publish = EmptyOperator(task_id="publish_silver_dataset", outlets=[DS_SILVER])

    bronze_results >> silver_results
    bronze_qualifying >> silver_qualifying
    [silver_results, silver_qualifying] >> publish
