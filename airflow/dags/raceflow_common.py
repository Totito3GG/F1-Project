"""
Éléments communs aux DAGs RaceFlow : chemins dans le conteneur, Datasets, fabrique de tâches.

Les 5 DAGs s'enchaînent tout seuls grâce aux Datasets Airflow : quand un DAG termine avec succès, il « publie » un
Dataset, et le DAG suivant, planifié sur ce Dataset, démarre.

    ingest_historical -> consume_kafka -> feature_engineering -> train_model -> predict_next_race
       (KAFKA)             (SILVER)            (GOLD)              (MODEL)

Les scripts du projet tournent dans un environnement Python séparé (/opt/raceflow-venv) pour ne pas mélanger les
dépendances d'Airflow avec celles de Spark, XGBoost et MLflow. Les variables d'environnement (Kafka, MLflow,
dossier de données) sont fournies par le conteneur (voir docker-compose.yml).
"""
from datetime import datetime, timedelta

from airflow.datasets import Dataset
from airflow.operators.bash import BashOperator

HOME = "/opt/raceflow"
VENV_BIN = "/opt/raceflow-venv/bin"
KAFKA_PACKAGE = "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1"

# Un Dataset = « cette étape a produit de nouvelles données ». Le nom n'est qu'un identifiant.
DS_KAFKA = Dataset("raceflow://kafka/f1")
DS_SILVER = Dataset("raceflow://lake/silver")
DS_GOLD = Dataset("raceflow://lake/gold")
DS_MODEL = Dataset("raceflow://mlflow/raceflow-winner-xgb")

START_DATE = datetime(2025, 1, 1)
DEFAULT_ARGS = {
    "owner": "raceflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def python_task(task_id: str, script: str, args: str = "", **kwargs) -> BashOperator:
    """Lance un script Python du projet avec l'environnement /opt/raceflow-venv."""
    return BashOperator(
        task_id=task_id,
        bash_command=f"cd {HOME} && {VENV_BIN}/python {script} {args}".rstrip(),
        execution_timeout=kwargs.pop("execution_timeout", timedelta(hours=1)),
        **kwargs,
    )


def spark_task(task_id: str, script: str, args: str = "", kafka: bool = False, **kwargs) -> BashOperator:
    """Lance un job Spark en mode local. `kafka=True` ajoute le connecteur Kafka de Spark.

    Le dossier de l'environnement du projet passe en tête du PATH : spark-submit y trouve pyspark.
    """
    packages = f"--packages {KAFKA_PACKAGE} " if kafka else ""
    command = (
        f"cd {HOME} && export PATH={VENV_BIN}:$PATH && "
        f"spark-submit {packages}--conf spark.jars.ivy=/tmp/.ivy2 --master local[2] --driver-memory 2g "
        f"{script} {args}"
    ).rstrip()
    return BashOperator(
        task_id=task_id,
        bash_command=command,
        execution_timeout=kwargs.pop("execution_timeout", timedelta(hours=1)),
        **kwargs,
    )
