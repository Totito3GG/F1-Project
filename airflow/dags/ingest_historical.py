"""
DAG 1/5 : ingest_historical
Jolpica API -> Kafka (topics f1.results et f1.qualifying).

Deux façons de le lancer :
- automatiquement chaque samedi à 22:00 UTC (étape 10) : après les qualifications, avant la course du dimanche.
  Les qualifications du week-end entrent dans le pipeline, Gold crée les lignes « course à venir », et
  predict_next_race prédit le prochain GP ;
- à la main (« Trigger DAG w/ config ») avec les saisons à charger.
Par défaut : de 2015 à la saison en cours. Les messages en double sont sans conséquence : la couche Silver
déduplique. Quand tout est envoyé, le DAG publie le Dataset DS_KAFKA et lance automatiquement consume_kafka.
"""
from datetime import datetime, timezone

from airflow import DAG
from airflow.models.param import Param
from airflow.operators.empty import EmptyOperator

from raceflow_common import DEFAULT_ARGS, DS_KAFKA, START_DATE, python_task

CURRENT_SEASON = datetime.now(timezone.utc).year

with DAG(
    dag_id="ingest_historical",
    description="API Jolpica -> Kafka (résultats et qualifications)",
    start_date=START_DATE,
    schedule="0 22 * * 6",   # samedi 22:00 UTC : qualifications terminées, course pas encore courue
    catchup=False,
    default_args=DEFAULT_ARGS,
    params={
        "start_year": Param(2015, type="integer", minimum=1950, maximum=2100, description="première saison"),
        "end_year": Param(CURRENT_SEASON, type="integer", minimum=1950, maximum=2100,
                          description="dernière saison (défaut : la saison en cours)"),
    },
    tags=["raceflow", "ingestion", "kafka"],
) as dag:
    years = "--start {{ params.start_year }} --end {{ params.end_year }}"

    create_topics = python_task("create_topics", "ingestion/create_topics.py")
    producer_results = python_task("producer_results", "ingestion/producer_results.py", years)
    producer_qualifying = python_task("producer_qualifying", "ingestion/producer_qualifying.py", years)
    publish = EmptyOperator(task_id="publish_kafka_dataset", outlets=[DS_KAFKA])

    create_topics >> [producer_results, producer_qualifying] >> publish
