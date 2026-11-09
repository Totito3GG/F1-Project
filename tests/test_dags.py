"""Tests des DAGs Airflow (étape 8).

Ignorés si Airflow n'est pas installé (cas du PC hors Docker) : ils tournent dans la CI GitHub Actions, où
apache-airflow est installé à la même version que l'image Docker.
"""
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

pytest.importorskip("airflow.models")  # le dossier airflow/ du dépôt masque un Airflow absent

DAGS_DIR = Path(__file__).resolve().parents[1] / "airflow" / "dags"
os.environ.setdefault("AIRFLOW__CORE__DAGS_FOLDER", str(DAGS_DIR))
os.environ.setdefault("AIRFLOW__CORE__LOAD_EXAMPLES", "false")
sys.path.insert(0, str(DAGS_DIR))

from airflow.models import DagBag  # noqa: E402

# DAG -> (Dataset qui le déclenche, Dataset qu'il publie)
CHAIN = {
    "ingest_historical": (None, "raceflow://kafka/f1"),
    "consume_kafka": ("raceflow://kafka/f1", "raceflow://lake/silver"),
    "feature_engineering": ("raceflow://lake/silver", "raceflow://lake/gold"),
    "train_model": ("raceflow://lake/gold", "raceflow://mlflow/raceflow-winner-xgb"),
    "predict_next_race": ("raceflow://mlflow/raceflow-winner-xgb", None),
}


@pytest.fixture(scope="module")
def dagbag():
    return DagBag(dag_folder=str(DAGS_DIR), include_examples=False)


def triggers(dag):
    cond = getattr(dag.timetable, "dataset_condition", None)
    return [ds.uri for _, ds in cond.iter_datasets()] if cond is not None else []


def publishes(dag):
    return sorted({o.uri for t in dag.tasks for o in t.outlets})


def test_no_import_errors(dagbag):
    assert dagbag.import_errors == {}


def test_the_five_dags_exist(dagbag):
    assert set(dagbag.dags) == set(CHAIN)


@pytest.mark.parametrize("dag_id", sorted(CHAIN))
def test_dags_are_chained_by_datasets(dagbag, dag_id):
    trigger, publish = CHAIN[dag_id]
    dag = dagbag.dags[dag_id]
    assert triggers(dag) == ([trigger] if trigger else [])
    assert publishes(dag) == ([publish] if publish else [])


def test_ingest_params_and_order(dagbag):
    dag = dagbag.dags["ingest_historical"]
    assert dag.params["start_year"] == 2015
    assert dag.params["end_year"] == datetime.now(timezone.utc).year     # étape 10 : jusqu'à la saison en cours
    assert dag.get_task("producer_results").upstream_task_ids == {"create_topics"}
    assert dag.get_task("producer_qualifying").upstream_task_ids == {"create_topics"}
    assert dag.get_task("publish_kafka_dataset").upstream_task_ids == {"producer_results", "producer_qualifying"}


def test_gold_is_published_only_after_quality_check(dagbag):
    dag = dagbag.dags["feature_engineering"]
    assert dag.get_task("check_gold").upstream_task_ids == {"silver_to_gold"}
    assert [o.uri for o in dag.get_task("check_gold").outlets] == ["raceflow://lake/gold"]
    assert dag.get_task("silver_to_gold").outlets == []


def test_commands_use_project_environment(dagbag):
    for dag in dagbag.dags.values():
        for task in dag.tasks:
            cmd = getattr(task, "bash_command", None)
            if cmd is None:
                continue
            name = f"{dag.dag_id}.{task.task_id}"
            assert "/opt/raceflow-venv/bin" in cmd, f"{name} n'utilise pas l'environnement du projet"
            assert cmd.startswith("cd /opt/raceflow"), name


def test_kafka_jobs_load_the_spark_kafka_connector(dagbag):
    dag = dagbag.dags["consume_kafka"]
    for task_id in ("kafka_to_bronze_results", "kafka_to_bronze_qualifying"):
        assert "spark-sql-kafka-0-10_2.12:3.5.1" in dag.get_task(task_id).bash_command
    assert "--packages" not in dag.get_task("bronze_to_silver_results").bash_command


def test_dags_have_tags_and_no_catchup(dagbag):
    for dag in dagbag.dags.values():
        assert "raceflow" in dag.tags
        assert dag.catchup is False


def test_ingest_runs_every_saturday_evening(dagbag):
    """Étape 10 : après les qualifications du samedi, avant la course du dimanche."""
    assert dagbag.dags["ingest_historical"].schedule_interval == "0 22 * * 6"
