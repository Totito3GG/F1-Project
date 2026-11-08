"""
RaceFlow : exporter Prometheus (étape 11).

Toutes les 30 secondes, lit les sources du pipeline, calcule les métriques (monitoring/metrics.py) et les publie sur
http://<hôte>:8000/metrics. Prometheus vient les chercher ; Grafana les affiche.

Sources lues :
    Kafka              offsets de fin des topics f1.results et f1.qualifying
    Bronze (Parquet)   derniers offsets écrits par Spark            -> lag Kafka
    Airflow (API REST) derniers runs des 5 DAGs                     -> durée, succès, âge du dernier succès
    Gold (Parquet)     features des courses courues                 -> drift des données (PSI)
    Prédictions        test_predictions / latest_race_predictions  -> drift de performance, fraîcheur
    MLflow (API REST)  dernier run et dernière version du modèle   -> verdict walk-forward

Chaque source est lue séparément : si l'une est indisponible, les autres continuent d'être publiées et la métrique
raceflow_source_up{source="..."} passe à 0.

Lancer :
    Docker (recommandé) : docker compose up -d exporter   puis http://localhost:8000/metrics
    Sur le PC           : python monitoring/exporter.py
"""
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from prometheus_client import Gauge, start_http_server

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from monitoring.metrics import (  # noqa: E402
    bronze_offsets,
    dag_run_metrics,
    feature_drift,
    kafka_lag,
    recent_accuracy,
)

DATA = Path(os.getenv("RACEFLOW_DATA", Path(__file__).resolve().parents[1] / "data"))
KAFKA = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
AIRFLOW = os.getenv("AIRFLOW_URL", "http://localhost:8081")
AIRFLOW_AUTH = (os.getenv("AIRFLOW_USER", "admin"), os.getenv("AIRFLOW_PASSWORD", "admin"))
MLFLOW = os.getenv("MLFLOW_URL", "http://localhost:5001")
PORT = int(os.getenv("EXPORTER_PORT", "8000"))
INTERVAL = int(os.getenv("EXPORTER_INTERVAL", "30"))

TOPICS = {"f1.results": "results", "f1.qualifying": "qualifying"}   # topic Kafka -> dossier Bronze
DAGS = ["ingest_historical", "consume_kafka", "feature_engineering", "train_model", "predict_next_race"]
EXPERIMENT, MODEL_NAME = "raceflow-winner", "raceflow-winner-xgb"

# ------------------------------------------------------------------ métriques publiées
SOURCE_UP = Gauge("raceflow_source_up", "1 si la source a pu être lue au dernier passage", ["source"])
KAFKA_END = Gauge("raceflow_kafka_messages_total", "Messages dans le topic Kafka (somme des offsets de fin)", ["topic"])
KAFKA_LAG = Gauge("raceflow_kafka_lag", "Messages dans Kafka pas encore écrits dans Bronze", ["topic"])
DAG_DURATION = Gauge("raceflow_dag_last_run_duration_seconds", "Durée du dernier run terminé", ["dag_id"])
DAG_SUCCESS = Gauge("raceflow_dag_last_run_success", "1 = dernier run terminé réussi, 0 = échec", ["dag_id"])
DAG_SINCE = Gauge("raceflow_dag_seconds_since_last_success", "Temps écoulé depuis le dernier succès", ["dag_id"])
DAG_RUNNING = Gauge("raceflow_dag_running", "1 si un run est en cours ou en attente", ["dag_id"])
PSI = Gauge("raceflow_feature_psi", "Drift des données : PSI saisons passées vs saison en cours", ["feature"])
DRIFT_SEASON = Gauge("raceflow_drift_current_season", "Saison comparée aux saisons passées pour le PSI")
RECENT = Gauge("raceflow_recent_winner_accuracy", "% de vainqueurs trouvés sur les 10 dernières courses", ["method"])
RECENT_N = Gauge("raceflow_recent_races", "Nombre de courses dans la fenêtre récente")
WF = Gauge("raceflow_walkforward_metric", "Métriques walk-forward du dernier run MLflow", ["metric"])
MODEL_VERSION = Gauge("raceflow_model_version", "Dernière version du modèle dans le registry MLflow")
PRED_AGE = Gauge("raceflow_prediction_age_seconds", "Âge du fichier de la dernière prédiction")
PRED_UPCOMING = Gauge("raceflow_prediction_is_upcoming", "1 si la dernière prédiction porte sur une course à venir")


def read_parquet(path: Path, columns=None) -> pd.DataFrame | None:
    return pd.read_parquet(path, columns=columns) if path.exists() else None


# ------------------------------------------------------------------ une fonction par source
def collect_kafka() -> None:
    from kafka import KafkaConsumer, TopicPartition

    consumer = KafkaConsumer(bootstrap_servers=KAFKA, request_timeout_ms=10000, api_version_auto_timeout_ms=10000)
    try:
        parts = [TopicPartition(t, p) for t in TOPICS for p in (consumer.partitions_for_topic(t) or [])]
        end = {(tp.topic, tp.partition): off for tp, off in consumer.end_offsets(parts).items()}
    finally:
        consumer.close()
    consumed = {}
    for topic, folder in TOPICS.items():
        bronze = read_parquet(DATA / "bronze" / folder, columns=["topic", "partition", "offset"])
        consumed.update(bronze_offsets(bronze))
    lag = kafka_lag(end, consumed)
    for topic in TOPICS:
        KAFKA_END.labels(topic).set(sum(v for (t, _), v in end.items() if t == topic))
        KAFKA_LAG.labels(topic).set(lag.get(topic, 0))


def collect_airflow() -> None:
    for dag_id in DAGS:
        r = requests.get(f"{AIRFLOW}/api/v1/dags/{dag_id}/dagRuns", auth=AIRFLOW_AUTH, timeout=10,
                         params={"order_by": "-execution_date", "limit": 10})
        r.raise_for_status()
        m = dag_run_metrics(r.json().get("dag_runs", []))
        DAG_DURATION.labels(dag_id).set(m["last_duration"])
        DAG_SUCCESS.labels(dag_id).set(m["last_success"])
        DAG_SINCE.labels(dag_id).set(m["since_success"])
        DAG_RUNNING.labels(dag_id).set(m["running"])


def collect_drift() -> None:
    gold = read_parquet(DATA / "gold" / "features")
    if gold is None:
        raise FileNotFoundError("table Gold absente")
    gold["season"] = gold["season"].astype(int)
    drift, season = feature_drift(gold)
    DRIFT_SEASON.set(season)
    for feature, value in drift.items():
        PSI.labels(feature).set(value)


def collect_predictions() -> None:
    test = read_parquet(DATA / "predictions" / "test_predictions.parquet")
    if test is not None:
        acc = recent_accuracy(test)
        RECENT_N.set(acc["races"])
        RECENT.labels("model").set(acc["model"])
        RECENT.labels("pole").set(acc["pole"])
    latest_path = DATA / "predictions" / "latest_race_predictions.parquet"
    latest = read_parquet(latest_path)
    if latest is None:
        raise FileNotFoundError("aucune prédiction")
    PRED_AGE.set(time.time() - latest_path.stat().st_mtime)
    PRED_UPCOMING.set(int("is_upcoming" in latest.columns and (latest["is_upcoming"] == 1).any()))


def collect_mlflow() -> None:
    r = requests.get(f"{MLFLOW}/api/2.0/mlflow/experiments/get-by-name",
                     params={"experiment_name": EXPERIMENT}, timeout=10)
    r.raise_for_status()
    exp_id = r.json()["experiment"]["experiment_id"]
    r = requests.post(f"{MLFLOW}/api/2.0/mlflow/runs/search", timeout=10,
                      json={"experiment_ids": [exp_id], "max_results": 1, "order_by": ["attributes.start_time DESC"]})
    r.raise_for_status()
    runs = r.json().get("runs", [])
    if runs:
        metrics = {m["key"]: m["value"] for m in runs[0]["data"].get("metrics", [])}
        for key in ("wf_winner_accuracy", "wf_pole_baseline_accuracy", "wf_diff_vs_pole",
                    "wf_diff_ci_low", "wf_diff_ci_high", "wf_n_races"):
            if key in metrics:
                WF.labels(key).set(metrics[key])
    r = requests.get(f"{MLFLOW}/api/2.0/mlflow/registered-models/get", params={"name": MODEL_NAME}, timeout=10)
    if r.ok and "registered_model" in r.json():
        versions = [int(v["version"]) for v in r.json()["registered_model"].get("latest_versions", [])]
        if versions:
            MODEL_VERSION.set(max(versions))


SOURCES = {"kafka": collect_kafka, "airflow": collect_airflow, "gold": collect_drift,
           "predictions": collect_predictions, "mlflow": collect_mlflow}


def collect_all() -> None:
    for name, collect in SOURCES.items():
        try:
            collect()
            SOURCE_UP.labels(name).set(1)
        except Exception as exc:  # une source en panne ne doit pas arrêter les autres
            SOURCE_UP.labels(name).set(0)
            print(f"[{time.strftime('%H:%M:%S')}] source {name} indisponible : {exc}", flush=True)


def main() -> None:
    start_http_server(PORT)
    print(f"Exporter RaceFlow : http://0.0.0.0:{PORT}/metrics (toutes les {INTERVAL} s)", flush=True)
    while True:
        collect_all()
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
