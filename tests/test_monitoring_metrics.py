"""Tests des calculs du monitoring (étape 11) : pandas seul, sans Kafka, Airflow ni Prometheus."""
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from monitoring.metrics import bronze_offsets, dag_run_metrics, feature_drift, kafka_lag, psi, recent_accuracy


def test_kafka_lag_compares_kafka_with_bronze():
    bronze = pd.DataFrame({"topic": ["f1.results"] * 3 + ["f1.qualifying"],
                           "partition": [0, 0, 1, 0], "offset": [0, 9, 4, 99]})
    consumed = bronze_offsets(bronze)
    assert consumed == {("f1.results", 0): 9, ("f1.results", 1): 4, ("f1.qualifying", 0): 99}
    # Kafka : partition 0 a 10 messages (0..9), partition 1 en a 8 (0..7) -> 3 pas encore dans Bronze
    end = {("f1.results", 0): 10, ("f1.results", 1): 8, ("f1.qualifying", 0): 100}
    assert kafka_lag(end, consumed) == {"f1.results": 3, "f1.qualifying": 0}


def test_kafka_lag_counts_everything_when_bronze_is_empty():
    assert kafka_lag({("f1.results", 0): 50, ("f1.results", 1): 20}, bronze_offsets(pd.DataFrame())) == {
        "f1.results": 70}


def test_psi_zero_for_same_distribution_and_high_for_a_shift():
    rng = np.random.default_rng(0)
    ref = pd.Series(rng.normal(0, 1, 5000))
    assert psi(ref, pd.Series(rng.normal(0, 1, 5000))) < 0.02
    assert psi(ref, pd.Series(rng.normal(1.5, 1, 5000))) > 0.2


def test_psi_ignores_missing_values_and_needs_enough_data():
    ref = pd.Series(list(range(100)) + [None] * 50)
    assert psi(ref, pd.Series(range(100))) < 0.02
    assert np.isnan(psi(ref, pd.Series([1.0, None])))


def test_feature_drift_compares_current_season_with_the_past():
    rng = np.random.default_rng(1)
    past = pd.DataFrame({"season": 2025, "grid": rng.integers(1, 21, 400), "q3_gap_pct": rng.normal(0.5, 0.2, 400)})
    now = pd.DataFrame({"season": 2026, "grid": rng.integers(1, 21, 200), "q3_gap_pct": rng.normal(1.5, 0.2, 200)})
    upcoming = pd.DataFrame({"season": 2027, "grid": [1], "q3_gap_pct": [9.9], "is_upcoming": [1]})
    drift, season = feature_drift(pd.concat([past.assign(is_upcoming=0), now.assign(is_upcoming=0), upcoming]),
                                  features=["grid", "q3_gap_pct"])
    assert season == 2026                    # la course à venir (2027) est ignorée
    assert drift["grid"] < 0.1 and drift["q3_gap_pct"] > 0.2


def test_recent_accuracy_model_vs_pole_on_last_races():
    # pilote a : favori du modèle (grille 2) ; pilote b : poleman ; c : 3e
    rows = []
    for race, winner in enumerate(["a", "b", "c", "a"]):
        for driver, grid, rank in (("a", 2, 1), ("b", 1, 2), ("c", 3, 3)):
            rows.append({"race_idx": 202401 + race, "driver_id": driver, "grid": grid, "pred_rank": rank,
                         "is_winner": int(driver == winner)})
    preds = pd.DataFrame(rows)
    assert recent_accuracy(preds, n_races=4) == {"races": 4, "model": 0.5, "pole": 0.25}
    assert recent_accuracy(preds, n_races=2) == {"races": 2, "model": 0.5, "pole": 0.0}   # 2 dernières courses


def test_dag_run_metrics():
    now = datetime(2026, 11, 8, 22, 0, tzinfo=timezone.utc)
    runs = [
        {"state": "running", "start_date": "2026-11-08T21:59:00+00:00", "end_date": None},
        {"state": "failed", "start_date": "2026-11-08T20:00:00+00:00", "end_date": "2026-11-08T20:05:00+00:00"},
        {"state": "success", "start_date": "2026-11-08T19:00:00+00:00", "end_date": "2026-11-08T19:02:30+00:00"},
    ]
    m = dag_run_metrics(runs, now=now)
    assert m["running"] == 1
    assert m["last_duration"] == 300.0 and m["last_success"] == 0.0          # dernier run terminé : échec, 5 min
    assert m["since_success"] == pytest.approx(2 * 3600 + 57 * 60 + 30)       # dernier succès à 19:02:30


def test_dag_run_metrics_without_runs():
    m = dag_run_metrics([])
    assert m["running"] == 0 and np.isnan(m["last_duration"]) and np.isnan(m["since_success"])
