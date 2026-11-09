"""
Calculs du monitoring RaceFlow (étape 11) : lag Kafka, drift des données (PSI), drift de performance, runs Airflow.

Ce module ne dépend ni de Kafka, ni d'Airflow, ni de Prometheus : il reçoit des données déjà lues et renvoie des
nombres. Il est donc testable avec pandas seul (tests/test_monitoring_metrics.py). La lecture des sources et la
publication des métriques sont dans monitoring/exporter.py.
"""
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# Features suivies pour le drift des données : les plus importantes du modèle (étape 7b).
DRIFT_FEATURES = ["grid", "q3_gap_pct", "q1_gap_pct", "drv_avg_pos_last5", "cons_points_last5"]
PSI_WARNING, PSI_ALERT = 0.1, 0.2   # seuils usuels : < 0,1 stable ; 0,1-0,2 à surveiller ; > 0,2 drift important


# ------------------------------------------------------------------ 1. Lag Kafka
def bronze_offsets(bronze: pd.DataFrame) -> dict:
    """Dernier offset écrit dans Bronze, par (topic, partition). Bronze garde les métadonnées Kafka (étape 4)."""
    if bronze is None or bronze.empty:
        return {}
    last = bronze.groupby(["topic", "partition"])["offset"].max()
    return {(t, int(p)): int(o) for (t, p), o in last.items()}


def kafka_lag(end_offsets: dict, consumed: dict) -> dict:
    """Lag par topic = messages présents dans Kafka mais pas encore écrits dans Bronze.

    end_offsets : {(topic, partition): offset du PROCHAIN message} (convention Kafka)
    consumed    : {(topic, partition): offset du DERNIER message écrit dans Bronze}

    Pourquoi pas le lag des « consumer groups » (kafka-exporter) : Spark Structured Streaming ne commit pas ses
    offsets dans Kafka, il les garde dans ses checkpoints. Ce lag-là vaudrait toujours 0 ; on compare donc Kafka
    directement à ce qui est arrivé dans Bronze.
    """
    lag = {}
    for (topic, partition), end in end_offsets.items():
        next_to_read = consumed.get((topic, partition), -1) + 1
        lag[topic] = lag.get(topic, 0) + max(int(end) - next_to_read, 0)
    return lag


# ------------------------------------------------------------------ 2. Drift des données
def psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Population Stability Index entre la distribution de référence et la distribution actuelle.

    Les classes sont les déciles de la référence ; les valeurs vides sont ignorées. 0 = distributions identiques.
    Renvoie NaN si l'une des deux séries n'a pas assez de valeurs.
    """
    ref, cur = reference.dropna().astype(float), current.dropna().astype(float)
    if len(ref) < bins or len(cur) < 5:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:                       # variable presque constante : pas de découpage possible
        return float("nan")
    edges[0], edges[-1] = -np.inf, np.inf
    eps = 1e-4                               # évite log(0) pour une classe vide
    p = np.histogram(ref, edges)[0] / len(ref) + eps
    q = np.histogram(cur, edges)[0] / len(cur) + eps
    return float(np.sum((q - p) * np.log(q / p)))


def feature_drift(gold: pd.DataFrame, features=DRIFT_FEATURES) -> tuple[dict, int]:
    """PSI de chaque feature : saisons passées (ce que le modèle a appris) contre la saison en cours.

    Seulement les courses courues. Renvoie ({feature: psi}, saison en cours).
    """
    done = gold[gold["is_upcoming"] != 1] if "is_upcoming" in gold.columns else gold
    season = int(done["season"].max())
    ref, cur = done[done["season"] < season], done[done["season"] == season]
    return {f: psi(ref[f], cur[f]) for f in features if f in done.columns}, season


# ------------------------------------------------------------------ 3. Drift de performance
def recent_accuracy(preds: pd.DataFrame, n_races: int = 10) -> dict:
    """Sur les n dernières courses prédites : % de vainqueurs trouvés par le modèle et par la règle de la pole.

    preds : data/predictions/test_predictions.parquet (race_idx, grid, is_winner, pred_rank).
    """
    done = preds[preds["is_winner"].notna()]
    last = sorted(done["race_idx"].unique())[-n_races:]
    recent = done[done["race_idx"].isin(last)]
    model = recent[recent["pred_rank"] == 1].groupby("race_idx")["is_winner"].max()
    starters = recent[recent["grid"] > 0]
    pole = starters.loc[starters.groupby("race_idx")["grid"].idxmin()].set_index("race_idx")["is_winner"]
    return {"races": len(last),
            "model": float(model.mean()) if len(model) else float("nan"),
            "pole": float(pole.mean()) if len(pole) else float("nan")}


# ------------------------------------------------------------------ 4. Runs Airflow
def _parse(ts):
    return datetime.fromisoformat(ts.replace("Z", "+00:00")) if ts else None


def dag_run_metrics(runs: list, now: datetime | None = None) -> dict:
    """Résumé des runs d'un DAG (réponse de l'API REST d'Airflow, du plus récent au plus ancien).

    Renvoie : durée du dernier run terminé (s), 1/0 selon son succès, âge du dernier succès (s), 1 si un run tourne.
    """
    now = now or datetime.now(timezone.utc)
    finished = [r for r in runs if r.get("state") in ("success", "failed") and r.get("end_date")]
    out = {"running": int(any(r.get("state") in ("running", "queued") for r in runs)),
           "last_duration": float("nan"), "last_success": float("nan"), "since_success": float("nan")}
    if finished:
        last = finished[0]
        out["last_duration"] = (_parse(last["end_date"]) - _parse(last["start_date"])).total_seconds()
        out["last_success"] = 1.0 if last["state"] == "success" else 0.0
    ok = [r for r in finished if r["state"] == "success"]
    if ok:
        out["since_success"] = (now - _parse(ok[0]["end_date"])).total_seconds()
    return out
