"""
Données du dashboard RaceFlow : lecture des fichiers Parquet et des métriques MLflow, calculs des tableaux.

Ce module n'importe pas Streamlit : ses fonctions sont testables avec pandas seul (tests/test_dashboard_data.py).
Le cache Streamlit est ajouté dans dashboard/app.py.

Sources lues :
    data/predictions/latest_race_predictions.parquet  écrit par ml/predict.py   (DAG predict_next_race)
    data/predictions/test_predictions.parquet         écrit par ml/train.py     (DAG train_model, saisons 2024+)
    data/gold/features                                écrit par silver_to_gold  (DAG feature_engineering)
    serveur MLflow, API REST                          métriques du dernier run de l'expérience raceflow-winner
"""
import os
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("RACEFLOW_DATA", ROOT / "data"))
LATEST_PATH = DATA_DIR / "predictions" / "latest_race_predictions.parquet"
TEST_PATH = DATA_DIR / "predictions" / "test_predictions.parquet"
GOLD_PATH = DATA_DIR / "gold" / "features"

# Sur le PC : MLflow de docker-compose sur le port 5001. Dans Docker : http://mlflow:5000 (voir docker-compose.yml).
MLFLOW_URL = os.getenv("MLFLOW_URL", "http://localhost:5001")
EXPERIMENT = "raceflow-winner"


# ------------------------------------------------------------------ lecture des fichiers
def read_parquet(path: Path) -> pd.DataFrame | None:
    """Lit un fichier (ou dossier) Parquet ; renvoie None s'il n'existe pas encore (pipeline pas encore lancé)."""
    if not Path(path).exists():
        return None
    df = pd.read_parquet(path)
    # la colonne de partition « season » revient en type category : on la remet en entier
    for c in ("season", "round", "race_idx"):
        if c in df.columns:
            df[c] = df[c].astype(int)
    return df


def race_label(season: int, rnd: int, name: str) -> str:
    """Nom lisible d'une course, ex. « 2025 · manche 24 · Abu Dhabi Grand Prix »."""
    return f"{season} · manche {rnd} · {name}"


def pretty(identifier) -> str:
    """Identifiant de l'API -> nom affichable : « max_verstappen » -> « Max Verstappen ». Vide ou NaN -> « — »."""
    if not isinstance(identifier, str) or not identifier:
        return "—"
    return identifier.replace("_", " ").title()


# ------------------------------------------------------------------ évaluation course par course
def race_summary(preds: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par course : favori du modèle, poleman, vrai vainqueur, et qui avait raison.

    `preds` doit contenir : race_idx, season, round, race_name, driver_id, grid, is_winner, pred_rank.
    """
    rows = []
    for race_idx, race in preds.groupby("race_idx"):
        first = race.iloc[0]
        favori = race.loc[race["pred_rank"] == 1, "driver_id"]
        pole = race.loc[race["grid"] == 1, "driver_id"]
        winner = race.loc[race["is_winner"] == 1, "driver_id"]
        favori = favori.iloc[0] if len(favori) else None
        pole = pole.iloc[0] if len(pole) else None
        winner = winner.iloc[0] if len(winner) else None
        rows.append({
            "race_idx": int(race_idx),
            "season": int(first["season"]),
            "round": int(first["round"]),
            "race_name": first["race_name"],
            "favori_modele": favori,
            "poleman": pole,
            "vainqueur": winner,
            "modele_juste": int(favori is not None and favori == winner),
            "pole_juste": int(pole is not None and pole == winner),
        })
    return pd.DataFrame(rows).sort_values("race_idx").reset_index(drop=True)


def season_scores(summary: pd.DataFrame) -> pd.DataFrame:
    """Par saison : nombre de courses, % de vainqueurs trouvés par le modèle et par la règle « pole gagne »."""
    out = (summary.groupby("season")
           .agg(courses=("race_idx", "count"), modele=("modele_juste", "mean"), pole=("pole_juste", "mean"))
           .reset_index())
    total = pd.DataFrame([{"season": "Total", "courses": len(summary),
                           "modele": summary["modele_juste"].mean(), "pole": summary["pole_juste"].mean()}])
    out["season"] = out["season"].astype(str)
    return pd.concat([out, total], ignore_index=True)


# ------------------------------------------------------------------ statistiques pilotes (table Gold)
def driver_stats(gold: pd.DataFrame, season: int) -> pd.DataFrame:
    """Statistiques des pilotes sur une saison : courses, victoires, podiums, poles, positions moyennes."""
    s = gold[gold["season"] == season]
    stats = (s.groupby("driver_id")
             .agg(ecurie=("constructor_id", "last"),
                  courses=("race_idx", "nunique"),
                  victoires=("is_winner", "sum"),
                  podiums=("is_podium", "sum"),
                  poles=("grid", lambda g: int((g == 1).sum())),
                  grille_moy=("grid", lambda g: g[g > 0].mean()),          # grid = 0 : départ des stands
                  arrivee_moy=("finish_position", "mean"))
             .reset_index()
             .sort_values(["victoires", "podiums", "arrivee_moy"], ascending=[False, False, True]))
    return stats.reset_index(drop=True)


def driver_history(gold: pd.DataFrame, driver_id: str, season: int) -> pd.DataFrame:
    """Course par course pour un pilote : place sur la grille et place à l'arrivée.

    grid = 0 (départ des stands) devient vide : sinon la courbe l'afficherait au-dessus de la pole.
    """
    d = gold[(gold["season"] == season) & (gold["driver_id"] == driver_id)]
    d = d.sort_values("round")[["round", "race_name", "grid", "finish_position"]].reset_index(drop=True)
    d["grid"] = d["grid"].where(d["grid"] > 0)
    return d


# ------------------------------------------------------------------ MLflow (API REST, sans la bibliothèque mlflow)
def latest_run(url: str = MLFLOW_URL, experiment: str = EXPERIMENT, timeout: float = 3.0) -> dict | None:
    """Métriques et paramètres du dernier run de l'expérience. None si MLflow est injoignable ou vide."""
    try:
        r = requests.get(f"{url}/api/2.0/mlflow/experiments/get-by-name",
                         params={"experiment_name": experiment}, timeout=timeout)
        r.raise_for_status()
        exp_id = r.json()["experiment"]["experiment_id"]
        r = requests.post(f"{url}/api/2.0/mlflow/runs/search",
                          json={"experiment_ids": [exp_id], "max_results": 1,
                                "order_by": ["attributes.start_time DESC"]},
                          timeout=timeout)
        r.raise_for_status()
        runs = r.json().get("runs", [])
    except (requests.RequestException, KeyError, ValueError):
        return None
    if not runs:
        return None
    run = runs[0]
    return {
        "run_id": run["info"]["run_id"],
        "run_name": run["info"].get("run_name", ""),
        "start_time": pd.to_datetime(int(run["info"]["start_time"]), unit="ms"),
        "metrics": {m["key"]: m["value"] for m in run["data"].get("metrics", [])},
        "params": {p["key"]: p["value"] for p in run["data"].get("params", [])},
    }
