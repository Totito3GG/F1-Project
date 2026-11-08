"""
Prédiction avec le modèle enregistré dans le registry MLflow (utilisé par le DAG predict_next_race).

Le script prend une course de la table Gold (par défaut : la plus récente), calcule pour chaque pilote la
probabilité de victoire et écrit le classement prédit dans data/predictions/latest_race_predictions.parquet,
lu ensuite par le dashboard (étape 9).

Limite actuelle : la table Gold ne contient que des courses déjà courues. Prédire une course à venir demande de
construire ses features à partir de la grille réelle (qualifications du samedi) : prévu pour la suite.

Lancer :
    python ml/predict.py
    python ml/predict.py --race 202405 --model-version 3
"""
import argparse
import os
import sys
from pathlib import Path

import mlflow
import mlflow.xgboost
import pandas as pd
from dotenv import load_dotenv
from mlflow import MlflowClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml.features import GOLD_PATH, ID_COLS, add_race_rank, load_gold  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data" / "predictions" / "latest_race_predictions.parquet"
MODEL_NAME = "raceflow-winner-xgb"

load_dotenv()
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")


def parse_args():
    p = argparse.ArgumentParser(description="RaceFlow - prédiction d'une course")
    p.add_argument("--gold", default=str(GOLD_PATH))
    p.add_argument("--race", default="latest", help="'latest' ou un race_idx (ex. 202405 = saison 2024, manche 5)")
    p.add_argument("--model-version", type=int, default=None, help="version du registry (défaut : la dernière)")
    return p.parse_args()


def pick_race(df: pd.DataFrame, race: str) -> int:
    if race == "latest":
        return int(df["race_idx"].max())
    idx = int(race)
    if idx not in set(df["race_idx"]):
        raise SystemExit(f"Course {idx} absente de la table Gold")
    return idx


def latest_version(client: MlflowClient) -> int:
    versions = [int(v.version) for v in client.search_model_versions(f"name='{MODEL_NAME}'")]
    if not versions:
        raise SystemExit(f"Aucune version de '{MODEL_NAME}' dans MLflow : lancer d'abord ml/train.py")
    return max(versions)


def main() -> None:
    args = parse_args()
    mlflow.set_tracking_uri(TRACKING_URI)
    version = args.model_version or latest_version(MlflowClient())
    model = mlflow.xgboost.load_model(f"models:/{MODEL_NAME}/{version}")
    features = list(model.get_booster().feature_names)

    df = load_gold(args.gold)
    race_idx = pick_race(df, args.race)
    race = df[df["race_idx"] == race_idx].copy()
    missing = [c for c in features if c not in race.columns]
    if missing:
        raise SystemExit(f"Colonnes absentes de Gold pour ce modèle : {missing} (relancer silver_to_gold)")

    race["proba"] = model.predict_proba(race[features])[:, 1]
    cols = [c for c in ID_COLS if c in race.columns]
    extra = [c for c in ("grid", "is_winner", "finish_position") if c in race.columns]
    out = add_race_rank(race)[cols + extra + ["proba", "win_prob", "pred_rank"]]
    out = out.sort_values("pred_rank")
    out["model_version"] = version
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)

    head = out.iloc[0]
    print(f"Course {head['race_name']} ({head['season']}, manche {head['round']}) | modèle v{version}")
    print(out[["pred_rank", "driver_id", "grid", "win_prob"]].head(5).round(3).to_string(index=False))
    print(f"Prédictions -> {OUT_PATH.relative_to(ROOT)}")
    print("Predict OK")


if __name__ == "__main__":
    main()
