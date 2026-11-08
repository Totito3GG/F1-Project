"""
Génère de FAUSSES données de course au format Bronze (comme si Kafka -> Bronze avait déjà tourné).

Sert au test de bout en bout de la CI GitHub Actions : sans appel à l'API Jolpica ni à Kafka, on peut faire tourner
Bronze -> Silver -> Gold -> entraînement -> prédiction à chaque push. Les résultats n'ont AUCUNE valeur sportive.

Lancer :
    python scripts/make_synthetic_data.py --out /tmp/raceflow_data
    puis RACEFLOW_DATA=/tmp/raceflow_data spark-submit streaming/bronze_to_silver.py ...
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DRIVERS = [f"driver_{i:02d}" for i in range(20)]
TEAMS = [f"team_{i}" for i in range(10)]          # 2 pilotes par écurie
POINTS = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}


def lap_time(seconds: float) -> str:
    """90.123 -> '1:30.123' (format de l'API Jolpica)."""
    minutes, rest = divmod(seconds, 60)
    return f"{int(minutes)}:{rest:06.3f}"


def make_races(seasons, rounds: int = 20, seed: int = 0):
    """Retourne (résultats, qualifications) : deux listes de dictionnaires au format des producers."""
    rng = np.random.default_rng(seed)
    skill = rng.normal(0, 1, len(DRIVERS))
    results, quali = [], []
    for season in seasons:
        for rnd in range(1, rounds + 1):
            form = skill + rng.normal(0, 0.4, len(DRIVERS))
            order = np.argsort(-(form + rng.normal(0, 0.6, len(DRIVERS))))   # classement des qualifications
            base = 85 + (rnd % 5) * 3                                        # un "circuit" par manche
            race_score = form + rng.normal(0, 0.9, len(DRIVERS))
            grid_of = {int(d): pos + 1 for pos, d in enumerate(order)}
            finish_order = np.argsort(-race_score)
            for pos, d in enumerate(order, start=1):
                t1 = base + pos * 0.12 + rng.normal(0, 0.05)
                t2 = t1 - 0.4 if pos <= 15 else None
                t3 = t1 - 0.8 if pos <= 10 else None
                quali.append({
                    "season": season, "round": rnd, "driver_id": DRIVERS[d], "constructor_id": TEAMS[d // 2],
                    "quali_position": pos, "q1": lap_time(t1),
                    "q2": lap_time(t2) if t2 else None, "q3": lap_time(t3) if t3 else None,
                    "ingested_at": "2020-01-01T00:00:00Z",
                })
            for place, d in enumerate(finish_order, start=1):
                dnf = rng.random() < 0.06 and place > 3
                lapped = (not dnf) and place > 14
                grid = 0 if rng.random() < 0.02 else grid_of[int(d)]         # départ des stands
                results.append({
                    "season": season, "round": rnd, "race_name": f"Grand Prix {rnd}",
                    "circuit_id": f"circuit_{rnd % 5}",
                    "date": f"{season}-{(rnd - 1) // 2 + 3:02d}-{(rnd % 2) * 14 + 1:02d}",
                    "driver_id": DRIVERS[d], "constructor_id": TEAMS[d // 2], "grid": int(grid),
                    "position": int(place), "points": float(POINTS.get(place, 0)) if not dnf else 0.0,
                    "laps": 55 if not dnf else 20, "status": "Retired" if dnf else ("Lapped" if lapped else "Finished"),
                    "ingested_at": "2020-01-01T00:00:00Z",
                })
    return results, quali


def to_bronze(messages, topic: str) -> pd.DataFrame:
    """Imite la sortie de kafka_to_bronze.py : clé, JSON brut, métadonnées Kafka. Ajoute des doublons."""
    df = pd.DataFrame({
        "key": [f'{m["season"]}-{m["round"]}-{m["driver_id"]}' for m in messages],
        "value": [json.dumps(m) for m in messages],
    })
    dup = df.sample(frac=0.05, random_state=1)                          # le Silver doit les supprimer
    df = pd.concat([df, dup], ignore_index=True)
    df["topic"] = topic
    df["partition"] = 0
    df["offset"] = np.arange(len(df), dtype="int64")
    df["kafka_timestamp"] = pd.date_range("2020-01-01", periods=len(df), freq="s").astype("datetime64[us]")
    df["ingestion_date"] = pd.Timestamp("2020-01-01").date()
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="dossier de données (équivalent de RACEFLOW_DATA)")
    p.add_argument("--first-season", type=int, default=2015)
    p.add_argument("--last-season", type=int, default=2020)
    args = p.parse_args()

    results, quali = make_races(range(args.first_season, args.last_season + 1))
    out = Path(args.out)
    for name, topic, msgs in (("results", "f1.results", results), ("qualifying", "f1.qualifying", quali)):
        folder = out / "bronze" / name
        folder.mkdir(parents=True, exist_ok=True)
        to_bronze(msgs, topic).to_parquet(folder / "part-0.parquet", index=False,
                                          coerce_timestamps="us", allow_truncated_timestamps=True)
        print(f"Bronze {name} : {len(msgs)} messages (+ doublons) -> {folder}")


if __name__ == "__main__":
    main()
