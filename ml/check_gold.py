"""
Contrôle qualité de la table Gold avant l'entraînement (utilisé par le DAG feature_engineering et la CI).

Échoue (code de sortie 1) si la table est vide, s'il manque des colonnes, si les courses n'ont pas un seul vainqueur,
ou si une feature est entièrement vide. Mieux vaut arrêter le pipeline ici que d'entraîner sur des données cassées.

Lancer :
    python ml/check_gold.py
    python ml/check_gold.py --gold data/gold/features
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml.features import BASE_FEATURES, GOLD_PATH, ID_COLS, TARGET, load_gold  # noqa: E402

MIN_SINGLE_WINNER_RATIO = 0.98   # part minimale des courses avec exactement un vainqueur


def check_gold(df: pd.DataFrame) -> list:
    """Retourne la liste des problèmes détectés (liste vide = table saine)."""
    problems = []
    if df.empty:
        return ["la table Gold est vide"]

    missing = [c for c in ID_COLS + BASE_FEATURES + [TARGET] if c not in df.columns]
    if missing:
        problems.append(f"colonnes manquantes : {missing}")
        return problems

    winners = df.groupby("race_idx")[TARGET].sum()
    ratio = float((winners == 1).mean())
    if ratio < MIN_SINGLE_WINNER_RATIO:
        problems.append(f"seulement {ratio:.1%} des courses ont exactement un vainqueur "
                        f"(min {MIN_SINGLE_WINNER_RATIO:.0%})")

    if df.duplicated(["race_idx", "driver_id"]).any():
        problems.append("doublons (course, pilote) dans la table Gold")

    empty = [c for c in BASE_FEATURES if df[c].isna().all()]
    if empty:
        problems.append(f"features entièrement vides : {empty}")
    return problems


def main() -> None:
    p = argparse.ArgumentParser(description="RaceFlow - contrôle qualité Gold")
    p.add_argument("--gold", default=str(GOLD_PATH))
    args = p.parse_args()

    df = load_gold(args.gold)
    problems = check_gold(df)
    print(f"Gold : {len(df)} lignes, {df['race_idx'].nunique()} courses, saisons {df.season.min()}-{df.season.max()}")
    if problems:
        for pb in problems:
            print(f"ERREUR : {pb}")
        sys.exit(1)
    print("Gold sain")


if __name__ == "__main__":
    main()
