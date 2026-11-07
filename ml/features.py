"""
Charge data/gold/features et prepare X (features) et y (vainqueur oui/non).
Contient aussi le split temporel, la validation walk-forward et les metriques "par course"
utilisees par train.py. Pas de dependance a Spark ni a MLflow : testable avec pandas seul.
"""
from pathlib import Path

import numpy as np
import pandas as pd

GOLD_PATH = Path(__file__).resolve().parents[1] / "data" / "gold" / "features"

ID_COLS = ["season", "round", "race_idx", "race_name", "circuit_id",
           "race_date", "driver_id", "constructor_id"]

BASE_FEATURES = [
    "grid", "pit_lane_start",
    "drv_avg_pos_last5", "drv_wins_last5", "drv_podiums_last5",
    "drv_finish_rate_last10", "drv_points_season", "drv_circuit_avg_pos",
    "cons_avg_pos_last5", "cons_points_last5",
]
# Etape 7b : temps de qualification (connus avant la course, donc sans fuite).
#   q1_gap_pct / q3_gap_pct : ecart au meilleur temps de la session, en %
#   quali_grid_delta        : place sur la grille - place en qualifications (penalites)
QUALI_FEATURES = ["q1_gap_pct", "q3_gap_pct", "quali_grid_delta"]
# Etape 7c : features "relatives" (toutes connues avant la course, fenetres sans la course courante)
#   q2_gap_pct            : ecart au meilleur temps de Q2
#   quali_gap_vs_teammate : ecart Q1 au coequipier (< 0 = plus rapide que lui) -> mesure le pilote
#   cons_quali_gap_last5  : rythme moyen de l'ecurie en qualification sur les 5 courses precedentes
#   drv_points_share / cons_points_share : points / meilleur de la course (comparable entre saisons)
EXTRA_FEATURES = ["q2_gap_pct", "quali_gap_vs_teammate", "cons_quali_gap_last5",
                  "drv_points_share", "cons_points_share"]
FEATURES = BASE_FEATURES  # compatibilite : jeu de base


def available_features(df: pd.DataFrame, use_quali: bool = True, use_extra: bool = True) -> list:
    """Features de base, + qualifications, + features 7c : seulement celles presentes dans le Gold."""
    feats = list(BASE_FEATURES)
    if use_quali:
        feats += [c for c in QUALI_FEATURES if c in df.columns]
    if use_extra:
        feats += [c for c in EXTRA_FEATURES if c in df.columns]
    return feats

TARGET = "is_winner"
PODIUM = "is_podium"
# Colonnes connues seulement APRES la course : jamais utilisees comme features.
LABEL_COLS = [TARGET, PODIUM, "finish_position"]


def load_gold(path=GOLD_PATH) -> pd.DataFrame:
    """Lit la table Gold (partitionnee par season) et remet les bons types."""
    df = pd.read_parquet(path)
    # la colonne de partition revient en "category" : on la remet en int
    df["season"] = df["season"].astype(int)
    df["race_idx"] = df["race_idx"].astype(int)
    cols = BASE_FEATURES + [c for c in QUALI_FEATURES + EXTRA_FEATURES if c in df.columns]
    df[cols] = df[cols].astype(float)  # NaN geres nativement par XGBoost
    df[TARGET] = df[TARGET].astype(int)
    if PODIUM in df.columns:
        df[PODIUM] = df[PODIUM].astype(int)
    return df.sort_values(["race_idx", "grid"]).reset_index(drop=True)


def temporal_split(df: pd.DataFrame, test_from: int = 2024):
    """Entrainement = saisons < test_from, test = saisons >= test_from. Jamais de split aleatoire."""
    train = df[df["season"] < test_from]
    test = df[df["season"] >= test_from]
    if train.empty or test.empty:
        raise ValueError(f"Split vide : verifier les saisons presentes autour de {test_from}")
    return train, test


def walk_forward(df: pd.DataFrame, fit_predict, first_test: int = 2018) -> pd.DataFrame:
    """Pour chaque saison T >= first_test : entraine sur les saisons < T, predit T.

    `fit_predict(train, test)` doit renvoyer un score par ligne de `test`.
    Retourne les lignes de test de toutes les saisons, avec la colonne `proba`.
    Plus fiable qu'un seul split : ~170 courses evaluees au lieu de ~50.
    """
    parts = []
    for season in sorted(df["season"].unique()):
        if season < first_test:
            continue
        train = df[df["season"] < season]
        test = df[df["season"] == season].copy()
        test["proba"] = fit_predict(train, test)
        parts.append(test)
    if not parts:
        raise ValueError(f"Aucune saison >= {first_test}")
    return pd.concat(parts)


# ---------------------------------------------------------------- metriques par course
def winner_hits(df: pd.DataFrame, proba_col: str = "proba") -> pd.Series:
    """0/1 par course : le favori du modele (plus forte probabilite) a-t-il gagne ?"""
    fav = df.loc[df.groupby("race_idx")[proba_col].idxmax()]
    return fav.set_index("race_idx")[TARGET].astype(float)


def pole_hits(df: pd.DataFrame) -> pd.Series:
    """0/1 par course : le poleman (plus petite grille > 0) a-t-il gagne ?"""
    starters = df[df["grid"] > 0]
    pole = starters.loc[starters.groupby("race_idx")["grid"].idxmin()]
    return pole.set_index("race_idx")[TARGET].astype(float)


def winner_accuracy(df: pd.DataFrame, proba_col: str = "proba") -> float:
    """% de courses ou le pilote avec la plus forte probabilite a reellement gagne."""
    return float(winner_hits(df, proba_col).mean())


def pole_baseline_accuracy(df: pd.DataFrame) -> float:
    """Baseline : toujours parier sur le poleman."""
    return float(pole_hits(df).mean())


def topk_accuracy(df: pd.DataFrame, k: int = 3, proba_col: str = "proba") -> float:
    """% de courses ou le vrai vainqueur est dans le top k du modele."""
    rank = df.groupby("race_idx")[proba_col].rank(ascending=False, method="first")
    return float(df.loc[rank <= k].groupby("race_idx")[TARGET].max().mean())


def podium_accuracy(df: pd.DataFrame, k: int = 3, proba_col: str = "proba") -> float:
    """Part moyenne des k pilotes les mieux classes par le modele qui finissent sur le podium."""
    rank = df.groupby("race_idx")[proba_col].rank(ascending=False, method="first")
    top = df.loc[rank <= k]
    return float(top.groupby("race_idx")[PODIUM].mean().mean())


def pole_podium_baseline(df: pd.DataFrame, k: int = 3) -> float:
    """Baseline podium : les k premiers de la grille finissent sur le podium."""
    starters = df[df["grid"] > 0]
    rank = starters.groupby("race_idx")["grid"].rank(method="first")
    top = starters.loc[rank <= k]
    return float(top.groupby("race_idx")[PODIUM].mean().mean())


def bootstrap_diff(model_hits: pd.Series, base_hits: pd.Series, n_boot: int = 4000, seed: int = 0):
    """Ecart moyen modele - baseline et son intervalle de confiance a 95 % (bootstrap sur les courses)."""
    m, b = model_hits.align(base_hits, join="inner")
    d = (m - b).to_numpy()
    rng = np.random.default_rng(seed)
    means = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def add_race_rank(df: pd.DataFrame, proba_col: str = "proba") -> pd.DataFrame:
    """Ajoute la probabilite normalisee par course (somme = 1) et le rang predit."""
    out = df.copy()
    out["win_prob"] = out[proba_col] / out.groupby("race_idx")[proba_col].transform("sum")
    out["pred_rank"] = (out.groupby("race_idx")[proba_col]
                       .rank(ascending=False, method="first").astype(int))
    return out


def softmax_by_race(df: pd.DataFrame, score_col: str = "score") -> pd.Series:
    """Transforme des scores de classement en probabilites par course (somme = 1)."""
    z = df[score_col] - df.groupby("race_idx")[score_col].transform("max")  # stabilite numerique
    e = np.exp(z)
    return e / e.groupby(df["race_idx"]).transform("sum")


def relevance(df: pd.DataFrame) -> pd.Series:
    """Pertinence pour le classement : 2 = vainqueur, 1 = autre podium, 0 = le reste."""
    rel = df[TARGET].astype(int)
    if PODIUM in df.columns:
        rel = rel + df[PODIUM].astype(int)
    return rel


def scale_pos_weight(y) -> float:
    """~1 vainqueur pour ~20 pilotes : poids pour compenser le desequilibre des classes."""
    y = np.asarray(y)
    pos = y.sum()
    return float((len(y) - pos) / pos) if pos else 1.0
