"""Tests des calculs du dashboard (dashboard/data.py) : pandas seul, sans Streamlit ni MLflow."""
import pandas as pd
import requests

from dashboard import data
from dashboard.data import (
    driver_history,
    driver_stats,
    is_upcoming_race,
    pretty,
    race_summary,
    read_parquet,
    season_scores,
)


def predictions() -> pd.DataFrame:
    """2 courses, 3 pilotes. Course 1 : le modèle a raison, pas la pole. Course 2 : la pole a raison, pas le modèle."""
    rows = [
        # race_idx, pilote, grille, vainqueur, rang prédit
        (202401, "verstappen", 1, 0, 2), (202401, "norris", 2, 1, 1), (202401, "leclerc", 3, 0, 3),
        (202402, "verstappen", 1, 1, 3), (202402, "norris", 2, 0, 1), (202402, "leclerc", 3, 0, 2),
    ]
    df = pd.DataFrame(rows, columns=["race_idx", "driver_id", "grid", "is_winner", "pred_rank"])
    df["season"] = 2024
    df["round"] = df["race_idx"] % 100
    df["race_name"] = "GP"
    return df


def test_race_summary_compares_model_and_pole():
    s = race_summary(predictions())
    assert list(s["favori_modele"]) == ["norris", "norris"]
    assert list(s["vainqueur"]) == ["norris", "verstappen"]
    assert list(s["modele_juste"]) == [1, 0]
    assert list(s["pole_juste"]) == [0, 1]


def test_season_scores_adds_a_total_line():
    scores = season_scores(race_summary(predictions()))
    assert list(scores["season"]) == ["2024", "Total"]
    total = scores.iloc[-1]
    assert total["courses"] == 2 and total["modele"] == 0.5 and total["pole"] == 0.5


def test_driver_stats_ignores_pit_lane_start_in_average_grid():
    gold = pd.DataFrame({
        "season": [2024] * 3, "race_idx": [202401, 202402, 202403], "driver_id": ["hamilton"] * 3,
        "constructor_id": ["ferrari"] * 3, "grid": [0, 2, 4], "is_winner": [0, 1, 0], "is_podium": [0, 1, 1],
        "finish_position": [10.0, 1.0, None],
    })
    row = driver_stats(gold, 2024).iloc[0]
    assert row["courses"] == 3 and row["victoires"] == 1 and row["podiums"] == 2
    # étape 10 : une course à venir (pole en qualification, pas de résultat) ne compte pas
    upcoming = {"season": 2024, "race_idx": 202404, "driver_id": "hamilton", "constructor_id": "ferrari",
                "grid": 1, "is_winner": None, "is_podium": None, "finish_position": None, "is_upcoming": 1}
    with_upcoming = pd.concat([gold.assign(is_upcoming=0), pd.DataFrame([upcoming])], ignore_index=True)
    row = driver_stats(with_upcoming, 2024).iloc[0]
    assert row["courses"] == 3 and row["poles"] == 0
    assert row["grille_moy"] == 3.0       # (2 + 4) / 2 : le départ des stands (grid = 0) n'est pas compté
    assert row["arrivee_moy"] == 5.5      # (10 + 1) / 2 : la course non classée n'est pas comptée


def test_driver_history_hides_pit_lane_start():
    gold = pd.DataFrame({"season": [2024, 2024], "round": [2, 1], "race_name": ["B", "A"],
                         "driver_id": ["hamilton"] * 2, "grid": [0, 3], "finish_position": [8.0, 2.0]})
    h = driver_history(gold, "hamilton", 2024)
    assert list(h["round"]) == [1, 2]       # trié par manche
    assert h["grid"].iloc[0] == 3 and pd.isna(h["grid"].iloc[1])


def test_pretty_names():
    assert pretty("max_verstappen") == "Max Verstappen"
    assert pretty(None) == "—"
    assert pretty(float("nan")) == "—"      # ex. course sans poleman dans les données


def test_read_parquet_returns_none_when_file_is_missing(tmp_path):
    assert read_parquet(tmp_path / "absent.parquet") is None


def test_latest_run_returns_none_when_mlflow_is_down(monkeypatch):
    def down(*args, **kwargs):
        raise requests.ConnectionError("MLflow arrêté")

    monkeypatch.setattr(data.requests, "get", down)
    assert data.latest_run("http://mlflow-absent:5000") is None


def test_is_upcoming_race():
    """Étape 10 : course à venir si is_upcoming = 1 ; sans cette colonne, si aucun résultat n'est connu."""
    done = predictions()
    assert not is_upcoming_race(done)
    assert is_upcoming_race(done.assign(is_upcoming=1, is_winner=float("nan")))
    assert not is_upcoming_race(done.assign(is_upcoming=0))
    assert is_upcoming_race(done.assign(is_winner=float("nan")))      # ancien format, sans is_upcoming
