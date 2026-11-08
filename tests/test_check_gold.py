"""Tests du contrôle qualité Gold et du générateur de données de la CI (pandas seul)."""
import pandas as pd

from ml.check_gold import check_gold
from ml.features import BASE_FEATURES, ID_COLS, TARGET
from scripts.make_synthetic_data import make_races, to_bronze


def healthy_gold(n_races: int = 3) -> pd.DataFrame:
    rows = []
    for race in range(n_races):
        for i in range(4):
            row = {c: 1.0 for c in BASE_FEATURES}
            row.update({c: "x" for c in ID_COLS})
            row.update({"race_idx": 202400 + race, "driver_id": f"d{i}", TARGET: int(i == 0)})
            rows.append(row)
    return pd.DataFrame(rows)


def test_healthy_gold_has_no_problem():
    assert check_gold(healthy_gold()) == []


def test_empty_gold_is_rejected():
    assert check_gold(pd.DataFrame()) == ["la table Gold est vide"]


def test_missing_column_is_reported():
    problems = check_gold(healthy_gold().drop(columns=["grid"]))
    assert len(problems) == 1 and "grid" in problems[0]


def test_race_without_winner_is_reported():
    df = healthy_gold()
    df.loc[df["race_idx"] == 202400, TARGET] = 0
    assert any("exactement un vainqueur" in p for p in check_gold(df))


def test_duplicates_and_empty_feature_are_reported():
    df = pd.concat([healthy_gold(), healthy_gold().head(1)], ignore_index=True)
    df["grid"] = float("nan")
    problems = check_gold(df)
    assert any("doublons" in p for p in problems)
    assert any("grid" in p and "vides" in p for p in problems)


def test_synthetic_races_have_one_winner_and_matching_quali():
    results, quali = make_races([2020], rounds=3, upcoming=False)
    df = pd.DataFrame(results)
    assert df.groupby("round")["position"].min().eq(1).all()
    assert df.groupby("round")["driver_id"].nunique().eq(20).all()
    assert len(quali) == len(results)
    assert {q["quali_position"] for q in quali if q["round"] == 1} == set(range(1, 21))


def test_synthetic_upcoming_race_has_qualifying_but_no_results():
    """Étape 10 : la dernière saison a une manche de plus, qualifiée mais pas encore courue."""
    results, quali = make_races([2019, 2020], rounds=3)
    assert {r["round"] for r in results if r["season"] == 2020} == {1, 2, 3}
    assert {q["round"] for q in quali if q["season"] == 2020} == {1, 2, 3, 4}
    assert {q["round"] for q in quali if q["season"] == 2019} == {1, 2, 3}       # saisons passées : complètes
    upcoming = [q for q in quali if q["season"] == 2020 and q["round"] == 4]
    assert len(upcoming) == 20 and all(q["race_name"] and q["circuit_id"] and q["race_date"] for q in upcoming)


def test_synthetic_bronze_adds_duplicates_with_kafka_columns():
    results, _ = make_races([2020], rounds=2)
    bronze = to_bronze(results, "f1.results")
    assert len(bronze) > len(results)
    assert bronze["key"].nunique() == len(results)
    assert {"key", "value", "topic", "partition", "offset", "kafka_timestamp", "ingestion_date"} <= set(bronze.columns)
