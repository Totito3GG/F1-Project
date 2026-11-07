"""Tests des metriques, du split et du walk-forward de l'etape 7 (pandas seul, sans XGBoost ni MLflow)."""
import pandas as pd
import pytest

from ml.features import (BASE_FEATURES, EXTRA_FEATURES, QUALI_FEATURES, relevance, softmax_by_race, add_race_rank, available_features, bootstrap_diff, podium_accuracy,
                         pole_baseline_accuracy, pole_hits, pole_podium_baseline,
                         scale_pos_weight, temporal_split, topk_accuracy, walk_forward,
                         winner_accuracy, winner_hits)


@pytest.fixture
def races():
    # 2 courses, 3 pilotes chacune
    # course 1 : le poleman (a) gagne, le modele prefere b  -> modele faux, baseline juste
    # course 2 : c gagne depuis P3 (le poleman d part des stands = grid 0 ignore), modele juste
    return pd.DataFrame({
        "season":    [2023, 2023, 2023, 2024, 2024, 2024],
        "race_idx":  [202301] * 3 + [202401] * 3,
        "driver_id": ["a", "b", "c", "d", "e", "c"],
        "grid":      [1, 2, 3, 0, 2, 3],
        "is_winner": [1, 0, 0, 0, 0, 1],
        "proba":     [0.3, 0.6, 0.1, 0.2, 0.1, 0.7],
    })


def test_winner_accuracy(races):
    assert winner_accuracy(races) == 0.5


def test_pole_baseline_ignores_pit_lane_start(races):
    # course 2 : grid 0 exclu, le "poleman" est e (grid 2) qui ne gagne pas
    assert pole_baseline_accuracy(races) == 0.5


def test_top3_accuracy(races):
    assert topk_accuracy(races, k=1) == 0.5
    assert topk_accuracy(races, k=2) == 1.0


def test_temporal_split_no_overlap(races):
    train, test = temporal_split(races, test_from=2024)
    assert train["season"].max() < test["season"].min()
    with pytest.raises(ValueError):
        temporal_split(races, test_from=2030)


def test_add_race_rank_sums_to_one(races):
    out = add_race_rank(races)
    assert out.groupby("race_idx")["win_prob"].sum().round(6).eq(1).all()
    assert out.loc[out.proba == 0.6, "pred_rank"].item() == 1


def test_scale_pos_weight():
    assert scale_pos_weight([1, 0, 0, 0]) == 3.0


def test_podium_accuracy_and_baseline():
    df = pd.DataFrame({
        "race_idx":  [1, 1, 1, 1],
        "grid":      [1, 2, 3, 4],
        "is_podium": [1, 1, 1, 0],
        "proba":     [0.1, 0.2, 0.3, 0.9],
    })
    # top 3 du modele = pilotes 4, 3, 2 -> podium 0, 1, 1
    assert podium_accuracy(df) == pytest.approx(2 / 3)
    # top 3 de la grille = pilotes 1, 2, 3 -> tous sur le podium
    assert pole_podium_baseline(df) == 1.0


def test_hits_are_per_race(races):
    assert list(winner_hits(races)) == [0.0, 1.0]
    assert list(pole_hits(races)) == [1.0, 0.0]


def test_bootstrap_diff_identical_and_better():
    same = pd.Series([1.0, 0.0, 1.0, 0.0], index=[1, 2, 3, 4])
    assert bootstrap_diff(same, same) == (0.0, 0.0, 0.0)
    all_win, none = pd.Series([1.0] * 5, index=range(5)), pd.Series([0.0] * 5, index=range(5))
    assert bootstrap_diff(all_win, none) == (1.0, 1.0, 1.0)


def test_walk_forward_never_trains_on_the_future():
    df = pd.DataFrame({"season": [2016, 2017, 2018, 2019], "x": range(4)})
    seen = []

    def fit_predict(train, test):
        assert train["season"].max() < test["season"].min()
        seen.append(int(test["season"].iloc[0]))
        return [0.5] * len(test)

    out = walk_forward(df, fit_predict, first_test=2018)
    assert seen == [2018, 2019]
    assert sorted(out["season"]) == [2018, 2019]
    with pytest.raises(ValueError):
        walk_forward(df, fit_predict, first_test=2030)


def test_available_features_with_and_without_quali():
    gold_old = pd.DataFrame(columns=BASE_FEATURES)
    gold_new = pd.DataFrame(columns=BASE_FEATURES + QUALI_FEATURES)
    assert available_features(gold_old) == BASE_FEATURES          # Gold sans qualifs : jeu de base
    assert available_features(gold_new) == BASE_FEATURES + QUALI_FEATURES
    assert available_features(gold_new, use_quali=False) == BASE_FEATURES


def test_available_features_extra_group():
    gold = pd.DataFrame(columns=BASE_FEATURES + QUALI_FEATURES + EXTRA_FEATURES)
    assert available_features(gold) == BASE_FEATURES + QUALI_FEATURES + EXTRA_FEATURES
    assert available_features(gold, use_extra=False) == BASE_FEATURES + QUALI_FEATURES
    assert available_features(gold, use_quali=False, use_extra=False) == BASE_FEATURES


def test_softmax_by_race_sums_to_one_and_keeps_order():
    df = pd.DataFrame({"race_idx": [1, 1, 1, 2, 2], "score": [2.0, 0.0, -1.0, 5.0, 5.0]})
    p = softmax_by_race(df)
    assert p.groupby(df["race_idx"]).sum().round(9).eq(1).all()
    assert p.iloc[0] > p.iloc[1] > p.iloc[2]
    assert p.iloc[3] == pytest.approx(0.5)


def test_relevance_winner_2_podium_1():
    df = pd.DataFrame({"is_winner": [1, 0, 0], "is_podium": [1, 1, 0]})
    assert list(relevance(df)) == [2, 1, 0]
