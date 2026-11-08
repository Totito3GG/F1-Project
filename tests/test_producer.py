"""Tests unitaires du producer : on vérifie la transformation, sans appeler l'API ni Kafka."""
from ingestion.producer_results import flatten

RACE = {
    "season": "2024",
    "round": "5",
    "raceName": "Chinese Grand Prix",
    "Circuit": {"circuitId": "shanghai"},
    "date": "2024-04-21",
}

RESULT = {
    "Driver": {"driverId": "max_verstappen"},
    "Constructor": {"constructorId": "red_bull"},
    "grid": "1",
    "position": "1",
    "points": "25",
    "laps": "56",
    "status": "Finished",
}


def test_flatten_types():
    msg = flatten(RACE, RESULT)
    assert msg["season"] == 2024
    assert msg["round"] == 5
    assert msg["grid"] == 1
    assert msg["points"] == 25.0


def test_flatten_ids():
    msg = flatten(RACE, RESULT)
    assert msg["driver_id"] == "max_verstappen"
    assert msg["constructor_id"] == "red_bull"
    assert msg["circuit_id"] == "shanghai"


def test_flatten_has_ingestion_time():
    msg = flatten(RACE, RESULT)
    assert msg["ingested_at"].endswith("Z")


def test_missing_values_become_none_instead_of_crashing():
    """Saison 2026 : l'API renvoie parfois grid = null. Le producer ne doit pas planter."""
    msg = flatten(RACE, {**RESULT, "grid": None, "laps": "", "points": None})
    assert msg["grid"] is None and msg["laps"] is None and msg["points"] == 0.0
    assert msg["position"] == 1


def test_pit_lane_start_stays_zero():
    """grid = "0" (départ des stands) reste 0 : ce n'est pas une valeur manquante."""
    assert flatten(RACE, {**RESULT, "grid": "0"})["grid"] == 0
