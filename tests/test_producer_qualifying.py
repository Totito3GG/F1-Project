"""Tests du producer de qualifications : transformation seule, sans API ni Kafka."""
from ingestion.producer_qualifying import flatten_qualifying

RACE = {"season": "2024", "round": "5", "raceName": "Chinese Grand Prix", "date": "2024-04-21",
        "Circuit": {"circuitId": "shanghai"}}


def _result(**times):
    base = {
        "number": "1", "position": "1",
        "Driver": {"driverId": "max_verstappen"},
        "Constructor": {"constructorId": "red_bull"},
    }
    base.update(times)
    return base


def test_flatten_full_qualifying():
    msg = flatten_qualifying(RACE, _result(Q1="1:35.000", Q2="1:34.500", Q3="1:34.012"))
    assert (msg["season"], msg["round"], msg["quali_position"]) == (2024, 5, 1)
    assert msg["driver_id"] == "max_verstappen" and msg["constructor_id"] == "red_bull"
    assert (msg["q1"], msg["q2"], msg["q3"]) == ("1:35.000", "1:34.500", "1:34.012")


def test_eliminated_in_q1_has_no_q2_q3():
    msg = flatten_qualifying(RACE, _result(position="18", Q1="1:37.900"))
    assert msg["quali_position"] == 18
    assert msg["q1"] == "1:37.900" and msg["q2"] is None and msg["q3"] is None


def test_empty_strings_become_none():
    msg = flatten_qualifying(RACE, _result(Q1="", Q2="", Q3=""))
    assert msg["q1"] is None and msg["q2"] is None and msg["q3"] is None


def test_race_info_is_kept_for_upcoming_races():
    """Étape 10 : le message porte le nom, le circuit et la date de la course (course à venir)."""
    msg = flatten_qualifying(RACE, _result(Q1="1:35.000"))
    assert (msg["race_name"], msg["circuit_id"], msg["race_date"]) == ("Chinese Grand Prix", "shanghai", "2024-04-21")


def test_race_info_missing_becomes_none():
    msg = flatten_qualifying({"season": "2024", "round": "5"}, _result(Q1="1:35.000"))
    assert msg["race_name"] is None and msg["circuit_id"] is None and msg["race_date"] is None
