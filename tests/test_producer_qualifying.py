"""Tests du producer de qualifications : transformation seule, sans API ni Kafka."""
from ingestion.producer_qualifying import flatten_qualifying

RACE = {"season": "2024", "round": "5", "raceName": "Chinese Grand Prix"}


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
