"""
Producer : API Jolpica (ex-Ergast) -> Kafka topic "f1.qualifying"
Un message = les temps de qualification (Q1, Q2, Q3) d'un pilote pour un Grand Prix.
Un pilote éliminé en Q1 n'a pas de Q2 ni de Q3 : ces champs valent null.
Lancer : python ingestion/producer_qualifying.py --start 2015 --end 2025
"""
import argparse
import json
import sys
import time
from pathlib import Path

from kafka import KafkaProducer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ingestion.producer_results import KAFKA_BOOTSTRAP, PAGE_SIZE, fetch_page  # noqa: E402

TOPIC = "f1.qualifying"


def _time(value):
    """'1:26.572' reste tel quel ; absent ou vide -> None (éliminé avant cette séance)."""
    return value if value else None


def flatten_qualifying(race, result):
    """Transforme la réponse imbriquée de l'API en un message plat."""
    return {
        "season": int(race["season"]),
        "round": int(race["round"]),
        "driver_id": result["Driver"]["driverId"],
        "constructor_id": result["Constructor"]["constructorId"],
        "quali_position": int(result["position"]),
        "q1": _time(result.get("Q1")),
        "q2": _time(result.get("Q2")),
        "q3": _time(result.get("Q3")),
        "ingested_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def main(start, end):
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        key_serializer=lambda k: k.encode("utf-8"),
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        acks="all",
    )

    total_sent = 0
    for season in range(start, end + 1):
        offset, total, sent_season = 0, None, 0
        while total is None or offset < total:
            data = fetch_page(season, offset, endpoint="qualifying")
            total = int(data["total"])
            for race in data["RaceTable"]["Races"]:
                for result in race.get("QualifyingResults", []):
                    msg = flatten_qualifying(race, result)
                    key = f'{msg["season"]}-{msg["round"]}-{msg["driver_id"]}'
                    producer.send(TOPIC, key=key, value=msg)
                    total_sent += 1
                    sent_season += 1
                    if total_sent == 1:
                        print("Exemple de message :", msg)
            offset += PAGE_SIZE
            time.sleep(0.5)  # respecter la limite de l'API
        print(f"Saison {season} : {sent_season} lignes de qualification envoyées")

    producer.flush()
    producer.close()
    print(f"Terminé : {total_sent} messages dans '{TOPIC}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=2015)
    parser.add_argument("--end", type=int, default=2025)
    args = parser.parse_args()
    main(args.start, args.end)
