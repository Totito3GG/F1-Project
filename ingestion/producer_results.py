"""
Producer : API Jolpica (ex-Ergast) -> Kafka topic "f1.results"
Un message = le résultat d'un pilote dans une course.
Lancer : python ingestion/producer_results.py --start 2015 --end 2024
"""
import argparse
import json
import os
import time

import requests
from dotenv import load_dotenv
from kafka import KafkaProducer

load_dotenv()
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
JOLPICA_URL = os.getenv("JOLPICA_URL", "https://api.jolpi.ca/ergast/f1")
TOPIC = "f1.results"
PAGE_SIZE = 100


def fetch_page(season, offset):
    """Appelle l'API pour une page, avec retry si on dépasse la limite (429)."""
    url = f"{JOLPICA_URL}/{season}/results.json"
    for attempt in range(5):
        r = requests.get(url, params={"limit": PAGE_SIZE, "offset": offset}, timeout=30)
        if r.status_code == 429:
            wait = 2 ** attempt
            print(f"  429 rate limit, attente {wait}s")
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.json()["MRData"]
    raise RuntimeError(f"API indisponible pour {season} offset {offset}")


def flatten(race, result):
    """Transforme la réponse imbriquée de l'API en un message plat."""
    return {
        "season": int(race["season"]),
        "round": int(race["round"]),
        "race_name": race["raceName"],
        "circuit_id": race["Circuit"]["circuitId"],
        "date": race["date"],
        "driver_id": result["Driver"]["driverId"],
        "constructor_id": result["Constructor"]["constructorId"],
        "grid": int(result["grid"]),
        "position": int(result["position"]),
        "points": float(result["points"]),
        "laps": int(result["laps"]),
        "status": result["status"],
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
        offset, total = 0, None
        while total is None or offset < total:
            data = fetch_page(season, offset)
            total = int(data["total"])
            for race in data["RaceTable"]["Races"]:
                for result in race["Results"]:
                    msg = flatten(race, result)
                    key = f'{msg["season"]}-{msg["round"]}-{msg["driver_id"]}'
                    producer.send(TOPIC, key=key, value=msg)
                    total_sent += 1
            offset += PAGE_SIZE
            time.sleep(0.5)  # respecter la limite de l'API
        print(f"Saison {season} : {total} résultats envoyés")

    producer.flush()
    producer.close()
    print(f"Terminé : {total_sent} messages dans '{TOPIC}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=2015)
    parser.add_argument("--end", type=int, default=2024)
    args = parser.parse_args()
    main(args.start, args.end)