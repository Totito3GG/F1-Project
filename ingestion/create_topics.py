"""
Crée les topics Kafka du projet s'ils n'existent pas (utilisé par le DAG ingest_historical).

Lancer :
    python ingestion/create_topics.py
Variable d'environnement : KAFKA_BOOTSTRAP (défaut localhost:9092 ; dans Docker : kafka:29092)
"""
import os
import sys
import time

from kafka import KafkaAdminClient
from kafka.admin import NewTopic
from kafka.errors import NoBrokersAvailable, TopicAlreadyExistsError

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
TOPICS = ["f1.results", "f1.laps", "f1.qualifying"]


def connect(retries: int = 15, wait: float = 4.0) -> KafkaAdminClient:
    """Kafka peut mettre quelques secondes à répondre après un démarrage : on réessaie."""
    for attempt in range(1, retries + 1):
        try:
            return KafkaAdminClient(bootstrap_servers=KAFKA_BOOTSTRAP)
        except NoBrokersAvailable:
            print(f"Kafka ({KAFKA_BOOTSTRAP}) ne répond pas encore ({attempt}/{retries})...")
            time.sleep(wait)
    sys.exit(f"Kafka injoignable sur {KAFKA_BOOTSTRAP}")


def main() -> None:
    admin = connect()
    try:
        for name in TOPICS:
            try:
                admin.create_topics([NewTopic(name=name, num_partitions=3, replication_factor=1)])
                print(f"Topic créé : {name}")
            except TopicAlreadyExistsError:
                print(f"Topic déjà présent : {name}")
    finally:
        admin.close()
    print("Topics OK")


if __name__ == "__main__":
    main()
