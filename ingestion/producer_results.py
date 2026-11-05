"""
Producer : API Jolpica (ex-Ergast) -> Kafka topic "f1.results"
Ex. URL : https://api.jolpi.ca/ergast/f1/2024/results.json?limit=100
"""
# TODO 1 : appeler l API avec requests (pagination limit/offset)
# TODO 2 : creer un KafkaProducer (bootstrap_servers="localhost:9092")
# TODO 3 : envoyer chaque resultat en JSON dans le topic "f1.results"

