"""
Spark Structured Streaming : Kafka (f1.results ou f1.qualifying) -> data/bronze/<nom> (Parquet)
Bronze = donnée brute : on garde le JSON tel quel + les métadonnées Kafka.
"""
import os
import sys

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, current_date

# Usage : kafka_to_bronze.py [topic] [nom]   (défaut : f1.results results)
#         kafka_to_bronze.py f1.qualifying qualifying
# Adresse interne Docker par défaut ; surchargée par Airflow via l'environnement.
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_INTERNAL", "kafka:29092")
DATA = os.getenv("RACEFLOW_DATA", "/opt/app/data")
TOPIC = sys.argv[1] if len(sys.argv) > 1 else "f1.results"
NAME = sys.argv[2] if len(sys.argv) > 2 else "results"
OUTPUT = f"{DATA}/bronze/{NAME}"
CHECKPOINT = f"{DATA}/checkpoints/bronze_{NAME}"

spark = SparkSession.builder.appName("f1-kafka-to-bronze").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

raw = (
    spark.readStream.format("kafka")
    .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
    .option("subscribe", TOPIC)
    .option("startingOffsets", "earliest")
    .load()
)

bronze = raw.select(
    col("key").cast("string").alias("key"),
    col("value").cast("string").alias("value"),   # JSON brut, non parsé
    col("topic"),
    col("partition"),
    col("offset"),
    col("timestamp").alias("kafka_timestamp"),
    current_date().alias("ingestion_date"),
)

query = (
    bronze.writeStream.format("parquet")
    .option("path", OUTPUT)
    .option("checkpointLocation", CHECKPOINT)
    .partitionBy("ingestion_date")
    .trigger(availableNow=True)
    .start()
)

query.awaitTermination()
print("Bronze OK")
