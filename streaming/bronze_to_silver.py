"""
Spark Structured Streaming : Kafka "f1.results" -> data/bronze/results (Parquet)
Bronze = donnée brute : on garde le JSON tel quel + les métadonnées Kafka.
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, current_date

KAFKA_BOOTSTRAP = "kafka:29092"   # adresse interne Docker
TOPIC = "f1.results"
OUTPUT = "/opt/app/data/bronze/results"
CHECKPOINT = "/opt/app/data/checkpoints/bronze_results"

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