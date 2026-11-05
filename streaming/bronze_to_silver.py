"""
Bronze -> Silver
- parse le JSON avec un schéma explicite
- déduplique sur la clé métier (saison-course-pilote), en gardant la version la plus récente
- filtre les lignes invalides, ajoute des colonnes utiles
"""
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import col, from_json, row_number, to_date, when
from pyspark.sql.types import (DoubleType, IntegerType, StringType,
                               StructField, StructType)

BRONZE = "/opt/app/data/bronze/results"
SILVER = "/opt/app/data/silver/results"

SCHEMA = StructType([
    StructField("season", IntegerType()),
    StructField("round", IntegerType()),
    StructField("race_name", StringType()),
    StructField("circuit_id", StringType()),
    StructField("date", StringType()),
    StructField("driver_id", StringType()),
    StructField("constructor_id", StringType()),
    StructField("grid", IntegerType()),
    StructField("position", IntegerType()),
    StructField("points", DoubleType()),
    StructField("laps", IntegerType()),
    StructField("status", StringType()),
    StructField("ingested_at", StringType()),
])

spark = SparkSession.builder.appName("f1-bronze-to-silver").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

bronze = spark.read.parquet(BRONZE)
print(f"Bronze : {bronze.count()} lignes")

# 1. Parsing : JSON texte -> colonnes typées
parsed = (
    bronze.select("key", "kafka_timestamp", from_json(col("value"), SCHEMA).alias("d"))
    .select("key", "kafka_timestamp", "d.*")
)

# 2. Déduplication : une ligne par clé, la plus récente
w = Window.partitionBy("key").orderBy(col("kafka_timestamp").desc())
dedup = (
    parsed.withColumn("rn", row_number().over(w))
    .filter(col("rn") == 1)
    .drop("rn")
)

# 3. Nettoyage + colonnes métier
silver = (
    dedup.filter(col("driver_id").isNotNull() & col("season").isNotNull())
    .withColumn("race_date", to_date("date"))
    .drop("date")
    .withColumn("is_winner", (col("position") == 1).cast("int"))
    .withColumn("finished",
                when((col("status") == "Finished") | col("status").startswith("+"), 1)
                .otherwise(0))
    .withColumn("pit_lane_start", (col("grid") == 0).cast("int"))
)

silver.write.mode("overwrite").partitionBy("season").parquet(SILVER)
print(f"Silver : {silver.count()} lignes (doublons supprimés)")
print("Silver OK")