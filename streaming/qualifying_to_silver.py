"""
Qualifications Bronze -> Silver
- parse le JSON avec un schéma explicite
- déduplique sur la clé métier (saison-course-pilote), en gardant la version la plus récente
- convertit les temps "m:ss.mmm" en secondes
- calcule l'écart (en %) au plus rapide de CHAQUE séance (Q1, Q2, Q3) de la même course :
  comparable d'un circuit à l'autre, contrairement à un temps brut
"""
import os

from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import col, from_json, lit
from pyspark.sql.functions import min as _min
from pyspark.sql.functions import row_number, size, split, when
from pyspark.sql.types import IntegerType, StringType, StructField, StructType

DATA = os.getenv("RACEFLOW_DATA", "/opt/app/data")
BRONZE = f"{DATA}/bronze/qualifying"
SILVER = f"{DATA}/silver/qualifying"

SCHEMA = StructType([
    StructField("season", IntegerType()),
    StructField("round", IntegerType()),
    StructField("driver_id", StringType()),
    StructField("constructor_id", StringType()),
    StructField("quali_position", IntegerType()),
    StructField("q1", StringType()),
    StructField("q2", StringType()),
    StructField("q3", StringType()),
    StructField("ingested_at", StringType()),
])


def to_seconds(c):
    """'1:26.572' -> 86.572 ; null ou vide -> null."""
    parts = split(c, ":")
    return (
        when(c.isNull() | (c == ""), lit(None).cast("double"))
        .when(size(parts) == 2, parts.getItem(0).cast("double") * 60 + parts.getItem(1).cast("double"))
        .otherwise(c.cast("double"))
    )


spark = SparkSession.builder.appName("f1-qualifying-to-silver").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

bronze = spark.read.parquet(BRONZE)
print(f"Bronze qualifying : {bronze.count()} lignes")

parsed = (
    bronze.select("key", "kafka_timestamp", from_json(col("value"), SCHEMA).alias("d"))
    .select("key", "kafka_timestamp", "d.*")
)

w = Window.partitionBy("key").orderBy(col("kafka_timestamp").desc())
dedup = (
    parsed.withColumn("rn", row_number().over(w))
    .filter(col("rn") == 1)
    .drop("rn")
    .filter(col("driver_id").isNotNull() & col("season").isNotNull() & col("round").isNotNull())
)

race = Window.partitionBy("season", "round")
silver = dedup
for q in ("q1", "q2", "q3"):
    silver = (
        silver.withColumn(f"{q}_s", to_seconds(col(q)))
        # écart au plus rapide de la séance, en % du temps : 0 = le plus rapide
        .withColumn(f"{q}_gap_pct", (col(f"{q}_s") / _min(f"{q}_s").over(race) - 1) * 100)
    )

silver = silver.select(
    "season", "round", "driver_id", "constructor_id", "quali_position",
    "q1_s", "q2_s", "q3_s", "q1_gap_pct", "q2_gap_pct", "q3_gap_pct",
)

silver.write.mode("overwrite").partitionBy("season").parquet(SILVER)

n = silver.count()
print(f"Silver qualifying : {n} lignes (doublons supprimés)")
print(f"  avec un temps Q1 : {silver.filter(col('q1_s').isNotNull()).count()}"
      f" | Q3 : {silver.filter(col('q3_s').isNotNull()).count()}")
print("Silver qualifying OK")
