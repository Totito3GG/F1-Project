"""
Silver -> Gold : une ligne par (pilote, course) avec des features connues AVANT la course.
Toutes les fenêtres s'arrêtent à la course précédente (-1) : pas de fuite de données.
"""
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import avg, coalesce, col, lit
from pyspark.sql.functions import sum as _sum

SILVER = "/opt/app/data/silver/results"
GOLD = "/opt/app/data/gold/features"

spark = SparkSession.builder.appName("f1-silver-to-gold").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

s = spark.read.parquet(SILVER).withColumn("race_idx", col("season") * 100 + col("round"))

# ---------- Features pilote ----------
wd = Window.partitionBy("driver_id").orderBy("race_idx")
last5 = wd.rowsBetween(-5, -1)
last10 = wd.rowsBetween(-10, -1)
season_before = (Window.partitionBy("driver_id", "season").orderBy("race_idx")
                 .rowsBetween(Window.unboundedPreceding, -1))
circuit_before = (Window.partitionBy("driver_id", "circuit_id").orderBy("race_idx")
                  .rowsBetween(Window.unboundedPreceding, -1))

drv = (
    s.withColumn("drv_avg_pos_last5", avg("position").over(last5))
    .withColumn("drv_wins_last5", _sum("is_winner").over(last5))
    .withColumn("drv_podiums_last5", _sum((col("position") <= 3).cast("int")).over(last5))
    .withColumn("drv_finish_rate_last10", avg("finished").over(last10))
    .withColumn("drv_points_season", coalesce(_sum("points").over(season_before), lit(0.0)))
    .withColumn("drv_circuit_avg_pos", avg("position").over(circuit_before))
)

# ---------- Features écurie (2 pilotes -> 1 ligne par écurie et par course) ----------
cons_race = s.groupBy("constructor_id", "race_idx").agg(
    avg("position").alias("c_pos"), _sum("points").alias("c_pts"))
wc = Window.partitionBy("constructor_id").orderBy("race_idx").rowsBetween(-5, -1)
cons = (
    cons_race.withColumn("cons_avg_pos_last5", avg("c_pos").over(wc))
    .withColumn("cons_points_last5", _sum("c_pts").over(wc))
    .select("constructor_id", "race_idx", "cons_avg_pos_last5", "cons_points_last5")
)

# ---------- Table Gold ----------
gold = drv.join(cons, ["constructor_id", "race_idx"], "left").select(
    # identifiants
    "season", "round", "race_idx", "race_name", "circuit_id", "race_date",
    "driver_id", "constructor_id",
    # features
    "grid", "pit_lane_start",
    "drv_avg_pos_last5", "drv_wins_last5", "drv_podiums_last5",
    "drv_finish_rate_last10", "drv_points_season", "drv_circuit_avg_pos",
    "cons_avg_pos_last5", "cons_points_last5",
    # cible
    "is_winner",
)

gold.write.mode("overwrite").partitionBy("season").parquet(GOLD)
print(f"Gold : {gold.count()} lignes, {len(gold.columns)} colonnes")
print("Gold OK")