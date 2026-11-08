"""
Silver -> Gold : une ligne par (pilote, course) avec des features connues AVANT la course.
Toutes les fenêtres s'arrêtent à la course précédente (-1) : pas de fuite de données.
"""
import os

from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import avg, coalesce, col, lit, to_date, when
from pyspark.sql.functions import count as _count
from pyspark.sql.functions import max as _max
from pyspark.sql.functions import sum as _sum

DATA = os.getenv("RACEFLOW_DATA", "/opt/app/data")
SILVER = f"{DATA}/silver/results"
SILVER_QUALI = f"{DATA}/silver/qualifying"   # optionnel : produit par qualifying_to_silver.py
GOLD = f"{DATA}/gold/features"

spark = SparkSession.builder.appName("f1-silver-to-gold").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

s = (spark.read.parquet(SILVER)
     .withColumn("race_idx", col("season") * 100 + col("round"))
     .withColumn("is_upcoming", lit(0)))

# ---------- Étape 10 : lignes « course à venir » ----------
# Le samedi soir, les qualifications d'une course existent déjà, mais pas ses résultats. Pour chaque course qui a des
# qualifications et qui vient APRÈS la dernière course courue, on crée une ligne par pilote : grille = place en
# qualification (les pénalités de grille ne sont pas encore connues), résultat vide (is_winner = null).
# Les fenêtres ci-dessous s'arrêtent toutes à la course précédente (-1) : les features de ces lignes sont calculées
# exactement comme celles d'une course passée, sans fuite possible.
if os.path.exists(SILVER_QUALI):
    last_done = s.agg(_max("race_idx")).first()[0]
    qa = spark.read.parquet(SILVER_QUALI).withColumn("race_idx", col("season") * 100 + col("round"))
    for c in ("race_name", "circuit_id", "race_date"):        # absentes des Silver produits avant l'étape 10
        if c not in qa.columns:
            qa = qa.withColumn(c, lit(None).cast("string"))
    upcoming = (
        qa.filter(col("race_idx") > last_done)
        .select("season", "round", "race_idx", "race_name", "circuit_id",
                to_date("race_date").alias("race_date"), "driver_id", "constructor_id",
                col("quali_position").alias("grid"))
        .withColumn("pit_lane_start", lit(0))
        .withColumn("is_upcoming", lit(1))
    )
    races = sorted(r.race_idx for r in upcoming.select("race_idx").distinct().collect())
    if races:
        print(f"Course(s) à venir (qualifications sans résultats) : {races}")
        s = s.unionByName(upcoming, allowMissingColumns=True)

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
    # etiquettes (connues APRES la course) : jamais des features, servent a evaluer le podium
    .withColumn("is_podium", (col("position") <= 3).cast("int"))
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

# ---------- Features de qualification (optionnelles) ----------
# Connues avant la course (qualifications du samedi), donc sans fuite de données.
HAS_QUALI = os.path.exists(SILVER_QUALI)
QUALI_COLS = []
CQ = None
if HAS_QUALI:
    qall = spark.read.parquet(SILVER_QUALI)
    q = qall.select("season", "round", "driver_id", "quali_position",
                    "q1_gap_pct", "q2_gap_pct", "q3_gap_pct")
    drv = (
        drv.join(q, ["season", "round", "driver_id"], "left")
        # > 0 : le pilote part derriere sa place de qualification (penalite) ; null si depart des stands
        .withColumn("quali_grid_delta", when(col("grid") > 0, col("grid") - col("quali_position")))
    )
    # Etape 7c : ecart au coequipier (meme voiture -> mesure le pilote). < 0 : plus rapide que lui.
    tm = Window.partitionBy("season", "round", "constructor_id")
    own = col("q1_gap_pct")
    n_other = _count("q1_gap_pct").over(tm) - when(own.isNotNull(), 1).otherwise(0)
    sum_other = _sum("q1_gap_pct").over(tm) - coalesce(own, lit(0.0))
    drv = drv.withColumn("quali_gap_vs_teammate",
                         when(own.isNotNull() & (n_other > 0), own - sum_other / n_other))
    # Etape 7c : rythme de l'ecurie en qualification sur les 5 courses PRECEDENTES (fenetre -5..-1)
    wcq = Window.partitionBy("constructor_id").orderBy("race_idx").rowsBetween(-5, -1)
    CQ = (qall.withColumn("race_idx", col("season") * 100 + col("round"))
          .groupBy("constructor_id", "race_idx").agg(avg("q1_gap_pct").alias("cq"))
          .withColumn("cons_quali_gap_last5", avg("cq").over(wcq))
          .select("constructor_id", "race_idx", "cons_quali_gap_last5"))
    QUALI_COLS = ["quali_position", "q1_gap_pct", "q2_gap_pct", "q3_gap_pct", "quali_grid_delta",
                  "quali_gap_vs_teammate", "cons_quali_gap_last5"]
else:
    print("Pas de Silver qualifications : Gold sans features de qualification")

# ---------- Table Gold ----------
full = drv.join(cons, ["constructor_id", "race_idx"], "left")
if CQ is not None:
    full = full.join(CQ, ["constructor_id", "race_idx"], "left")
# Etape 7c : force relative au meilleur de la course (comparable d'une saison a l'autre,
# contrairement aux points bruts dont le bareme a change)
rw = Window.partitionBy("race_idx")
full = (full
        .withColumn("drv_points_share", when(_max("drv_points_season").over(rw) > 0,
                                             col("drv_points_season") / _max("drv_points_season").over(rw)))
        .withColumn("cons_points_share", when(_max("cons_points_last5").over(rw) > 0,
                                              col("cons_points_last5") / _max("cons_points_last5").over(rw))))

gold = full.select(
    # identifiants
    "season", "round", "race_idx", "race_name", "circuit_id", "race_date",
    "driver_id", "constructor_id",
    # features
    "grid", "pit_lane_start",
    "drv_avg_pos_last5", "drv_wins_last5", "drv_podiums_last5",
    "drv_finish_rate_last10", "drv_points_season", "drv_circuit_avg_pos",
    "cons_avg_pos_last5", "cons_points_last5",
    "drv_points_share", "cons_points_share",
    *QUALI_COLS,
    # cible + etiquettes d'evaluation (post-course : ne pas utiliser comme features)
    "is_winner", "is_podium", col("position").alias("finish_position"),
    # étape 10 : 1 = course pas encore courue (cible vide), à prédire ; 0 = course courue
    "is_upcoming",
)

gold.write.mode("overwrite").partitionBy("season").parquet(GOLD)
n = gold.count()
print(f"Gold : {n} lignes, {len(gold.columns)} colonnes")
print(f"  dont lignes de course à venir : {gold.filter(col('is_upcoming') == 1).count()}")
if HAS_QUALI:
    print(f"  qualifications jointes : {gold.filter(col('q1_gap_pct').isNotNull()).count()} / {n} lignes")
print("Gold OK")
