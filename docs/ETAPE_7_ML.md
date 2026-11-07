# Étape 7 — Modèle XGBoost + MLflow (7, 7b, 7c)

Branche : `etape-7-ml` · Date : 7 novembre 2026 · Auteur : Yassir (avec l'équipe RaceFlow)

## 1. Objectif

Prédire le vainqueur d'un Grand Prix à partir de la table Gold, puis mesurer honnêtement si le modèle fait mieux
que la baseline la plus simple : **« le poleman gagne »**.

Chaîne complète exécutée sur la machine locale :

```
Jolpica API → producer (Kafka) → Bronze → Silver → Gold → XGBoost + MLflow
```

## 2. Ce qui a été fait aujourd'hui

### 2.1 Mise en route sur la machine
- Dépôt cloné et pipeline relancé de bout en bout (Kafka, Spark, Bronze → Silver → Gold, entraînement).
- **Kafka** : ajout de `hostname: kafka` et d'écoutes explicites sur `0.0.0.0` dans `docker-compose.yml`
  (correction de l'`UnknownHostException` au démarrage sous Docker Desktop).
- Conflit de ports 9092 / 8080 avec d'autres conteneurs : résolu en arrêtant les anciens conteneurs.
- Script d'installation `scripts/setup.ps1` (environnement Python, Docker, producers, Spark, entraînement).

### 2.2 Correction d'un bug de qualité des données
- Le statut `Lapped` (pilote classé, mais à un tour ou plus du vainqueur) n'était pas compté comme « a terminé ».
  Corrigé dans `streaming/bronze_to_silver.py`. Cela a modifié les features de taux de finition.

### 2.3 Étape 7 — modèle et évaluation robuste
- `ml/features.py` : chargement de Gold, split temporel, validation **walk-forward**, métriques par course,
  intervalle de confiance **bootstrap** sur l'écart avec la baseline.
- `ml/train.py` : XGBoost peu profond (`max_depth=2`, 150 arbres), suivi **MLflow** (paramètres, métriques, importance
  des features, modèle enregistré dans le registry `raceflow-winner-xgb`).
- Pas de fuite de données : toutes les fenêtres de Gold s'arrêtent à la course précédente (`rowsBetween(-5, -1)`).
- Évaluation :
  - split temporel : entraînement 2015-2023, test 2024-2025 (48 courses) ;
  - walk-forward : saisons testées 2018-2025 (173 courses), chaque saison prédite avec les saisons précédentes seulement ;
  - métriques : `winner_accuracy`, baseline pole, `top3_accuracy`, `podium_accuracy`, ROC-AUC, log-loss.

### 2.4 Étape 7b — temps de qualification
Les features d'historique n'apportaient rien au-delà de la place sur la grille. Ajout des temps de qualification :

- `ingestion/producer_qualifying.py` : Jolpica → topic Kafka `f1.qualifying` (4 684 lignes, saisons 2015-2025).
- `streaming/kafka_to_bronze.py` généralisé (topic et nom en argument) ;
  `streaming/qualifying_to_silver.py` : déduplication, conversion des temps en secondes, écart en % au plus rapide.
- `streaming/silver_to_gold.py` : jointure des qualifications (4 612 / 4 698 lignes jointes).
- Features ajoutées : `q1_gap_pct`, `q3_gap_pct`, `quali_grid_delta` (place sur la grille − place en qualifications).

### 2.5 Étape 7c — features relatives et modèle de classement
Idées définies avant de regarder les résultats, pour éviter d'ajuster le modèle sur le test :

- `q2_gap_pct`, `quali_gap_vs_teammate` (écart au coéquipier), `cons_quali_gap_last5` (rythme de l'écurie sur les 5 courses
  précédentes), `drv_points_share` et `cons_points_share` (points relatifs au meilleur de la course).
- Un modèle de **classement par course** (XGBoost Ranker) en plus du classifieur « vainqueur oui/non ».
- `train.py` compare automatiquement les 4 variantes sur les mêmes courses.

## 3. Résultats

### 3.1 Évolution du modèle (walk-forward 2018-2025, 173 courses)

| Version | Vainqueur trouvé | Baseline pole | Écart (IC 95 %) |
|---|---|---|---|
| Étape 7, avant la correction `Lapped` | 47,9 % (test 2024+) | 56,2 % | non évalué en walk-forward |
| Étape 7, après correction | 50,9 % | 52,0 % | −1,2 % [−6,9 ; +4,6] |
| **Étape 7b, avec qualifications** | **56,1 %** | 52,0 % | **+4,0 % [−1,2 ; +9,8]** |

### 3.2 Comparaison des variantes (même run, mêmes courses)

| Variante | Vainqueur | Écart vs pole (IC 95 %) | Podium |
|---|---|---|---|
| Base (étape 7) | 51,4 % | −0,6 % [−6,4 ; +5,2] | 65,9 % |
| **+ qualifications (7b)** | **56,1 %** | **+4,0 % [−1,2 ; +9,8]** | 67,4 % |
| + features 7c | 54,9 % | +2,9 % [−4,0 ; +9,8] | 67,1 % |
| Classement + features 7c | 53,8 % | +1,7 % [−4,0 ; +7,5] | 69,4 % |

Baseline podium (les 3 premiers de la grille) : 69,2 %.

Importance des features du modèle 7b : `q3_gap_pct` (0,276) et `grid` (0,276) en tête, puis `drv_podiums_last5` (0,130)
et `drv_wins_last5` (0,085).

### 3.3 Lecture honnête
- Les **temps de qualification** apportent l'essentiel du gain (environ +5 points par rapport à la base).
- Les features 7c et le modèle de classement **n'améliorent pas** le résultat.
- **Aucune variante ne bat la pole avec certitude** : toutes les bornes basses des intervalles de confiance sont négatives.
  Les écarts entre variantes sont dans le bruit.
- Sur le test final 2024-2025 (48 courses), le modèle 7b fait 54,2 % contre 56,2 % pour la pole. Les résultats varient
  beaucoup selon la saison (2024 : 54,2 % contre 45,8 % ; 2025 : 54,2 % contre 66,7 %), d'où l'intérêt du walk-forward.

## 4. Décisions

- **Modèle retenu : XGBoost avec les 13 features de l'étape 7b.** Il est le plus simple et le meilleur en walk-forward.
- Les features 7c restent disponibles avec l'option `--with-extra`, non retenues par défaut.
- On arrête d'ajouter des features sur ces 173 courses : continuer reviendrait à ajuster le modèle sur le jeu de test.
- Limites connues : la stratégie de course, la météo, les incidents et les arrêts aux stands ne sont pas dans les données.

## 5. Fichiers de l'étape

| Fichier | Rôle |
|---|---|
| `ml/features.py` | features, split temporel, walk-forward, métriques, bootstrap |
| `ml/train.py` | entraînement, comparaison des variantes, suivi MLflow |
| `ingestion/producer_qualifying.py` | qualifications Jolpica → Kafka |
| `streaming/qualifying_to_silver.py` | Bronze → Silver des qualifications |
| `streaming/silver_to_gold.py` | Gold avec features de qualification et features 7c |
| `streaming/kafka_to_bronze.py`, `streaming/bronze_to_silver.py` | généralisation + correction `Lapped` |
| `scripts/setup.ps1` | installation et régénération complète du pipeline |
| `tests/test_ml_features.py`, `tests/test_producer_qualifying.py` | 20 tests unitaires au total |

## 6. Reproduire

Depuis la racine du dépôt, dans PowerShell :

```powershell
.\.venv\Scripts\python.exe -m pytest -q
docker compose up -d
docker exec kafka /opt/kafka/bin/kafka-topics.sh --create --if-not-exists --topic f1.qualifying --bootstrap-server localhost:9092
.\.venv\Scripts\python.exe ingestion/producer_qualifying.py --start 2015 --end 2025
docker exec spark /opt/spark/bin/spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1 --conf spark.jars.ivy=/tmp/.ivy2 /opt/app/streaming/kafka_to_bronze.py f1.qualifying qualifying
docker exec spark /opt/spark/bin/spark-submit /opt/app/streaming/qualifying_to_silver.py
docker exec spark /opt/spark/bin/spark-submit /opt/app/streaming/silver_to_gold.py
.\.venv\Scripts\python.exe ml/train.py
.\.venv\Scripts\mlflow ui --backend-store-uri sqlite:///mlflow.db     # puis http://localhost:5000
```

Options de `train.py` : `--no-quali` (ignorer les qualifications), `--with-extra` (ajouter les features 7c),
`--balance` (pondérer les classes), `--no-register` (ne pas enregistrer le modèle).

## 7. Prochaines étapes

- **Étape 8** : Airflow avec 5 DAGs (`ingest_historical`, `consume_kafka`, `feature_engineering`, `train_model`,
  `predict_next_race`), MLflow et Airflow dans `docker-compose.yml`.
- **Étape 9** : dashboard Streamlit (à partir de `data/predictions/test_predictions.parquet` et du modèle MLflow),
  notebook d'analyse, README final avec schéma d'architecture.
- Pistes ultérieures : données météo et pneus (OpenF1, saisons récentes), ingestion de la saison 2026 pour prédire la
  prochaine course.
