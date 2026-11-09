# Étape 8 : orchestration Airflow + MLflow dans Docker + CI GitHub Actions

## 1. Objectif
Remplacer les lancements manuels (producers, spark-submit, entraînement) par un pipeline **orchestré** et **testé
automatiquement** à chaque push.

## 2. Architecture

```
ingest_historical ──▶ consume_kafka ──▶ feature_engineering ──▶ train_model ──▶ predict_next_race
 (API → Kafka)       (Kafka→Bronze→Silver)  (Silver→Gold + qualité)  (XGBoost+MLflow)   (modèle → prédictions)
      DS_KAFKA            DS_SILVER              DS_GOLD                 DS_MODEL
```

Les 5 DAGs sont chaînés par des **Datasets Airflow** : un DAG démarre tout seul quand le précédent a publié son
Dataset. Seul `ingest_historical` se lance à la main (« Trigger DAG w/ config », paramètres `start_year` / `end_year`).
Si le contrôle qualité Gold (`ml/check_gold.py`) échoue, `DS_GOLD` n'est pas publié et l'entraînement ne démarre pas.

| DAG | Tâches | Fichiers |
|---|---|---|
| `ingest_historical` | create_topics → producers résultats + qualifs → publish | `ingestion/*.py` |
| `consume_kafka` | Kafka→Bronze (×2) → Silver (×2) → publish | `streaming/*.py` |
| `feature_engineering` | silver_to_gold → check_gold | `streaming/silver_to_gold.py`, `ml/check_gold.py` |
| `train_model` | train | `ml/train.py` |
| `predict_next_race` | predict | `ml/predict.py` |

## 3. Services Docker (`docker-compose.yml`)
| Service | URL (PC) | Rôle |
|---|---|---|
| Kafka UI | http://localhost:8080 | voir les topics |
| **Airflow** | http://localhost:8081 (admin / `AIRFLOW_ADMIN_PASSWORD`, défaut `admin`) | orchestration |
| **MLflow** | http://localhost:5001 | suivi des runs + registry `raceflow-winner-xgb` |

Choix techniques :
- Airflow 2.10.5 dans **un seul conteneur** (SQLite + SequentialExecutor : suffisant en développement, **pas** pour la production).
- Les scripts du projet tournent dans un environnement Python séparé (`/opt/raceflow-venv`, pyspark 3.5.1 en mode local +
  Java 17, mlflow 3.18.0) : aucun conflit de dépendances avec Airflow.
- Le serveur MLflow reçoit `--allowed-hosts` (sinon erreur 403 « Invalid Host header » pour le nom `mlflow`).
- Ports 5001 et 8081 pour ne pas gêner `mlflow ui` (5000) et Kafka UI (8080).
- `.gitattributes` force les fins de ligne LF (les scripts bash cassent en CRLF sous Windows).

## 4. GitHub Actions (`.github/workflows/ci.yml`)
Déclenché à chaque push, pull request ou à la main.

| Job | Ce qu'il vérifie |
|---|---|
| `lint` | `ruff check .` |
| `test` | `pytest tests/` (producers, features ML, check_gold) |
| `dags` | les 5 DAGs se chargent avec le vrai Airflow et s'enchaînent bien (`tests/test_dags.py`) |
| `pipeline-smoke` | Bronze synthétique → Silver → Gold → check_gold → train (MLflow) → predict, **sans Kafka ni API** |
| `docker` | `docker compose config` + build des images Airflow, MLflow et dashboard |

`scripts/make_synthetic_data.py` fabrique de fausses courses (avec doublons) : elles n'ont **aucune valeur sportive**,
elles servent seulement à prouver que le pipeline tourne de bout en bout. Dependabot (`.github/dependabot.yml`) propose
chaque semaine les mises à jour pip / Actions / images Docker.

## 5. Lancer sur le PC
```powershell
docker compose up -d --build        # premier build : environ 10 min (Airflow + Java + dépendances)
docker compose ps                   # kafka, kafka-ui, spark, mlflow, airflow (+ dashboard) : Up
```
1. Ouvrir http://localhost:8081 (admin / admin). Les 5 DAGs sont actifs dès la création.
2. `ingest_historical` → **Trigger DAG w/ config** → `start_year` 2015, `end_year` 2025.
3. Les 4 autres DAGs démarrent tout seuls ; voir les runs dans http://localhost:5001.
4. Résultat : `data/predictions/latest_race_predictions.parquet`.

Tests locaux : `ruff check .` puis `pytest tests/`.

⚠️ Un autre projet (CrimeWatch, SkyWatch...) qui utilise déjà les noms `kafka` / `kafka-ui` ou les ports 9092 / 8080
bloque le démarrage (« container name already in use » ou « port is already allocated »). L'arrêter, ou renommer
ses conteneurs (`docker stop kafka` puis `docker rename kafka crimewatch-kafka`), avant `docker compose up`.

## 6. Validation sur un PC (8 novembre 2026, Mohamed)
Première construction des images et premier lancement complet par Airflow, sur un PC Windows (Docker Desktop).

| Vérification | Résultat |
|---|---|
| Construction des images Airflow et MLflow | ✅ (environ 10 min la première fois) |
| Les 5 DAGs chargés, sans « Broken DAG » | ✅ |
| `ingest_historical` | ✅ 4 698 messages `f1.results`, 4 684 messages `f1.qualifying` |
| `consume_kafka` : Spark lit Kafka **depuis Airflow** | ✅ |
| `feature_engineering` → `train_model` → `predict_next_race` enchaînés par les Datasets | ✅ |
| Run et version 1 du modèle dans MLflow (http://localhost:5001) | ✅ |

Deux bugs trouvés et corrigés pendant cette validation :

| Bug | Symptôme | Cause | Correction |
|---|---|---|---|
| Airflow ne démarre pas | conteneur `airflow` arrêté (code 2), logs : `invalid choice: '/opt/airflow/start.sh'` | l'entrypoint officiel de l'image Airflow passe la commande à `airflow ...`, sauf si elle commence par `bash` ou `python` | `docker/airflow/Dockerfile` : `CMD ["bash", "/opt/airflow/start.sh"]` |
| `create_topics` échoue | `ImportError: cannot import name 'NoBrokersAvailable' from 'kafka.errors'` | `kafka-python>=2.1.0` a installé la version 3.0, qui a des changements incompatibles | `requirements.txt` : `kafka-python>=2.1.0,<3` |

Résultat du modèle dans Airflow (walk-forward 2018-2025, 173 courses) : **54,9 %** de vainqueurs trouvés contre
**52,0 %** pour « le poleman gagne », écart +2,9 points, IC 95 % [−2,9 ; +8,7]. C'est un peu moins que les 56,1 % du
7 novembre, avec les mêmes données et les mêmes réglages : l'image Docker installe des versions plus récentes des
bibliothèques (XGBoost, pandas). L'écart représente 2 courses sur 173 et la conclusion ne change pas : l'écart avec
la pole n'est pas statistiquement prouvé.

## 7. Limites connues
- `predict_next_race` prédit la **dernière course présente dans Gold**, pas une course à venir (il faudrait construire
  les features à partir de la vraie grille) : prévu ensuite.
- SQLite + SequentialExecutor : une tâche à la fois, à remplacer (Postgres + LocalExecutor) pour un usage réel.
- Les versions des bibliothèques ne sont fixées que par des bornes (`>=`) : un résultat peut varier légèrement d'une
  installation à l'autre (voir section 6). Un fichier de versions figées (`pip freeze`) rendrait le modèle reproductible.

## 8. Étape suivante
Étape 9 : dashboard Streamlit, voir `docs/ETAPE_9_DASHBOARD.md`.
