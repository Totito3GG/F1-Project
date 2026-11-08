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
| `docker` | `docker compose config` + build des images Airflow et MLflow |

`scripts/make_synthetic_data.py` fabrique de fausses courses (avec doublons) : elles n'ont **aucune valeur sportive**,
elles servent seulement à prouver que le pipeline tourne de bout en bout. Dependabot (`.github/dependabot.yml`) propose
chaque semaine les mises à jour pip / Actions / images Docker.

## 5. Lancer sur le PC
```powershell
docker compose up -d --build        # premier build : plusieurs minutes (Airflow + Java + dépendances)
docker compose ps
```
1. Ouvrir http://localhost:8081, activer les 5 DAGs (interrupteur).
2. `ingest_historical` → **Trigger DAG w/ config** → `start_year` 2015, `end_year` 2025.
3. Suivre l'enchaînement dans Airflow ; voir les runs dans http://localhost:5001.
4. Résultat : `data/predictions/latest_race_predictions.parquet`.

Tests locaux : `ruff check .` puis `pytest tests/`.

## 6. Limites connues
- Les images Docker n'ont pas pu être construites dans l'environnement de développement de Claude (accès Docker Hub
  bloqué) : le premier `docker compose up --build` sur le PC (et le job `docker` de la CI) en est la vérification.
  Les scripts, les DAGs et l'intégration MLflow ont, eux, été testés séparément.
- `predict_next_race` prédit la **dernière course présente dans Gold**, pas une course à venir (il faudrait construire
  les features à partir de la vraie grille) : prévu ensuite.
- SQLite + SequentialExecutor : une tâche à la fois, à remplacer (Postgres + LocalExecutor) pour un usage réel.

## 7. Étape suivante
Étape 9 : dashboard Streamlit (lit `data/predictions/*.parquet` et le modèle MLflow), notebook EDA, README final avec schéma d'architecture.
