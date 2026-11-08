# RaceFlow

Plateforme Data Engineering qui collecte les données de Formule 1, les fait passer par un pipeline streaming et batch, puis prédit le vainqueur d'un Grand Prix.

## Architecture

```
Jolpica (ex-Ergast) + OpenF1
        │
        ▼
  Kafka (producers Python)
        │
        ▼
  Spark (PySpark) ──► Data Lake  Bronze → Silver → Gold  (Parquet / Delta)
        │
        ▼
  Feature engineering ──► XGBoost / LightGBM ──► MLflow
        │
        ▼
  Streamlit (prédictions + stats)

  Airflow orchestre le tout · Docker Compose lance le tout
```

## Stack

| Couche | Outil | Rôle |
|---|---|---|
| Sources | Jolpica API, OpenF1 API | Résultats historiques, données de course |
| Streaming | Apache Kafka | Transport des messages entre API et traitements |
| Traitement | Apache Spark (PySpark) | Nettoyage, jointures, agrégations |
| Stockage | Parquet / Delta Lake | Data lake Bronze / Silver / Gold |
| Orchestration | Apache Airflow | DAGs d'ingestion, transformation, entraînement |
| ML | XGBoost, LightGBM, MLflow | Prédiction du vainqueur, suivi des expériences |
| Dashboard | Streamlit | Restitution |
| Infra | Docker Compose | Lancement en une commande |

## Avancement

- [x] 1. Structure du projet
- [x] 2. Kafka + Kafka UI (Docker Compose)
- [x] 3. Producer Jolpica -> Kafka (+ tests, CI)
- [x] 4. Spark : Kafka -> Bronze
- [x] 5. Bronze -> Silver (parsing, dedup)
- [x] 6. Silver -> Gold (features sans fuite)
- [ ] 7. XGBoost + MLflow : code ecrit et teste (`ml/train.py`), a executer sur les vraies donnees
- [ ] 7b. Temps de qualification (`producer_qualifying.py`, `qualifying_to_silver.py`) -> features `q1_gap_pct`, `q3_gap_pct`, `quali_grid_delta` ; `train.py` compare sans/avec qualifications (`--no-quali` pour les ignorer)
- [x] 8. Orchestration Airflow (5 DAGs chaînés par Datasets) + MLflow dans docker-compose + CI GitHub Actions (voir `docs/ETAPE_8_AIRFLOW.md`)
- [ ] 9. Dashboard Streamlit
- [ ] 10. Producer OpenF1 (optionnel)

## Demarrage rapide (Windows)

Prerequis : Git, Python 3.11+, Docker Desktop lance.

```powershell
git clone https://github.com/Totito3GG/F1-Project.git
cd F1-Project
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
```

Le script cree le venv, lance les tests, demarre Kafka/Spark, regenere Bronze -> Silver -> Gold
(environ 15 min) puis entraine le modele. Voir les runs :
`.\.venv\Scripts\mlflow ui --backend-store-uri sqlite:///mlflow.db` puis http://localhost:5000.

## Structure du dépôt

```
ingestion/      producers Python : API → Kafka
streaming/      jobs Spark : Kafka → Data Lake
airflow/dags/   DAGs d'orchestration
data/           Data Lake local (bronze / silver / gold), non versionné
ml/             feature engineering, entraînement, MLflow
dashboard/      application Streamlit
tests/          tests unitaires
scripts/        installation Windows (setup.ps1)
docs/           schémas, captures
```



Mustapha HAMDAOUI ALAOUI
