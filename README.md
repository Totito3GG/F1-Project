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

## Structure du dépôt

```
ingestion/      producers Python : API → Kafka
streaming/      jobs Spark : Kafka → Data Lake
airflow/dags/   DAGs d'orchestration
data/           Data Lake local (bronze / silver / gold), non versionné
ml/             feature engineering, entraînement, MLflow
dashboard/      application Streamlit
tests/          tests unitaires
docs/           schémas, captures
```

## Avancement

- [x] Étape 1 : structure du projet
- [ ] Étape 2 : Docker Compose (Kafka)
- [ ] Étape 3 : Producer Python (API → Kafka)
- [ ] Étape 4 : Spark (Kafka → Bronze)
- [ ] Étape 5 : Silver / Gold
- [ ] Étape 6 : Airflow
- [ ] Étape 7 : Machine Learning + MLflow
- [ ] Étape 8 : Dashboard Streamlit
- [ ] Étape 9 : Tests + CI

## Équipe

Mustapha HAMDAOUI ALAOUI · Yassir · Mohamed (EMSI Rabat)
