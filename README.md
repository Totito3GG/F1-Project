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
  Spark (PySpark) ──► Data Lake  Bronze → Silver → Gold  (Parquet)
        │
        ▼
  Feature engineering ──► XGBoost ──► MLflow (runs + registry)
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
| Stockage | Parquet | Data lake Bronze / Silver / Gold |
| Orchestration | Apache Airflow | DAGs d'ingestion, transformation, entraînement |
| ML | XGBoost, MLflow | Prédiction du vainqueur, suivi des expériences, registry des modèles |
| Dashboard | Streamlit, Plotly | Prédictions, comparaison avec la pole, stats pilotes, verdict du modèle |
| Infra | Docker Compose | Lancement en une commande |

## Avancement

- [x] 1. Structure du projet
- [x] 2. Kafka + Kafka UI (Docker Compose)
- [x] 3. Producer Jolpica -> Kafka (+ tests, CI)
- [x] 4. Spark : Kafka -> Bronze
- [x] 5. Bronze -> Silver (parsing, dedup)
- [x] 6. Silver -> Gold (features sans fuite)
- [x] 7. XGBoost + MLflow, validation walk-forward (`docs/ETAPE_7_ML.md`)
- [x] 7b. Temps de qualification : features `q1_gap_pct`, `q3_gap_pct`, `quali_grid_delta`
- [x] 8. Orchestration Airflow (5 DAGs chaînés par Datasets) + MLflow dans Docker + CI GitHub Actions, validée sur PC (`docs/ETAPE_8_AIRFLOW.md`)
- [x] 9. Dashboard Streamlit (`docs/ETAPE_9_DASHBOARD.md`)
- [ ] 10. Producer OpenF1 : météo, pneus (optionnel)

## Résultat

Validation walk-forward 2018-2025 (173 courses) : le modèle trouve le vainqueur dans **54,9 %** des courses, contre
**52,0 %** pour la règle « le poleman gagne ». L'intervalle de confiance à 95 % de l'écart, [−2,9 ; +8,7] points,
contient 0 : le modèle ne bat pas la pole de façon prouvée. En F1, la position de départ explique déjà l'essentiel.

## Démarrage rapide (Windows, Docker)

Prérequis : Git, Docker Desktop lancé.

```powershell
git clone https://github.com/Totito3GG/F1-Project.git
cd F1-Project
copy .env.example .env
docker compose up -d --build        # premier build : environ 15 min
docker compose ps                   # 6 services : Up
```

Puis dans Airflow (http://localhost:8081, admin / admin) : DAG `ingest_historical` → **Trigger DAG w/ config**
(`start_year` 2015, `end_year` 2025). Les 4 autres DAGs s'enchaînent tout seuls (environ 10 min). Ouvrir ensuite le
dashboard : http://localhost:8501.

| Service | Adresse | Rôle |
|---|---|---|
| Kafka UI | http://localhost:8080 | topics et messages Kafka |
| Airflow | http://localhost:8081 | orchestration (admin / admin) |
| MLflow | http://localhost:5001 | runs et registry du modèle |
| Dashboard | http://localhost:8501 | prédictions et statistiques |

Sans Airflow : `powershell -ExecutionPolicy Bypass -File scripts\setup.ps1` (venv, tests, Kafka/Spark,
Bronze → Silver → Gold, entraînement).

Tests : `ruff check .` puis `pytest tests/`.

## Structure du dépôt

```
ingestion/      producers Python : API → Kafka
streaming/      jobs Spark : Kafka → Data Lake
airflow/dags/   DAGs d'orchestration
data/           Data Lake local (bronze / silver / gold), non versionné
ml/             feature engineering, entraînement, MLflow
dashboard/      application Streamlit (app.py : affichage, data.py : calculs)
docker/         images Airflow, MLflow, dashboard
tests/          tests unitaires
scripts/        installation Windows (setup.ps1)
docs/           documentation par étape (ETAPE_7, 8, 9)
```

---

Équipe : Mustapha HAMDAOUI ALAOUI, Yassir DRISSI, Mohamed BENHEDDANE · EMSI Rabat
