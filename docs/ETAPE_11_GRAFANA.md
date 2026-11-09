# Étape 11 : monitoring avec Prometheus et Grafana

## 1. Objectif
Le cahier des charges demande que **Grafana monitore les métriques du pipeline : lag Kafka, durée des DAGs, drift du
modèle**. L'étape 11 ajoute un tableau de bord Grafana, mis à jour toutes les 30 secondes.

## 2. Architecture
```
Kafka · Bronze · Airflow · Gold · prédictions · MLflow
                │  lus toutes les 30 s
                ▼
   exporter  (monitoring/exporter.py, port 8000)  ──►  /metrics
                │  « scrape » toutes les 30 s
                ▼
   Prometheus  (port 9090)  : stocke l'historique des métriques (15 jours)
                │
                ▼
   Grafana  (port 3000)  : tableau de bord « RaceFlow : monitoring du pipeline »
```

| Service | Adresse | Rôle |
|---|---|---|
| exporter | http://localhost:8000/metrics | calcule les métriques de RaceFlow, au format Prometheus |
| Prometheus | http://localhost:9090 | collecte et stocke les métriques |
| Grafana | http://localhost:3000 (admin / admin) | affiche le tableau de bord |

La source de données et le tableau de bord sont créés **automatiquement** au démarrage de Grafana (provisioning) :
aucun clic à faire.

## 3. Les métriques

| Groupe | Métrique Prometheus | Calcul |
|---|---|---|
| Sources | `raceflow_source_up{source}` | 1 si l'exporter a pu lire la source (kafka, airflow, gold, predictions, mlflow) |
| **Lag Kafka** | `raceflow_kafka_lag{topic}` | messages dans Kafka (offsets de fin) **moins** messages déjà écrits dans Bronze (dernier offset par partition) |
| | `raceflow_kafka_messages_total{topic}` | messages dans le topic |
| **Durée des DAGs** | `raceflow_dag_last_run_duration_seconds{dag_id}` | durée du dernier run terminé (API REST d'Airflow) |
| | `raceflow_dag_last_run_success{dag_id}` | 1 = succès, 0 = échec |
| | `raceflow_dag_seconds_since_last_success{dag_id}` | temps écoulé depuis le dernier succès |
| | `raceflow_dag_running{dag_id}` | 1 si un run est en cours |
| **Drift de performance** | `raceflow_recent_winner_accuracy{method}` | % de vainqueurs trouvés sur les 10 dernières courses de test, modèle contre « le poleman gagne » |
| | `raceflow_walkforward_metric{metric}` | verdict walk-forward du dernier entraînement (MLflow) |
| **Drift des données** | `raceflow_feature_psi{feature}` | PSI de 5 features : saisons passées contre saison en cours |
| Modèle | `raceflow_model_version`, `raceflow_prediction_age_seconds`, `raceflow_prediction_is_upcoming` | version du registry, âge et type de la dernière prédiction |

### Pourquoi calculer le lag soi-même
L'outil habituel (kafka-exporter) mesure le lag des **consumer groups**. Or Spark Structured Streaming ne commit pas
ses offsets dans Kafka : il les garde dans ses checkpoints. Ce lag vaudrait toujours 0. Comme Bronze garde les
métadonnées Kafka (partition, offset : étape 4), on compare directement Kafka et Bronze.

### Le PSI (Population Stability Index)
On découpe la distribution de référence (saisons passées) en 10 classes (déciles), puis on compare la part de chaque
classe dans la saison en cours : `PSI = Σ (actuel − référence) × ln(actuel / référence)`.
Seuils usuels : **< 0,1 stable**, **0,1 à 0,2 à surveiller**, **> 0,2 drift important**. En 2026 (nouveau règlement
technique), un drift sur les écarts de qualification ou les points des écuries est attendu : c'est exactement ce que
ce panneau doit montrer.

## 4. Fichiers

| Fichier | Rôle |
|---|---|
| `monitoring/metrics.py` | calculs : lag, PSI, précision récente, résumé des runs Airflow (testés, pandas seul) |
| `monitoring/exporter.py` | lit chaque source séparément (une source en panne n'arrête pas les autres) et publie `/metrics` |
| `monitoring/prometheus.yml` | Prometheus lit `exporter:8000` toutes les 30 s |
| `monitoring/grafana/provisioning/` | source de données Prometheus + chargement du tableau de bord |
| `monitoring/grafana/dashboards/raceflow.json` | le tableau de bord (17 panneaux, 4 rangées) |
| `docker/exporter/Dockerfile`, `monitoring/requirements.txt` | image de l'exporter (Python 3.11, prometheus-client) |
| `docker-compose.yml` | services `exporter`, `prometheus`, `grafana` ; API REST d'Airflow activée (basic auth) |
| `tests/test_monitoring_metrics.py` | 8 tests des calculs |

## 5. Lancer
```powershell
docker compose up -d --build        # construit l'exporter, télécharge Prometheus et Grafana, recrée Airflow
docker compose ps                   # 9 services : Up
```
Puis ouvrir http://localhost:3000 (admin / admin) : le tableau de bord « RaceFlow : monitoring du pipeline » s'ouvre.
Pour vérifier l'exporter seul : http://localhost:8000/metrics. Pour vérifier Prometheus : http://localhost:9090,
menu Status → Targets, la cible `raceflow` doit être « UP ».

## 6. Limites
- Prometheus garde 15 jours d'historique (valeur par défaut) : suffisant pour suivre quelques week-ends de course.
- Pas d'alertes (e-mail, Slack) : les seuils sont visibles par les couleurs du tableau de bord. Grafana Alerting
  pourrait les ajouter.
- Le drift de performance porte sur 10 courses : chiffre très variable (1 course = 10 points), à lire avec le verdict
  walk-forward.
- Le lag est calculé par rapport à Bronze, pas par rapport à la position interne de Spark (checkpoints).
