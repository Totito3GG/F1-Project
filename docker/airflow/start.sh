#!/usr/bin/env bash
# Démarrage d'Airflow en un seul conteneur : base SQLite, scheduler en arrière-plan, webserver au premier plan.
# Réservé au développement local (SequentialExecutor : une tâche à la fois, suffisant pour ce pipeline linéaire).
set -e

airflow db migrate

# Compte administrateur local. Le mot de passe vient de AIRFLOW_ADMIN_PASSWORD (voir .env.example).
# L'erreur « utilisateur existe déjà » aux redémarrages est ignorée.
airflow users create \
    --username admin \
    --password "${AIRFLOW_ADMIN_PASSWORD:-admin}" \
    --firstname Raceflow --lastname Admin \
    --role Admin --email admin@raceflow.local || true

airflow scheduler &
exec airflow webserver --port 8080
