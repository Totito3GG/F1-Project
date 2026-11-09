# Image Airflow de RaceFlow : Airflow 2.10 + Java (pour Spark) + un environnement Python séparé pour le projet.
# Les dépendances du projet (Spark, XGBoost, MLflow...) vivent dans /opt/raceflow-venv, jamais dans l'Python d'Airflow :
# aucun conflit de versions possible avec Airflow.
FROM apache/airflow:2.10.5-python3.11

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends openjdk-17-jre-headless procps \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /opt/airflow/db /opt/airflow/logs /opt/raceflow \
    && python -m venv /opt/raceflow-venv \
    && chown -R airflow:root /opt/airflow/db /opt/airflow/logs /opt/raceflow /opt/raceflow-venv

COPY docker/airflow/start.sh /opt/airflow/start.sh
# Windows peut convertir les fins de ligne en CRLF, ce qui casserait le script bash : on les normalise.
RUN sed -i 's/\r$//' /opt/airflow/start.sh && chmod 755 /opt/airflow/start.sh

USER airflow
COPY requirements.txt /tmp/requirements.txt
# mlflow est épinglé à la version du serveur (docker/mlflow/Dockerfile) ; pyspark à celle de Spark du projet.
RUN /opt/raceflow-venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt "mlflow==3.18.0" "pyspark==3.5.1"

# L'entrypoint officiel d'Airflow (/entrypoint) transmet la commande à « airflow ... », sauf si elle commence
# par « bash » ou « python ». Sans « bash » devant, il lancerait « airflow /opt/airflow/start.sh » -> erreur.
CMD ["bash", "/opt/airflow/start.sh"]