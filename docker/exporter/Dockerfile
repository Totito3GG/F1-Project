# Exporter Prometheus de RaceFlow (étape 11) : calcule les métriques du pipeline et les publie sur /metrics.
FROM python:3.11-slim

WORKDIR /app
COPY monitoring/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Le code (monitoring/) et les données (data/) sont montés par docker-compose.yml.
EXPOSE 8000
CMD ["python", "monitoring/exporter.py"]
