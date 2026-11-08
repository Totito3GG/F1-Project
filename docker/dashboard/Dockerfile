# Dashboard Streamlit de RaceFlow (étape 9).
# Image légère : pas de Spark ni de MLflow. Les métriques MLflow sont lues par son API REST (dashboard/data.py).
FROM python:3.11-slim

WORKDIR /app
COPY dashboard/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Le code (dashboard/) et les données (data/) sont montés par docker-compose.yml :
# une modification de app.py est visible sans reconstruire l'image (bouton « Rerun » de Streamlit).
EXPOSE 8501
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--browser.gatherUsageStats=false"]
