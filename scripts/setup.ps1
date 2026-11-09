# RaceFlow - installation et regeneration complete du pipeline sous Windows (PowerShell)
# Prerequis : Git, Python 3.11+, Docker Desktop lance.
# Lancer depuis la racine du depot :
#     powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
# Options :
#     -SkipIngestion   ne relance pas le producer (Kafka contient deja les messages)
#     -SkipTrain       s'arrete a la table Gold
param(
    [int]$Start = 2015,
    [int]$End = 2025,
    [switch]$SkipIngestion,
    [switch]$SkipTrain
)
# "Continue" : sous Windows PowerShell 5.1, "Stop" transforme le stderr de docker/spark en erreur.
# Les echecs sont detectes par Check (code de sortie).
$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)

function Step($msg) { Write-Host "`n=== $msg" -ForegroundColor Cyan }
function Check($what) { if ($LASTEXITCODE -ne 0) { throw "Echec : $what (code $LASTEXITCODE)" } }

Step "1. Environnement Python"
if (-not (Test-Path ".venv")) { python -m venv .venv; Check "creation du venv" }
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install --upgrade pip -q
& $py -m pip install -r requirements.txt -q; Check "pip install"
if (-not (Test-Path ".env")) { Copy-Item ".env.example" ".env" }
& $py -m pytest -q; Check "tests unitaires"

Step "2. Docker : Kafka, Kafka UI, Spark"
docker info *> $null; Check "Docker Desktop n'est pas lance"
docker compose up -d; Check "docker compose up"
Write-Host "Attente de Kafka..."
for ($i = 0; $i -lt 30; $i++) {
    docker exec kafka /opt/kafka/bin/kafka-topics.sh --list --bootstrap-server localhost:9092 *> $null
    if ($LASTEXITCODE -eq 0) { break }
    Start-Sleep -Seconds 2
}
Check "Kafka ne repond pas"
foreach ($t in "f1.results", "f1.laps", "f1.qualifying") {
    docker exec kafka /opt/kafka/bin/kafka-topics.sh --create --if-not-exists --topic $t --bootstrap-server localhost:9092
    Check "creation du topic $t"
}

if (-not $SkipIngestion) {
    Step "3. Producer Jolpica -> Kafka ($Start-$End, ~10 min)"
    & $py ingestion/producer_results.py --start $Start --end $End; Check "producer results"
    & $py ingestion/producer_qualifying.py --start $Start --end $End; Check "producer qualifying"
}

Step "4. Spark : Kafka -> Bronze -> Silver -> Gold"
docker exec spark /opt/spark/bin/spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1 --conf spark.jars.ivy=/tmp/.ivy2 /opt/app/streaming/kafka_to_bronze.py
Check "kafka_to_bronze"
docker exec spark /opt/spark/bin/spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1 --conf spark.jars.ivy=/tmp/.ivy2 /opt/app/streaming/kafka_to_bronze.py f1.qualifying qualifying
Check "kafka_to_bronze qualifying"
docker exec spark /opt/spark/bin/spark-submit /opt/app/streaming/bronze_to_silver.py; Check "bronze_to_silver"
docker exec spark /opt/spark/bin/spark-submit /opt/app/streaming/qualifying_to_silver.py; Check "qualifying_to_silver"
docker exec spark /opt/spark/bin/spark-submit /opt/app/streaming/silver_to_gold.py; Check "silver_to_gold"
& $py -c "import pandas as pd; print('Lignes Gold :', len(pd.read_parquet('data/gold/features')))"

if (-not $SkipTrain) {
    Step "5. Etape 7 : entrainement XGBoost + MLflow"
    & $py ml/train.py; Check "train"
    Write-Host "`nRuns MLflow : .\.venv\Scripts\mlflow ui --backend-store-uri sqlite:///mlflow.db  puis http://localhost:5000"
}
Write-Host "`nTermine. Kafka UI : http://localhost:8080" -ForegroundColor Green
