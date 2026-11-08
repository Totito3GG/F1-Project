# Étape 9 : dashboard Streamlit

## 1. Objectif
Montrer les résultats du pipeline dans une page web : la prédiction de la dernière course, la comparaison du modèle
avec la règle « le poleman gagne », les statistiques des pilotes et le verdict de l'entraînement.

## 2. Les 4 onglets (http://localhost:8501)

| Onglet | Contenu | Données lues | Écrites par |
|---|---|---|---|
| 🏁 Prédiction | favori du modèle, poleman, vainqueur réel ; probabilité de victoire de chaque pilote (vainqueur réel en rouge) | `data/predictions/latest_race_predictions.parquet` | `ml/predict.py` (DAG `predict_next_race`) |
| 📊 Historique | % de vainqueurs trouvés par le modèle et par la pole, par saison ; détail course par course (saisons 2024+) | `data/predictions/test_predictions.parquet` | `ml/train.py` (DAG `train_model`) |
| 👤 Pilotes | par saison : courses, victoires, podiums, poles, grille et arrivée moyennes ; courbe grille / arrivée d'un pilote | `data/gold/features` | `silver_to_gold.py` (DAG `feature_engineering`) |
| 🧪 Modèle | verdict walk-forward (173 courses) avec l'intervalle de confiance ; comparaison des variantes ; paramètres | serveur MLflow (API REST) | `ml/train.py` |

Si une source manque (pipeline pas encore lancé, MLflow arrêté), l'onglet affiche quel DAG lancer au lieu d'une erreur.

## 3. Fichiers

| Fichier | Rôle |
|---|---|
| `dashboard/app.py` | l'interface : les 4 onglets, les graphiques Plotly, le cache (`st.cache_data`, 60 s) |
| `dashboard/data.py` | lecture des Parquet et de MLflow, tous les calculs. **N'importe pas Streamlit** : testable avec pytest |
| `dashboard/requirements.txt` | streamlit, plotly, pandas, pyarrow, requests (rien d'autre) |
| `docker/dashboard/Dockerfile` | image `python:3.11-slim` + ces bibliothèques |
| `tests/test_dashboard_data.py` | 7 tests des calculs (lancés par la CI, job `test`) |

## 4. Choix techniques
- **Un conteneur séparé** (service `dashboard` de `docker-compose.yml`) : le dashboard démarre avec le reste du projet,
  sans dépendre de la version de Python installée sur le PC.
- **Image légère** : pas de Spark, de XGBoost ni de la bibliothèque `mlflow`. Les métriques sont lues par l'API REST
  de MLflow (2 requêtes HTTP : `experiments/get-by-name` puis `runs/search`).
- **Calculs séparés de l'affichage** (`data.py` / `app.py`) : les calculs sont testés automatiquement.
- **Lecture seule** : `./data` est monté en `:ro`, le dashboard ne peut rien modifier dans le data lake.
- **Code monté en volume** (`./dashboard`) : une modification de `app.py` est visible sans reconstruire l'image
  (bouton « Rerun » de Streamlit).
- Deux adresses de MLflow : `MLFLOW_URL=http://mlflow:5000` pour le conteneur, `MLFLOW_UI_URL=http://localhost:5001`
  pour le lien « ouvrir dans MLflow » affiché dans le navigateur.
- Départ des stands (`grid = 0`) : exclu de la grille moyenne et absent de la courbe, sinon il passerait pour une pole.

## 5. Lancer
```powershell
docker compose up -d --build dashboard      # premier build : 2 à 3 min
docker compose ps dashboard                 # Up
```
Puis ouvrir http://localhost:8501. Le pipeline (étape 8) doit avoir tourné au moins une fois.

Sans Docker (Python 3.11+) : `pip install -r dashboard/requirements.txt` puis `streamlit run dashboard/app.py`
(MLflow attendu sur http://localhost:5001, modifiable avec la variable `MLFLOW_URL`).

Tests : `pytest tests/test_dashboard_data.py -q` (7 tests) et `ruff check .`.

## 6. Ce que montre le dashboard (données réelles, 8 novembre 2026)
- **Prédiction**, Abu Dhabi 2025 : Verstappen favori à 56 %, parti en pole, vainqueur → prédiction juste.
- **Historique**, test 2024-2025 (48 courses) : modèle 56,2 %, pole 56,2 %.
  2024 : modèle 54 %, pole 46 % (ex. Canada, Espagne : Verstappen gagne sans partir en pole, le modèle l'avait vu).
  2025 : modèle 58 %, pole 67 %.
- **Modèle**, walk-forward 2018-2025 (173 courses) : 54,9 % contre 52,0 %, écart +2,9 points, IC 95 % [−2,9 ; +8,7].
  L'intervalle contient 0 : le modèle ne bat pas la pole de façon prouvée (bandeau jaune).

## 7. Limites
- La « prédiction » porte sur la dernière course de Gold, déjà courue (même limite que `predict_next_race`).
- Le cache de 60 s : après un nouveau run du pipeline, recharger la page au bout d'une minute.
- Pas d'authentification : le dashboard est fait pour un usage local.
