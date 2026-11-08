# Étape 10 : prédire le prochain GP (avant la course)

## 1. Objectif
Le cahier des charges demande que le dashboard affiche **la prédiction du prochain GP**. Jusqu'à l'étape 9, la table
Gold ne contenait que des courses déjà courues (elle est construite à partir des résultats) : la « prédiction » portait
sur la dernière course de 2025, déjà courue. L'étape 10 prédit une course **avant** qu'elle ait lieu.

## 2. L'idée
Le samedi soir, les **qualifications** du Grand Prix sont connues, mais pas encore ses **résultats**. Or le modèle n'a
besoin que de ce qui est connu avant le départ : la grille, les temps de qualification et l'historique de chaque pilote.

Pour chaque course qui a des qualifications et qui vient **après la dernière course courue**, `silver_to_gold.py` crée
une ligne par pilote, marquée `is_upcoming = 1` :

| Colonne | Valeur pour une course à venir |
|---|---|
| `grid` | place en qualification (`quali_position`) |
| features pilote / écurie | calculées par **les mêmes fenêtres Spark** que pour les courses passées (`rowsBetween(-5, -1)`...) |
| `q1_gap_pct`, `q3_gap_pct` | temps de qualification de la course |
| `is_winner`, `is_podium`, `finish_position` | vides (pas encore connus) |

Les fenêtres s'arrêtent toutes à la course précédente : ces lignes sont calculées exactement comme celles d'une course
passée, sans fuite possible.

## 3. Ce qui a changé

| Fichier | Changement |
|---|---|
| `ingestion/producer_qualifying.py` | les messages portent aussi `race_name`, `circuit_id`, `race_date` |
| `streaming/qualifying_to_silver.py` | garde ces 3 colonnes dans Silver |
| `streaming/silver_to_gold.py` | crée les lignes « course à venir » ; nouvelle colonne `is_upcoming` |
| `ml/features.py` | `load_gold()` ne renvoie que les courses courues (entraînement, évaluation, contrôle qualité) ; `load_gold(include_upcoming=True)` garde aussi les courses à venir |
| `ml/predict.py` | `--race next` (défaut) : la prochaine course à venir si Gold en contient une, sinon la dernière courue |
| `airflow/dags/ingest_historical.py` | lancé **chaque samedi à 22:00 UTC** (qualifications finies, course pas encore courue) ; saisons par défaut : 2015 → saison en cours |
| `dashboard/app.py`, `dashboard/data.py` | onglet Prédiction : « 🔮 Prochain GP », pole position, date de la course, pas de vainqueur réel |
| `scripts/make_synthetic_data.py` | la dernière saison a une manche de plus, qualifiée mais pas courue (testée par la CI) |
| `tests/` | tests des nouveaux champs, de la course à venir synthétique, du dashboard et du DAG |

`train.py` et `check_gold.py` n'ont pas changé : ils utilisent `load_gold()`, qui ignore maintenant les courses à venir.

## 4. Le déroulement d'un week-end
```
samedi : qualifications ──► 22:00 UTC : ingest_historical (2015 → saison en cours)
   ──► consume_kafka ──► feature_engineering (Gold + lignes « course à venir »)
   ──► train_model (courses courues seulement) ──► predict_next_race (la course à venir)
   ──► dashboard : « 🔮 Prochain GP »
dimanche : la course ──► samedi suivant : ses résultats entrent dans Gold, la course suivante devient « à venir »
```
À la main (sans attendre le samedi) : Airflow → `ingest_historical` → **Trigger DAG w/ config** (`start_year` 2015,
`end_year` 2026).

## 5. Vérification sur données synthétiques
6 saisons (2015-2020), la manche 21 de 2020 a des qualifications mais pas de résultats :

```
Course(s) à venir (qualifications sans résultats) : [202021]
Gold : 2420 lignes, 31 colonnes
  dont lignes de course à venir : 20
Gold sain                                   (check_gold : 2 400 lignes, courses courues seulement)
Gold : 2400 lignes | train 2015-2018 | test 2019-2020
Course Grand Prix 21 (2020, manche 21) : À VENIR (avant la course) | modèle v1
```
Contrôle anti-fuite : pour le poleman de la manche 21, `drv_points_season` = 208 = somme exacte de ses points des
manches 1 à 20, et `drv_avg_pos_last5` = 7,4 = moyenne de ses 5 dernières arrivées.

## 6. Limites
- **Grille = ordre des qualifications.** Les pénalités de grille annoncées après les qualifications ne sont pas prises
  en compte (`quali_grid_delta` = 0 pour une course à venir).
- **Saison 2026 = nouveau règlement technique** (nouvelles voitures, nouvelles écuries) : le modèle a appris surtout sur
  2015-2025 ; ses prédictions 2026 sont plus incertaines. Les courses 2026 courues entrent dans l'entraînement chaque
  semaine.
- Les week-ends **sprint** : seules les qualifications du Grand Prix sont utilisées (pas la course sprint).
- Le DAG du samedi recharge toutes les saisons depuis 2015 (environ 3 min) : simple et robuste, mais pas optimal.
- Si le PC est éteint le samedi soir, lancer `ingest_historical` à la main avant la course.
