"""
RaceFlow : dashboard Streamlit (étapes 9 et 10).

4 onglets :
    Prédiction   probabilités de victoire de chaque pilote pour le prochain GP, avant la course (étape 10),
                 ou pour la dernière course courue s'il n'y a pas de course à venir (ml/predict.py)
    Historique   le modèle face à la règle « le poleman gagne », course par course, sur les saisons de test
    Pilotes      statistiques par saison tirées de la table Gold
    Modèle       métriques du dernier entraînement, lues dans MLflow

Lancer :
    Docker (recommandé) : docker compose up -d dashboard   puis http://localhost:8501
    Sur le PC           : streamlit run dashboard/app.py
"""
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dashboard.data import (  # noqa: E402
    GOLD_PATH,
    LATEST_PATH,
    MLFLOW_URL,
    TEST_PATH,
    driver_history,
    driver_stats,
    is_upcoming_race,
    latest_run,
    pretty,
    race_label,
    race_summary,
    read_parquet,
    season_scores,
)

# Adresse de MLflow à ouvrir dans le navigateur du PC (différente de MLFLOW_URL quand le dashboard tourne dans Docker)
MLFLOW_UI = os.getenv("MLFLOW_UI_URL", "http://localhost:5001")
WINNER_COLOR, OTHER_COLOR = "#e10600", "#9aa0a6"  # rouge F1 pour le vainqueur réel, gris pour les autres

st.set_page_config(page_title="RaceFlow", page_icon="🏎️", layout="wide")


# ------------------------------------------------------------------ chargement (mis en cache 60 s)
@st.cache_data(ttl=60)
def load(path: str) -> pd.DataFrame | None:
    return read_parquet(Path(path))


@st.cache_data(ttl=60)
def load_run() -> dict | None:
    return latest_run(MLFLOW_URL)


def pct(x: float | None) -> str:
    return "—" if x is None or pd.isna(x) else f"{x:.1%}"


def missing(what: str, dag: str) -> None:
    st.info(f"Données introuvables ({what}) : lancer le pipeline dans Airflow (DAG `{dag}`), puis recharger la page.")


def proba_chart(race: pd.DataFrame, top: int = 10):
    """Barres horizontales : probabilité de victoire des `top` premiers pilotes, vainqueur réel en rouge."""
    d = race.sort_values("win_prob", ascending=False).head(top).copy()
    d["pilote"] = d["driver_id"].map(pretty)
    # course à venir (étape 10) : is_winner vide -> tous en gris
    d["résultat"] = d["is_winner"].fillna(0).map({1: "Vainqueur réel", 0: "Autres"}) if "is_winner" in d else "Autres"
    fig = px.bar(d, x="win_prob", y="pilote", orientation="h", color="résultat",
                 color_discrete_map={"Vainqueur réel": WINNER_COLOR, "Autres": OTHER_COLOR},
                 text=d["win_prob"].map(lambda p: f"{p:.0%}"),
                 labels={"win_prob": "Probabilité de victoire", "pilote": ""})
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, xaxis_tickformat=".0%",
                      height=40 * len(d) + 120, margin={"l": 0, "r": 0, "t": 10, "b": 0}, legend_title="")
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(showlegend=d["résultat"].nunique() > 1)   # course à venir : pas de légende
    return fig


def race_table(race: pd.DataFrame) -> pd.DataFrame:
    """Tableau d'une course : rang prédit, pilote, écurie, grille, probabilité, arrivée réelle."""
    t = race.sort_values("pred_rank")
    return pd.DataFrame({
        "Rang prédit": t["pred_rank"],
        "Pilote": t["driver_id"].map(pretty),
        "Écurie": t["constructor_id"].map(pretty),
        # grid vide (null) : donnée absente de l'API (vu en 2026) ; 0 : départ des stands
        "Grille": t["grid"].map(lambda g: "—" if pd.isna(g) else ("stands" if g == 0 else f"{int(g)}")),
        "Probabilité": t["win_prob"].map(lambda p: f"{p:.1%}"),
        "Arrivée réelle": t["finish_position"].map(lambda p: "—" if pd.isna(p) else f"{int(p)}"),
    })


# ------------------------------------------------------------------ en-tête
st.title("🏎️ RaceFlow")
st.caption("Prédiction du vainqueur d'un Grand Prix de F1 · API Jolpica → Kafka → Spark → XGBoost → MLflow")

tab_pred, tab_hist, tab_drivers, tab_model = st.tabs(["🏁 Prédiction", "📊 Historique", "👤 Pilotes", "🧪 Modèle"])

# ------------------------------------------------------------------ 1. Prédiction
with tab_pred:
    latest = load(str(LATEST_PATH))
    if latest is None:
        missing("prédictions de la dernière course", "predict_next_race")
    else:
        head = latest.iloc[0]
        upcoming = is_upcoming_race(latest)
        title = race_label(head["season"], head["round"], head["race_name"])
        st.subheader(f"🔮 Prochain GP : {title}" if upcoming else title)
        st.caption(f"Modèle `raceflow-winner-xgb` version {int(head['model_version'])} (registry MLflow)")

        favori = latest.loc[latest["pred_rank"] == 1].iloc[0]
        pole = latest.loc[latest["grid"] == 1, "driver_id"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Favori du modèle", pretty(favori["driver_id"]), f"{favori['win_prob']:.0%} de chances",
                  delta_color="off")
        c2.metric("Pole position" if upcoming else "Poleman", pretty(pole.iloc[0] if len(pole) else None))
        if upcoming:
            date = pd.to_datetime(head["race_date"])
            c3.metric("Course", "à venir", f"le {date:%d/%m/%Y}" if pd.notna(date) else None, delta_color="off")
        else:
            winner = latest.loc[latest["is_winner"] == 1, "driver_id"]
            if len(winner):
                ok = winner.iloc[0] == favori["driver_id"]
                c3.metric("Vainqueur réel", pretty(winner.iloc[0]),
                          "prédiction juste" if ok else "prédiction fausse",
                          delta_color="normal" if ok else "inverse")

        left, right = st.columns([3, 2])
        left.plotly_chart(proba_chart(latest), width="stretch", key="chart_latest")
        right.dataframe(race_table(latest), hide_index=True, width="stretch")
        if upcoming:
            st.caption("Prédiction faite AVANT la course, à partir des qualifications du samedi. Grille = ordre des "
                       "qualifications : les pénalités de grille annoncées ensuite ne sont pas prises en compte.")
        else:
            st.caption("Course déjà courue : aucune course à venir n'est dans la table Gold (le DAG ingest_historical "
                       "tourne chaque samedi à 22:00 UTC, après les qualifications).")

# ------------------------------------------------------------------ 2. Historique
with tab_hist:
    test = load(str(TEST_PATH))
    if test is None:
        missing("prédictions des saisons de test", "train_model")
    else:
        summary = race_summary(test)
        scores = season_scores(summary)
        total = scores.iloc[-1]
        st.subheader("Le modèle face à la règle « le poleman gagne »")
        c1, c2, c3 = st.columns(3)
        c1.metric("Courses testées", int(total["courses"]))
        c2.metric("Vainqueur trouvé par le modèle", pct(total["modele"]))
        c3.metric("Vainqueur = poleman", pct(total["pole"]),
                  f"{(total['modele'] - total['pole']) * 100:+.1f} pts pour le modèle", delta_color="off")

        long = scores.melt(id_vars="season", value_vars=["modele", "pole"], var_name="méthode", value_name="taux")
        long["méthode"] = long["méthode"].map({"modele": "Modèle XGBoost", "pole": "Le poleman gagne"})
        fig = px.bar(long, x="season", y="taux", color="méthode", barmode="group",
                     text=long["taux"].map(lambda v: f"{v:.0%}"),
                     labels={"season": "Saison", "taux": "Vainqueurs trouvés", "méthode": ""},
                     color_discrete_sequence=[WINNER_COLOR, OTHER_COLOR])
        fig.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1], xaxis_type="category", height=360,
                          margin={"t": 10})
        st.plotly_chart(fig, width="stretch", key="chart_seasons")
        st.caption("Ces chiffres portent sur les seules saisons de test. Le verdict statistique (walk-forward, "
                   "toutes les saisons depuis 2018) est dans l'onglet Modèle.")

        st.subheader("Course par course")
        labels = {r.race_idx: race_label(r.season, r["round"], r.race_name) for _, r in summary.iterrows()}
        race_idx = st.selectbox("Course", options=list(labels)[::-1], format_func=labels.get)
        race = test[test["race_idx"] == race_idx]
        left, right = st.columns([3, 2])
        left.plotly_chart(proba_chart(race), width="stretch", key="chart_race")
        right.dataframe(race_table(race), hide_index=True, width="stretch")

        with st.expander("Toutes les courses"):
            st.dataframe(pd.DataFrame({
                "Course": summary["race_idx"].map(labels),
                "Favori du modèle": summary["favori_modele"].map(pretty),
                "Poleman": summary["poleman"].map(pretty),
                "Vainqueur": summary["vainqueur"].map(pretty),
                "Modèle": summary["modele_juste"].map({1: "✅", 0: "❌"}),
                "Pole": summary["pole_juste"].map({1: "✅", 0: "❌"}),
            }), hide_index=True, width="stretch")

# ------------------------------------------------------------------ 3. Pilotes
with tab_drivers:
    gold = load(str(GOLD_PATH))
    if gold is None:
        missing("table Gold", "feature_engineering")
    else:
        seasons = sorted(gold["season"].unique(), reverse=True)
        season = st.selectbox("Saison", seasons)
        stats = driver_stats(gold, season)
        st.dataframe(pd.DataFrame({
            "Pilote": stats["driver_id"].map(pretty),
            "Écurie": stats["ecurie"].map(pretty),
            "Courses": stats["courses"],
            "Victoires": stats["victoires"],
            "Podiums": stats["podiums"],
            "Poles": stats["poles"],
            "Grille moy.": stats["grille_moy"].round(1),
            "Arrivée moy.": stats["arrivee_moy"].round(1),
        }), hide_index=True, width="stretch")

        driver = st.selectbox("Pilote", stats["driver_id"], format_func=pretty)
        hist = driver_history(gold, driver, season)
        long = hist.melt(id_vars=["round", "race_name"], value_vars=["grid", "finish_position"],
                         var_name="type", value_name="position")
        long["type"] = long["type"].map({"grid": "Grille de départ", "finish_position": "Arrivée"})
        fig = px.line(long, x="round", y="position", color="type", markers=True, hover_data=["race_name"],
                      labels={"round": "Manche", "position": "Position", "type": ""},
                      color_discrete_sequence=[OTHER_COLOR, WINNER_COLOR])
        fig.update_yaxes(autorange="reversed")  # la 1re place en haut
        fig.update_layout(height=380, margin={"t": 10})
        st.plotly_chart(fig, width="stretch", key="chart_driver")
        st.caption("Point manquant : départ des stands (grille) ou pilote non classé (arrivée).")

# ------------------------------------------------------------------ 4. Modèle (MLflow)
with tab_model:
    run = load_run()
    if run is None:
        st.info(f"MLflow injoignable ({MLFLOW_URL}) ou aucun entraînement : lancer le DAG `train_model`.")
    else:
        m = run["metrics"]
        st.subheader("Verdict : validation walk-forward")
        n_wf = m.get("wf_n_races")
        st.caption(f"Dernier run « {run['run_name']} » du {run['start_time']:%d/%m/%Y %H:%M} UTC · "
                   f"[ouvrir dans MLflow]({MLFLOW_UI})")
        c1, c2, c3 = st.columns(3)
        c1.metric("Vainqueur trouvé par le modèle", pct(m.get("wf_winner_accuracy")),
                  f"sur {int(n_wf)} courses" if n_wf else None, delta_color="off")
        c2.metric("Vainqueur = poleman", pct(m.get("wf_pole_baseline_accuracy")))
        diff, lo, hi = m.get("wf_diff_vs_pole"), m.get("wf_diff_ci_low"), m.get("wf_diff_ci_high")
        if diff is not None:
            c3.metric("Écart modèle − pole", f"{diff * 100:+.1f} pts",
                      f"IC 95 % [{lo * 100:+.1f} ; {hi * 100:+.1f}]", delta_color="off")
            if lo > 0:
                st.success("La borne basse de l'intervalle est positive : le modèle bat vraiment la règle de la pole.")
            else:
                st.warning("L'intervalle de confiance contient 0 : l'écart avec la règle de la pole n'est pas "
                           "statistiquement prouvé.")

        variants = [("base", "Base (étape 7)"), ("quali", "+ qualifications (7b)"),
                    ("extra", "+ features 7c"), ("ranker", "Classement + features 7c")]
        rows = [{"Variante": label,
                 "Vainqueur trouvé": pct(m.get(f"wf_{key}_winner_accuracy")),
                 "Écart vs pole": f"{m[f'wf_{key}_diff_vs_pole'] * 100:+.1f} pts",
                 "IC 95 %": f"[{m[f'wf_{key}_diff_ci_low'] * 100:+.1f} ; {m[f'wf_{key}_diff_ci_high'] * 100:+.1f}]",
                 "Podium": pct(m.get(f"wf_{key}_podium_accuracy"))}
                for key, label in variants if f"wf_{key}_winner_accuracy" in m]
        if rows:
            st.subheader("Comparaison des variantes (mêmes courses)")
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

        with st.expander("Paramètres du modèle"):
            p = run["params"]
            features = p.pop("features", "")
            st.write(f"**Features ({len(features.split(','))}) :** {features.replace(',', ', ')}")
            st.dataframe(pd.DataFrame(sorted(p.items()), columns=["Paramètre", "Valeur"]),
                         hide_index=True, width="stretch")
