"""
Entrainement XGBoost + suivi MLflow (params, metriques, modele)
Split temporel : entrainer sur les saisons passees, tester sur les dernieres.
En plus du split final (test 2024+), une validation walk-forward (2018-2025, ~170 courses)
donne un verdict statistiquement fiable face a la baseline "le poleman gagne".

Lancer :
    python ml/train.py
    python ml/train.py --max-depth 3 --learning-rate 0.03 --balance
Voir les runs :
    mlflow ui --backend-store-uri sqlite:///mlflow.db    # puis http://localhost:5000
"""
import argparse
import os
import sys
import tempfile
from pathlib import Path

import mlflow
import mlflow.xgboost
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import log_loss, roc_auc_score
from xgboost import XGBClassifier, XGBRanker

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml.features import (GOLD_PATH, available_features, relevance, softmax_by_race, ID_COLS, PODIUM, TARGET,  # noqa: E402
                         add_race_rank, bootstrap_diff, load_gold, podium_accuracy,
                         pole_baseline_accuracy, pole_hits, pole_podium_baseline,
                         scale_pos_weight, temporal_split, topk_accuracy,
                         walk_forward, winner_accuracy, winner_hits)

ROOT = Path(__file__).resolve().parents[1]
PRED_PATH = ROOT / "data" / "predictions" / "test_predictions.parquet"
EXPERIMENT = "raceflow-winner"
MODEL_NAME = "raceflow-winner-xgb"

load_dotenv()
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")


def parse_args():
    p = argparse.ArgumentParser(description="RaceFlow - entrainement XGBoost")
    p.add_argument("--gold", default=str(GOLD_PATH))
    p.add_argument("--test-from", type=int, default=2024)
    p.add_argument("--wf-from", type=int, default=2018, help="premiere saison testee en walk-forward")
    # modele peu profond : les modeles plus complexes ne font pas mieux sur ces features (cf. analyse)
    p.add_argument("--n-estimators", type=int, default=150)
    p.add_argument("--max-depth", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--subsample", type=float, default=0.8)
    p.add_argument("--colsample-bytree", type=float, default=0.8)
    p.add_argument("--min-child-weight", type=float, default=5)
    p.add_argument("--balance", action="store_true", help="scale_pos_weight (desequilibre des classes)")
    p.add_argument("--no-quali", action="store_true", help="ignorer les features de qualification")
    p.add_argument("--with-extra", action="store_true",
                   help="ajouter les features 7c au modele final (non retenues par defaut : pas de gain mesure)")
    p.add_argument("--no-register", action="store_true", help="ne pas enregistrer le modele dans le registry")
    return p.parse_args()


def make_params(args, y) -> dict:
    params = dict(
        n_estimators=args.n_estimators, max_depth=args.max_depth,
        learning_rate=args.learning_rate, subsample=args.subsample,
        colsample_bytree=args.colsample_bytree, min_child_weight=args.min_child_weight,
        objective="binary:logistic", eval_metric="logloss",
        tree_method="hist", random_state=42,
    )
    if args.balance:
        params["scale_pos_weight"] = scale_pos_weight(y)
    return params


def evaluate(df: pd.DataFrame, prefix: str = "") -> dict:
    m = {
        f"{prefix}winner_accuracy": winner_accuracy(df),
        f"{prefix}pole_baseline_accuracy": pole_baseline_accuracy(df),
        f"{prefix}top3_accuracy": topk_accuracy(df, k=3),
        f"{prefix}n_races": float(df["race_idx"].nunique()),
    }
    if PODIUM in df.columns:
        m[f"{prefix}podium_accuracy"] = podium_accuracy(df)
        m[f"{prefix}pole_podium_baseline"] = pole_podium_baseline(df)
    return m


def verdict(lo: float, hi: float) -> str:
    if lo > 0:
        return "BAT la baseline pole position (ecart significatif)"
    if hi < 0:
        return "est MOINS BON que la baseline pole position (ecart significatif)"
    return "ne se distingue pas de la baseline pole position (ecart non significatif)"


def main():
    args = parse_args()
    df = load_gold(args.gold)
    train, test = temporal_split(df, args.test_from)
    print(f"Gold : {len(df)} lignes | train {train.season.min()}-{train.season.max()} "
          f"({len(train)}) | test {test.season.min()}-{test.season.max()} ({len(test)})")
    if PODIUM not in df.columns:
        print("(colonne is_podium absente : relancer silver_to_gold pour avoir les metriques podium)")

    BASE = available_features(df, use_quali=False, use_extra=False)
    QUALI = available_features(df, use_quali=True, use_extra=False)
    ALL = available_features(df, use_quali=True, use_extra=True)
    FEATURES = BASE if args.no_quali else (ALL if args.with_extra else QUALI)
    print(f"Features ({len(FEATURES)}) : {', '.join(FEATURES)}")

    def make_fit_predict(cols, ranker=False):
        def fit_predict(tr, te):
            if not ranker:
                model = XGBClassifier(**make_params(args, tr[TARGET]))
                model.fit(tr[cols], tr[TARGET])
                return model.predict_proba(te[cols])[:, 1]
            # classement par course : le modele apprend a ordonner les pilotes d'une meme course
            # (tr est deja trie par course : load_gold trie par race_idx)
            model = XGBRanker(
                objective="rank:pairwise", n_estimators=args.n_estimators, max_depth=args.max_depth,
                learning_rate=args.learning_rate, subsample=args.subsample,
                colsample_bytree=args.colsample_bytree, min_child_weight=args.min_child_weight,
                tree_method="hist", random_state=42)
            model.fit(tr[cols], relevance(tr), group=tr.groupby("race_idx", sort=False).size().to_numpy())
            scored = te[["race_idx"]].copy()
            scored["score"] = model.predict(te[cols])
            return softmax_by_race(scored).to_numpy()
        return fit_predict

    fit_predict = make_fit_predict(FEATURES)

    params = make_params(args, train[TARGET])

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"xgb_d{args.max_depth}_n{args.n_estimators}_lr{args.learning_rate}") as run:
        # ---- 1. modele final : train < test_from, test >= test_from
        model = XGBClassifier(**params)
        model.fit(train[FEATURES], train[TARGET])
        test = test.copy()
        test["proba"] = model.predict_proba(test[FEATURES])[:, 1]

        metrics = evaluate(test)
        metrics["roc_auc"] = roc_auc_score(test[TARGET], test["proba"])
        metrics["log_loss"] = log_loss(test[TARGET], test["proba"])
        for season, part in test.groupby("season"):
            metrics.update(evaluate(part, prefix=f"s{season}_"))

        # ---- 2. walk-forward : un verdict sur ~170 courses
        wf = walk_forward(df, fit_predict, first_test=args.wf_from)
        diff, lo, hi = bootstrap_diff(winner_hits(wf), pole_hits(wf))
        metrics.update(evaluate(wf, prefix="wf_"))
        metrics.update({"wf_diff_vs_pole": diff, "wf_diff_ci_low": lo, "wf_diff_ci_high": hi,
                        "beats_baseline": float(lo > 0)})

        # ---- 3. comparaison des variantes (meme walk-forward, memes courses)
        variants = [("base (etape 7)", "base", BASE, False)]
        if QUALI != BASE:
            variants.append(("+ qualifications (7b)", "quali", QUALI, False))
        if ALL != QUALI:
            variants.append(("+ features 7c", "extra", ALL, False))
        variants.append(("classement + features 7c", "ranker", ALL, True))
        table = []
        for label, key, cols, rk in variants:
            w = walk_forward(df, make_fit_predict(cols, rk), first_test=args.wf_from)
            d, l, h = bootstrap_diff(winner_hits(w), pole_hits(w))
            row = {"label": label, "acc": winner_accuracy(w), "diff": d, "lo": l, "hi": h}
            metrics.update({f"wf_{key}_winner_accuracy": row["acc"], f"wf_{key}_diff_vs_pole": d,
                            f"wf_{key}_diff_ci_low": l, f"wf_{key}_diff_ci_high": h})
            if PODIUM in df.columns:
                row["podium"] = podium_accuracy(w)
                metrics[f"wf_{key}_podium_accuracy"] = row["podium"]
            table.append(row)

        mlflow.log_params({**params, "test_from": args.test_from, "wf_from": args.wf_from,
                           "balance": args.balance, "features": ",".join(FEATURES)})
        mlflow.log_metrics(metrics)

        importance = (pd.Series(model.feature_importances_, index=FEATURES, name="gain")
                      .sort_values(ascending=False))
        with tempfile.TemporaryDirectory() as tmp:
            imp_file = Path(tmp) / "feature_importance.csv"
            importance.to_csv(imp_file)
            mlflow.log_artifact(str(imp_file))

        mlflow.xgboost.log_model(
            model, "model",
            input_example=test[FEATURES].head(5),
            registered_model_name=None if args.no_register else MODEL_NAME,
        )

        # predictions du jeu de test -> utilisees par le dashboard (etape 9)
        extra = [c for c in (PODIUM, "finish_position") if c in test.columns]
        preds = add_race_rank(test)[ID_COLS + FEATURES + [TARGET] + extra + ["proba", "win_prob", "pred_rank"]]
        preds["run_id"] = run.info.run_id
        PRED_PATH.parent.mkdir(parents=True, exist_ok=True)
        preds.to_parquet(PRED_PATH, index=False)

    print("\nImportance des features :")
    print(importance.round(3).to_string())

    def line(label, key):
        return f"  {label:<24}: {metrics[key]:.1%}"

    print(f"\nTest {args.test_from}+ ({int(metrics['n_races'])} courses)")
    print(line("winner_accuracy", "winner_accuracy"))
    print(line("pole_baseline_accuracy", "pole_baseline_accuracy"))
    print(line("top3_accuracy", "top3_accuracy"))
    if PODIUM in df.columns:
        print(line("podium_accuracy", "podium_accuracy"))
        print(line("pole_podium_baseline", "pole_podium_baseline"))
    print(f"  roc_auc / log_loss      : {metrics['roc_auc']:.3f} / {metrics['log_loss']:.3f}")

    print(f"\nWalk-forward {args.wf_from}-{int(wf.season.max())} ({int(metrics['wf_n_races'])} courses)")
    print(line("winner_accuracy", "wf_winner_accuracy"))
    print(line("pole_baseline_accuracy", "wf_pole_baseline_accuracy"))
    print(f"  ecart modele - pole     : {diff:+.1%}  (IC95 % [{lo:+.1%}, {hi:+.1%}])")
    if PODIUM in df.columns:
        print(line("podium_accuracy", "wf_podium_accuracy"))
        print(line("pole_podium_baseline", "wf_pole_podium_baseline"))
    print(f"  -> le modele {verdict(lo, hi)}")
    print("\nComparaison des variantes (walk-forward, memes courses)")
    print(f"  {'variante':<28}{'vainqueur':>10}{'ecart vs pole':>15}   IC95 %          podium")
    for r in table:
        pod = f"{r['podium']:.1%}" if "podium" in r else "-"
        print(f"  {r['label']:<28}{r['acc']:>10.1%}{r['diff']:>+15.1%}   [{r['lo']:+.1%}, {r['hi']:+.1%}]  {pod:>7}")
    print("  (plusieurs variantes comparees sur les memes courses : un seul meilleur score ne prouve rien)")
    print(f"\nRun MLflow {run.info.run_id} | predictions -> {PRED_PATH.relative_to(ROOT)}")
    print("Train OK")


if __name__ == "__main__":
    main()
