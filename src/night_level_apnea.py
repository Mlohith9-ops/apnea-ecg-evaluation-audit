"""Whole-night evaluation with no test-set information in any threshold.

For each held-out group (leave-one-subject-out by default):
  1. Inner GroupKFold over the TRAINING subjects gives honest out-of-fold minute
     probabilities for the training nights.
  2. From those training nights only, fit
       - a decision threshold on mean night probability (A vs C nights, max balanced accuracy)
       - a linear map: mean night probability -> apnea minutes per hour
  3. Fit the minute model on all training minutes, predict the held-out nights,
     and apply the training-derived threshold and map.

Night-level targets (documented by PhysioNet, taken from the record name):
  a = apnea night, c = control night, b = borderline (excluded from the binary task,
  kept for the per-hour estimates).

Metrics:
  night_auc_A_vs_C   threshold-free ranking of A vs C nights by mean probability
  night_acc_A_vs_C   accuracy using the training-derived threshold
  spearman_all       rank correlation of mean probability with true apnea fraction (35 nights)
  mae_min_per_h      error of the training-calibrated apnea-minutes-per-hour estimate
95% CIs: bootstrap over subject groups.

Usage:
  python src/night_level_apnea.py
  python src/night_level_apnea.py --normalize record --out results/apnea_night_norm
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 0
N_BOOT = 2000


def make_model(name):
    if name == "logreg":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(max_iter=2000, class_weight="balanced"))
    if name == "hist_gb":
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                              class_weight="balanced", random_state=SEED)
    if name == "random_forest":
        return make_pipeline(SimpleImputer(strategy="median"),
                             RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                                    class_weight="balanced", n_jobs=-1,
                                                    random_state=SEED))
    raise ValueError(name)


def night_table(records, labels, prob):
    d = pd.DataFrame({"record": records, "label": labels, "prob": prob})
    t = d.groupby("record").agg(mean_prob=("prob", "mean"), true_frac=("label", "mean"))
    t["night_class"] = t.index.str[0]
    return t


def fit_threshold(t):
    """Threshold on mean_prob maximising balanced accuracy for A vs C nights."""
    ac = t[t.night_class.isin(["a", "c"])]
    y = (ac.night_class == "a").to_numpy()
    best, best_thr = -1, 0.5
    for thr in np.unique(ac.mean_prob):
        pred = ac.mean_prob.to_numpy() >= thr
        bal = np.mean([pred[y].mean() if y.any() else 0, (~pred[~y]).mean() if (~y).any() else 0])
        if bal > best:
            best, best_thr = bal, thr
    # place the cut halfway to the next lower value so it is not tied to one training night
    lower = ac.mean_prob[ac.mean_prob < best_thr]
    return (best_thr + lower.max()) / 2 if len(lower) else best_thr


def night_metrics(t):
    ac = t[t.night_class.isin(["a", "c"])]
    y = (ac.night_class == "a").to_numpy()
    return {
        "night_auc_A_vs_C": roc_auc_score(y, ac.mean_prob) if 0 < y.sum() < len(y) else np.nan,
        "night_acc_A_vs_C": (ac.pred_apnea_night == y).mean(),
        "spearman_all": spearmanr(t.mean_prob, t.true_frac).statistic,
        "mae_min_per_h": (t.est_min_per_h - t.true_frac * 60).abs().mean(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="data/processed/apnea_minutes.csv")
    p.add_argument("--out", default="results/apnea_night")
    p.add_argument("--normalize", choices=["none", "record"], default="none")
    p.add_argument("--grouping", choices=["subject", "record"], default="subject")
    p.add_argument("--models", default="logreg,hist_gb",
                   help="comma list; random_forest is supported but slow here")
    args = p.parse_args()

    df = pd.read_csv(args.features)
    df = df[df["record"].str[0].isin(list("abc"))].reset_index(drop=True)
    feat = [c for c in df.columns if c.startswith(("m1_", "m5_"))]
    if args.normalize == "record":
        grp = df.groupby("record")[feat]
        df[feat] = (df[feat] - grp.transform("median")) / (grp.transform("std") + 1e-9)

    X, y = df[feat].to_numpy(float), df["label"].to_numpy()
    rec = df["record"].to_numpy()
    subj = df["inferred_subject"].to_numpy()
    groups = subj if args.grouping == "subject" else df["record"].replace({"c06": "c05"}).to_numpy()
    print(f"{df.record.nunique()} nights, {len(set(groups))} held-out groups "
          f"({args.grouping}), normalize={args.normalize}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    summary = []
    for model_name in args.models.split(","):
        night_rows = []
        for tr, te in LeaveOneGroupOut().split(X, y, groups):
            assert not set(subj[tr]) & set(subj[te]) or args.grouping == "record"
            # 1. honest probabilities for the training nights (inner CV over training subjects)
            inner = np.zeros(len(tr))
            n_inner = min(5, len(set(groups[tr])))
            for itr, ite in GroupKFold(n_inner).split(X[tr], y[tr], groups[tr]):
                m = make_model(model_name).fit(X[tr][itr], y[tr][itr])
                inner[ite] = m.predict_proba(X[tr][ite])[:, 1]
            t_train = night_table(rec[tr], y[tr], inner)
            # 2. training-only threshold and calibration map
            thr = fit_threshold(t_train)
            cal = LinearRegression().fit(t_train[["mean_prob"]], t_train.true_frac * 60)
            # 3. outer model on all training minutes, applied to held-out nights
            m = make_model(model_name).fit(X[tr], y[tr])
            t_test = night_table(rec[te], y[te], m.predict_proba(X[te])[:, 1])
            t_test["pred_apnea_night"] = t_test.mean_prob >= thr
            t_test["est_min_per_h"] = np.clip(cal.predict(t_test[["mean_prob"]]), 0, 60)
            t_test["threshold_used"] = thr
            t_test["held_out_group"] = groups[te][0]
            night_rows.append(t_test)
        nights = pd.concat(night_rows)
        nights.to_csv(out / f"nights_{model_name}.csv")

        point = night_metrics(nights)
        g_of_night = nights["held_out_group"].to_numpy()
        uniq = np.unique(g_of_night)
        draws = []
        for _ in range(N_BOOT):
            pick = rng.choice(uniq, len(uniq), replace=True)
            boot = pd.concat([nights[g_of_night == u] for u in pick])
            if boot.night_class.isin(["a"]).any() and boot.night_class.isin(["c"]).any():
                draws.append(night_metrics(boot))
        d = pd.DataFrame(draws)
        lo, hi = d.quantile(0.025), d.quantile(0.975)
        summary.append({"model": model_name,
                        **{k: f"{point[k]:.3f} [{lo[k]:.3f}, {hi[k]:.3f}]" for k in point}})
        print(f"{model_name} done")

    s = pd.DataFrame(summary).set_index("model")
    s.to_csv(out / "night_summary.csv")
    pd.set_option("display.width", 250)
    print(f"\nWhole-night results ({args.grouping} held out), point [95% CI over subjects]:\n")
    print(s.to_string())
    print("\nPer-night predictions saved to", out)


if __name__ == "__main__":
    main()
