"""Apnea baseline v2: the same models under a leaky and two grouped evaluations.

Splits (learning records a/b/c only; the x records stay untouched):
  * minute_wise (leaky):          5-fold StratifiedKFold over minutes. Diagnostic only.
  * leave-one-record-out:         each record is held out once; c05/c06 held out together
                                  (PhysioNet documents them as one original recording).
  * leave-one-subject-out:        each suspected-subject group (identical age/sex/height/
                                  weight) is held out once.

Confidence intervals: 95% bootstrap over suspected-subject groups (resampling people,
not minutes), applied to the pooled out-of-fold predictions.

--normalize record  z-scores every feature within its own record before modelling.
  This uses only that record's unlabelled features (no labels), so it is not label
  leakage, but it is transductive: it needs the whole night's recording. Say so in the paper.

Usage:
  python src/baseline_apnea.py
  python src/baseline_apnea.py --normalize record --out results/apnea_baseline_norm
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 0
N_BOOT = 1000


def make_models():
    return {
        "logreg": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                LogisticRegression(max_iter=2000, class_weight="balanced")),
        "random_forest": make_pipeline(SimpleImputer(strategy="median"),
                                       RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                                              class_weight="balanced",
                                                              n_jobs=-1, random_state=SEED)),
        "hist_gb": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                  class_weight="balanced", random_state=SEED),
    }


def metrics(y, prob):
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    return {"accuracy": (tp + tn) / len(y), "sensitivity": sens, "specificity": spec,
            "balanced_acc": np.nanmean([sens, spec]),
            "auc": roc_auc_score(y, prob) if len(set(y)) == 2 else np.nan}


def bootstrap_ci(y, prob, groups, rng):
    """95% CI by resampling whole subject groups with replacement."""
    uniq = np.unique(groups)
    idx_by_g = {u: np.flatnonzero(groups == u) for u in uniq}
    draws = []
    for _ in range(N_BOOT):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_g[u] for u in pick])
        if len(set(y[idx])) < 2:
            continue
        draws.append(metrics(y[idx], prob[idx]))
    d = pd.DataFrame(draws)
    return d.quantile(0.025), d.quantile(0.975)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="data/processed/apnea_minutes.csv")
    p.add_argument("--out", default="results/apnea_baseline")
    p.add_argument("--normalize", choices=["none", "record"], default="none")
    args = p.parse_args()

    df = pd.read_csv(args.features)
    df = df[df["record"].str[0].isin(list("abc"))].reset_index(drop=True)   # learning set only
    feat = [c for c in df.columns if c.startswith(("m1_", "m5_"))]
    if args.normalize == "record":
        grp = df.groupby("record")[feat]
        df[feat] = (df[feat] - grp.transform("median")) / (grp.transform("std") + 1e-9)

    X, y = df[feat].to_numpy(float), df["label"].to_numpy()
    subj = df["inferred_subject"].to_numpy()
    rec_groups = df["record"].replace({"c06": "c05"}).to_numpy()
    print(f"{len(df)} minutes, {df.record.nunique()} records, {len(set(subj))} subject groups, "
          f"{len(feat)} features, normalize={args.normalize}")

    splits = {
        "minute_wise (leaky)": list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y)),
        "leave-one-record-out": list(LeaveOneGroupOut().split(X, y, rec_groups)),
        "leave-one-subject-out": list(LeaveOneGroupOut().split(X, y, subj)),
    }
    for tr, te in splits["leave-one-record-out"]:
        assert not set(rec_groups[tr]) & set(rec_groups[te]), "record leaked across folds"
    for tr, te in splits["leave-one-subject-out"]:
        assert not set(subj[tr]) & set(subj[te]), "subject leaked across folds"

    rng = np.random.default_rng(SEED)
    rows, oof = [], pd.DataFrame({"record": df["record"], "subject": subj, "label": y})
    for split_name, folds in splits.items():
        for model_name in make_models():
            prob = np.zeros(len(y))
            for tr, te in folds:
                m = make_models()[model_name]
                m.fit(X[tr], y[tr])
                prob[te] = m.predict_proba(X[te])[:, 1]
            oof[f"{model_name}|{split_name}"] = prob
            point = metrics(y, prob)
            lo, hi = bootstrap_ci(y, prob, subj, rng)
            rows.append({"model": model_name, "split": split_name,
                         **{k: f"{point[k]:.3f} [{lo[k]:.3f}, {hi[k]:.3f}]" for k in point}})
            print(f"{split_name:24s} {model_name:14s} AUC {point['auc']:.3f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    oof.to_csv(out / "oof_predictions.csv", index=False)
    summary = pd.DataFrame(rows).set_index(["model", "split"])
    summary.to_csv(out / "summary.csv")
    pd.set_option("display.width", 250)
    print("\nPoint estimate [95% CI, bootstrap over subject groups]:\n")
    print(summary.to_string())


if __name__ == "__main__":
    main()
