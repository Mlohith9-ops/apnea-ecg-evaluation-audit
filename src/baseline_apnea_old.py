"""First apnea baseline: the same models under a leaky and a grouped evaluation.

  * minute-wise: StratifiedKFold over minutes (minutes of one person land in train AND test)
  * grouped:     StratifiedGroupKFold over suspected-subject groups (no person in both)

Uses only the learning records (a/b/c). The x records stay untouched.
All preprocessing lives inside the sklearn Pipeline, so it is fitted on training folds only.

Usage:
  python src/baseline_apnea.py --features data/processed/apnea_minutes.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEEDS = [0, 1, 2, 3, 4]
N_FOLDS = 5


def models(seed):
    return {
        "logreg": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                LogisticRegression(max_iter=2000, class_weight="balanced")),
        "random_forest": make_pipeline(SimpleImputer(strategy="median"),
                                       RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                                              class_weight="balanced",
                                                              n_jobs=-1, random_state=seed)),
        "hist_gb": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                  class_weight="balanced", random_state=seed),
    }


def metrics(y, prob):
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    return {"accuracy": (tp + tn) / len(y), "sensitivity": sens, "specificity": spec,
            "balanced_acc": np.nanmean([sens, spec]),
            "auc": roc_auc_score(y, prob) if len(set(y)) == 2 else np.nan}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="data/processed/apnea_minutes.csv")
    p.add_argument("--out", default="results/apnea_baseline")
    args = p.parse_args()

    df = pd.read_csv(args.features)
    df = df[df["record"].str[0].isin(list("abc"))].reset_index(drop=True)   # learning set only
    feat = [c for c in df.columns if c.startswith(("m1_", "m5_"))]
    X, y, g = df[feat].to_numpy(float), df["label"].to_numpy(), df["inferred_subject"].to_numpy()
    print(f"{len(df)} minutes, {df.record.nunique()} records, {len(set(g))} subject groups, "
          f"{len(feat)} features")

    rows = []
    for seed in SEEDS:
        splitters = {
            "minute_wise (leaky)": StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed).split(X, y),
            "grouped (subject)": StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=seed).split(X, y, g),
        }
        for split_name, folds in splitters.items():
            folds = list(folds)
            for tr, te in folds:   # leakage guard: grouped folds must never share a subject
                if split_name.startswith("grouped"):
                    assert not set(g[tr]) & set(g[te]), "subject leaked across folds"
            for model_name, _ in models(seed).items():
                prob = np.zeros(len(y))
                for tr, te in folds:
                    m = models(seed)[model_name]
                    m.fit(X[tr], y[tr])
                    prob[te] = m.predict_proba(X[te])[:, 1]
                rows.append({"seed": seed, "split": split_name, "model": model_name, **metrics(y, prob)})
            print(f"seed {seed}: {split_name} done")

    res = pd.DataFrame(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "per_seed.csv", index=False)

    summary = (res.groupby(["model", "split"])
                  [["accuracy", "sensitivity", "specificity", "balanced_acc", "auc"]]
                  .agg(lambda s: f"{s.mean():.3f} ± {1.96 * s.std(ddof=1) / np.sqrt(len(s)):.3f}"))
    summary.to_csv(out / "summary.csv")
    pd.set_option("display.width", 200)
    print("\nMean ± 95% CI over seeds (pooled out-of-fold predictions):\n")
    print(summary.to_string())


if __name__ == "__main__":
    main()
