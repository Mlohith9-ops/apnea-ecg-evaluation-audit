"""Two diagnostics for the conflict between leave-one-subject-out and the official split.

1. Duplicate-signal scan: for every pair of records, slide one record's minute-by-minute
   mean RR series against the other's (lags up to +-300 min) and report the best
   correlation over at least 60 overlapping minutes. Separate nights of one person
   should correlate weakly; copies of the same recording (like the documented c05/c06)
   should correlate close to 1.

2. Per-record results on the official x test set, from official_split_apnea.py output.

Usage:
  python src/diagnose_overlap.py
"""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

MIN_OVERLAP = 60
MAX_LAG = 300


def best_lag_corr(a, b):
    best, best_lag = -1.0, 0
    for lag in range(-MAX_LAG, MAX_LAG + 1):
        if lag >= 0:
            x, y = a[lag:], b[:len(a) - lag]
        else:
            x, y = a[:len(a) + lag], b[-lag:]
        n = min(len(x), len(y))
        if n < MIN_OVERLAP:
            continue
        x, y = x[:n], y[:n]
        ok = ~(np.isnan(x) | np.isnan(y))
        if ok.sum() < MIN_OVERLAP or x[ok].std() == 0 or y[ok].std() == 0:
            continue
        c = np.corrcoef(x[ok], y[ok])[0, 1]
        if c > best:
            best, best_lag = c, lag
    return best, best_lag


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="data/processed/apnea_minutes_all.csv")
    p.add_argument("--preds", default="results/apnea_official_split/x_predictions_logreg.csv")
    p.add_argument("--out", default="results/apnea_diagnostics")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)

    df = pd.read_csv(args.features).sort_values(["record", "minute"])
    series = {r: d["m1_rr_mean"].to_numpy(float) for r, d in df.groupby("record")}
    subj = df.groupby("record")["inferred_subject"].first()

    rows = []
    for r1, r2 in combinations(sorted(series), 2):
        c, lag = best_lag_corr(series[r1], series[r2])
        rows.append({"rec_1": r1, "rec_2": r2, "best_corr": round(c, 3), "lag_min": lag,
                     "same_suspected_subject": subj[r1] == subj[r2]})
    pairs = pd.DataFrame(rows).sort_values("best_corr", ascending=False)
    pairs.to_csv(out / "pair_correlations.csv", index=False)

    print("1. Most similar record pairs (best lagged correlation of minute mean RR):\n")
    print(pairs.head(20).to_string(index=False))
    same, diff = pairs[pairs.same_suspected_subject], pairs[~pairs.same_suspected_subject]
    print(f"\nMedian best correlation: same suspected subject {same.best_corr.median():.3f} "
          f"(n={len(same)}), different {diff.best_corr.median():.3f} (n={len(diff)})")
    print(f"Pairs with correlation > 0.9: {(pairs.best_corr > 0.9).sum()}")

    if Path(args.preds).exists():
        x = pd.read_csv(args.preds)
        per = []
        for r, d in x.groupby("record"):
            pred = d.prob >= 0.5
            per.append({"record": r, "subject": d.subject.iloc[0], "status": d.group_status.iloc[0],
                        "apnea_min": int(d.label.sum()), "acc": round((pred == d.label).mean(), 3),
                        "mean_prob": round(d.prob.mean(), 3),
                        "auc": round(roc_auc_score(d.label, d.prob), 3) if d.label.nunique() == 2 else np.nan})
        per = pd.DataFrame(per).sort_values(["status", "acc"])
        per.to_csv(out / "x_per_record.csv", index=False)
        print("\n2. Official split, per x record (worst first within each group):\n")
        print(per.to_string(index=False))


if __name__ == "__main__":
    main()
