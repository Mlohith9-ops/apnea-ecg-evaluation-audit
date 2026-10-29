"""Analyse saved out-of-fold predictions from baseline_apnea.py.

1. Paired subject-level bootstrap for AUC differences (same minutes, same resampled people),
   e.g. raw vs record-normalized, or leave-one-record-out vs leave-one-subject-out.
2. Per-subject breakdown: which held-out people the models fail on.
3. Per-record (whole-night) estimates: predicted vs true apnea minutes per hour.

Usage:
  python src/analyze_apnea_results.py --raw results/apnea_baseline/oof_predictions.csv ^
      --norm results/apnea_baseline_norm/oof_predictions.csv --out results/apnea_analysis
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

N_BOOT = 2000
SPLIT = "leave-one-subject-out"


def paired_auc_diff(y, pa, pb, groups, rng):
    """AUC(b) - AUC(a) with a 95% CI from resampling whole subjects."""
    uniq = np.unique(groups)
    idx_by_g = {u: np.flatnonzero(groups == u) for u in uniq}
    diffs = []
    for _ in range(N_BOOT):
        idx = np.concatenate([idx_by_g[u] for u in rng.choice(uniq, len(uniq), replace=True)])
        if len(set(y[idx])) < 2:
            continue
        diffs.append(roc_auc_score(y[idx], pb[idx]) - roc_auc_score(y[idx], pa[idx]))
    diffs = np.array(diffs)
    point = roc_auc_score(y, pb) - roc_auc_score(y, pa)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p_two = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return point, lo, hi, min(p_two, 1.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default="results/apnea_baseline/oof_predictions.csv")
    p.add_argument("--norm", default="results/apnea_baseline_norm/oof_predictions.csv")
    p.add_argument("--out", default="results/apnea_analysis")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    rng = np.random.default_rng(0)

    raw, norm = pd.read_csv(args.raw), pd.read_csv(args.norm)
    assert (raw[["record", "label"]].values == norm[["record", "label"]].values).all(), "row order differs"
    y, g = raw["label"].to_numpy(), raw["subject"].to_numpy()
    models = sorted({c.split("|")[0] for c in raw.columns if "|" in c})

    # 1. paired comparisons
    rows = []
    for m in models:
        comps = {
            "record-out -> subject-out (raw)": (raw[f"{m}|leave-one-record-out"], raw[f"{m}|{SPLIT}"]),
            "leaky -> subject-out (raw)": (raw[f"{m}|minute_wise (leaky)"], raw[f"{m}|{SPLIT}"]),
            "raw -> normalized (subject-out)": (raw[f"{m}|{SPLIT}"], norm[f"{m}|{SPLIT}"]),
        }
        for name, (a, b) in comps.items():
            d, lo, hi, pv = paired_auc_diff(y, a.to_numpy(), b.to_numpy(), g, rng)
            rows.append({"model": m, "comparison": name, "auc_change": round(d, 3),
                         "ci_low": round(lo, 3), "ci_high": round(hi, 3), "bootstrap_p": round(pv, 3)})
    comp = pd.DataFrame(rows)
    comp.to_csv(out / "paired_auc_differences.csv", index=False)
    print("1. Paired AUC changes (negative = worse), 95% CI from subject bootstrap:\n")
    print(comp.to_string(index=False))

    # 2. per-subject breakdown under leave-one-subject-out
    per = []
    for s, d in raw.groupby("subject"):
        row = {"subject": s, "records": ",".join(sorted(d.record.unique())), "minutes": len(d),
               "apnea_frac": round(d.label.mean(), 3)}
        for src, tbl in (("raw", raw), ("norm", norm)):
            for m in models:
                pred = (tbl.loc[d.index, f"{m}|{SPLIT}"] >= 0.5).astype(int)
                row[f"{src}_{m}_acc"] = round((pred == d.label).mean(), 3)
        per.append(row)
    per = pd.DataFrame(per).sort_values("apnea_frac", ascending=False)
    per.to_csv(out / "per_subject_accuracy.csv", index=False)
    print("\n2. Per-subject accuracy under leave-one-subject-out:\n")
    print(per.to_string(index=False))

    # 3. per-record (whole-night) apnea minutes per hour
    rec_rows = []
    for src, tbl in (("raw", raw), ("norm", norm)):
        for m in models:
            col = f"{m}|{SPLIT}"
            agg = tbl.groupby("record").apply(
                lambda d: pd.Series({"true_per_h": d.label.mean() * 60,
                                     "pred_per_h": (d[col] >= 0.5).mean() * 60}),
                include_groups=False)
            rho = spearmanr(agg.true_per_h, agg.pred_per_h).statistic
            mae = (agg.true_per_h - agg.pred_per_h).abs().mean()
            # Apnea-ECG convention: A records >= 100 apnea minutes, C records < 5;
            # here a simple per-hour cut at 5 apnea minutes/hour separates apnea vs normal nights
            true_pos, pred_pos = agg.true_per_h >= 5, agg.pred_per_h >= 5
            rec_rows.append({"features": src, "model": m, "spearman_rho": round(rho, 3),
                             "mae_apnea_min_per_h": round(mae, 1),
                             "night_level_acc_at_5_per_h": round((true_pos == pred_pos).mean(), 3)})
    rec = pd.DataFrame(rec_rows)
    rec.to_csv(out / "per_record_summary.csv", index=False)
    print("\n3. Whole-night estimates (leave-one-subject-out):\n")
    print(rec.to_string(index=False))


if __name__ == "__main__":
    main()
