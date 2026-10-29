"""Analysis only: why is learning-set LOSO hist_gb AUC (~0.689) so far below official x AUC (~0.913)?

Uses existing predictions only. Nothing is retrained and no new predictions are made on x.

  1. within-record AUC   per-record ROC AUC (records with both classes), median and IQR,
                         learning LOSO vs official x, never pooled across records
  2. hard subjects       pooled LOSO AUC with suspected subject groups P01, P09, P02 removed
  3. within-record rank  pooled LOSO AUC after replacing each record's probabilities with
                         within-record average ranks scaled to [0, 1]

Pooled AUCs get a 95% CI from the same subject-group bootstrap as baseline_apnea.py
(resample whole suspected-subject groups with replacement, 1000 draws, seed 0).

Usage:
  python src/explain_gap.py
  python src/explain_gap.py --out results/apnea_gap_analysis_v2
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

SEED = 0
N_BOOT = 1000
LOSO_COL = "hist_gb|leave-one-subject-out"
HARD_GROUPS = ["P01", "P09", "P02"]


def auc_ci(y, prob, groups, rng):
    """95% CI for pooled AUC by resampling whole subject groups with replacement."""
    uniq = np.unique(groups)
    idx_by_g = {u: np.flatnonzero(groups == u) for u in uniq}
    draws = []
    for _ in range(N_BOOT):
        idx = np.concatenate([idx_by_g[u] for u in rng.choice(uniq, size=len(uniq), replace=True)])
        if len(set(y[idx])) == 2:
            draws.append(roc_auc_score(y[idx], prob[idx]))
    return np.quantile(draws, 0.025), np.quantile(draws, 0.975)


def validate(df, name, prob_col):
    """Count invalid rows; return (valid mask, one-row summary). Invalid rows are reported, not hidden."""
    bad_label = ~df.label.isin([0, 1])
    bad_prob = df[prob_col].isna() | ~np.isfinite(df[prob_col]) | ~df[prob_col].between(0, 1)
    bad_id = df.record.isna() | df.subject.isna()
    valid = ~(bad_label | bad_prob | bad_id)
    return valid, {
        "set": name, "prob_column": prob_col, "rows": len(df), "records": df.record.nunique(),
        "subject_groups": df.subject.nunique(), "label_values": ",".join(map(str, sorted(df.label.unique()))),
        "apnea_minutes": int((df.label == 1).sum()), "invalid_label": int(bad_label.sum()),
        "invalid_prob": int(bad_prob.sum()), "missing_id": int(bad_id.sum()),
        "full_row_duplicates": int(df.duplicated().sum()), "rows_used": int(valid.sum()),
    }


def per_record_auc(df, prob_col, name):
    rows = []
    for rec, g in df.groupby("record", sort=True):
        both = g.label.nunique() == 2
        rows.append({"set": name, "record": rec, "subject": g.subject.iloc[0], "minutes": len(g),
                     "apnea_minutes": int(g.label.sum()), "eligible": both,
                     "auc": roc_auc_score(g.label, g[prob_col]) if both else np.nan})
    return pd.DataFrame(rows)


def rank_within_record(df, prob_col):
    """Average ranks within each record, scaled to [0, 1]; a 1-minute record gets 0.5."""
    r = df.groupby("record")[prob_col].rank(method="average")
    n = df.groupby("record")[prob_col].transform("size")
    return ((r - 1) / (n - 1)).where(n > 1, 0.5)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--learning", default="results/apnea_baseline_final/oof_predictions.csv")
    p.add_argument("--test", default="results/apnea_official_final_v2/x_predictions_hist_gb.csv")
    p.add_argument("--out", default="results/apnea_gap_analysis")
    args = p.parse_args()

    out = Path(args.out)
    if out.exists():
        sys.exit(f"{out} already exists; choose a new --out (earlier runs are never overwritten)")

    learn = pd.read_csv(args.learning)
    test = pd.read_csv(args.test)
    for name, df, cols in [("learning", learn, ["record", "subject", "label", LOSO_COL]),
                           ("x test", test, ["record", "subject", "label", "prob"])]:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            sys.exit(f"{name} predictions lack columns {missing}; analysis not supported")

    # ---- validation
    lv, lsum = validate(learn, "learning LOSO", LOSO_COL)
    tv, tsum = validate(test, "official x", "prob")
    validation = pd.DataFrame([lsum, tsum])
    if (validation.rows != validation.rows_used).any():
        print("WARNING: invalid rows found and excluded; see validation.csv")
    learn, test = learn[lv].copy(), test[tv].copy()

    out.mkdir(parents=True)
    rng = np.random.default_rng(SEED)
    y, prob, groups = learn.label.values, learn[LOSO_COL].values, learn.subject.values
    orig_auc = roc_auc_score(y, prob)
    orig_lo, orig_hi = auc_ci(y, prob, groups, rng)

    # ---- analysis 1: within-record AUC
    per_rec = pd.concat([per_record_auc(learn, LOSO_COL, "learning LOSO"),
                         per_record_auc(test, "prob", "official x")])
    a1 = []
    for name, g in per_rec.groupby("set", sort=False):
        e = g[g.eligible]
        a1.append({"set": name, "records": len(g), "eligible_records": len(e),
                   "excluded_one_class": int((~g.eligible).sum()),
                   "excluded_records": ",".join(g.loc[~g.eligible, "record"]),
                   "median_auc": e.auc.median(), "q25_auc": e.auc.quantile(0.25),
                   "q75_auc": e.auc.quantile(0.75), "min_auc": e.auc.min(), "max_auc": e.auc.max()})
    a1 = pd.DataFrame(a1)
    per_rec.to_csv(out / "analysis1_within_record_auc_per_record.csv", index=False)
    a1.to_csv(out / "analysis1_within_record_auc_summary.csv", index=False)

    # ---- analysis 2: hard-subject sensitivity
    present = {g: sorted(learn.loc[learn.subject == g, "record"].unique()) for g in HARD_GROUPS}
    if not all(present.values()):
        sys.exit(f"hard groups missing from learning predictions: {present}; not guessing")
    keep = ~learn.subject.isin(HARD_GROUPS)
    k = learn[keep]
    a2_auc = roc_auc_score(k.label, k[LOSO_COL])
    a2_lo, a2_hi = auc_ci(k.label.values, k[LOSO_COL].values, k.subject.values, rng)
    a2 = pd.DataFrame([
        {"variant": "original pooled LOSO", "excluded_groups": "", "excluded_records": "",
         "rows": len(learn), "records": learn.record.nunique(), "subject_groups": learn.subject.nunique(),
         "apnea_minutes": int(learn.label.sum()), "auc": orig_auc, "ci_low": orig_lo, "ci_high": orig_hi},
        {"variant": "without P01, P09, P02", "excluded_groups": ",".join(HARD_GROUPS),
         "excluded_records": ",".join(r for g in HARD_GROUPS for r in present[g]),
         "rows": len(k), "records": k.record.nunique(), "subject_groups": k.subject.nunique(),
         "apnea_minutes": int(k.label.sum()), "auc": a2_auc, "ci_low": a2_lo, "ci_high": a2_hi}])
    a2.to_csv(out / "analysis2_hard_subject_sensitivity.csv", index=False)

    # ---- analysis 3: within-record ranks
    ranked = rank_within_record(learn, LOSO_COL).values
    a3_auc = roc_auc_score(y, ranked)
    a3_lo, a3_hi = auc_ci(y, ranked, groups, rng)
    a3 = pd.DataFrame([
        {"variant": "original pooled LOSO probabilities", "rows": len(learn), "records": learn.record.nunique(),
         "auc": orig_auc, "ci_low": orig_lo, "ci_high": orig_hi},
        {"variant": "within-record average ranks scaled to [0,1]", "rows": len(learn),
         "records": learn.record.nunique(), "auc": a3_auc, "ci_low": a3_lo, "ci_high": a3_hi}])
    a3.to_csv(out / "analysis3_within_record_rank_auc.csv", index=False)
    validation.to_csv(out / "validation.csv", index=False)

    # ---- terminal
    pd.set_option("display.width", 200)
    fmt = lambda d: d.to_string(index=False, float_format=lambda v: f"{v:.3f}")
    print("Validation:\n" + fmt(validation.T.reset_index().rename(columns={"index": "check"})))
    print("\nAnalysis 1 - within-record AUC (records with both classes):\n" + fmt(a1))
    print("\nAnalysis 2 - hard-subject sensitivity (pooled LOSO hist_gb):\n" + fmt(a2))
    print("\nAnalysis 3 - within-record ranking (pooled LOSO hist_gb):\n" + fmt(a3))

    # ---- report
    def ci(a, lo, hi):
        return f"{a:.3f} [{lo:.3f}, {hi:.3f}]"
    s = a1.set_index("set")
    L, X = s.loc["learning LOSO"], s.loc["official x"]
    report = f"""# Learning LOSO vs official x AUC gap: analysis of existing predictions

Analysis only: no model was retrained, the official split was not rerun, and no new predictions
were made on x. Script: `src/explain_gap.py`.

## Inputs
- Learning: `{args.learning}`, column `{LOSO_COL}` (out-of-fold, leave-one-suspected-subject-out)
- Official x: `{args.test}`, column `prob` (hist_gb trained on 34 learning records)

## Validation
| set | rows | records | subject groups | labels | apnea min | invalid label | invalid prob | missing id | full-row dups | rows used |
|---|---|---|---|---|---|---|---|---|---|---|
""" + "\n".join(
        f"| {r.set} | {r.rows} | {r.records} | {r.subject_groups} | {r.label_values} | {r.apnea_minutes} | "
        f"{r.invalid_label} | {r.invalid_prob} | {r.missing_id} | {r.full_row_duplicates} | {r.rows_used} |"
        for r in validation.itertuples()) + f"""

The learning file has no minute column. Its rows line up one-to-one, in order, with
`data/processed/apnea_minutes.csv` (record, label and subject all match), so the full-row duplicates
are distinct minutes: early minutes of c02 and c09 with zero detected beats, whose imputed features
and predictions are therefore identical. They are kept. Subject IDs match
`splits/apnea_ecg_subject_map.csv` in both files.

## Metric definitions
- **Within-record AUC**: ROC AUC from one record's minute labels and probabilities. Defined only when
  the record has both apnea and normal minutes. Summarised as median and IQR (25th-75th percentile)
  over records, never pooled across records.
- **Pooled AUC**: one ROC AUC over all minutes of all records together. It mixes within-record
  ranking with between-record differences in probability level.
- **95% CI**: percentile bootstrap resampling whole suspected-subject groups with replacement
  ({N_BOOT} draws, seed {SEED}), as in `baseline_apnea.py`. The draws differ from that script's,
  so CI ends can differ slightly from the baseline table.

## 1. Within-record AUC
| | learning LOSO | official x |
|---|---|---|
| records | {L.records} | {X.records} |
| eligible (both classes) | {L.eligible_records} | {X.eligible_records} |
| excluded (one class only) | {L.excluded_one_class} ({L.excluded_records or "none"}) | {X.excluded_one_class} ({X.excluded_records or "none"}) |
| median AUC | {L.median_auc:.3f} | {X.median_auc:.3f} |
| IQR | {L.q25_auc:.3f} - {L.q75_auc:.3f} | {X.q25_auc:.3f} - {X.q75_auc:.3f} |
| range | {L.min_auc:.3f} - {L.max_auc:.3f} | {X.min_auc:.3f} - {X.max_auc:.3f} |

## 2. Hard-subject sensitivity (pooled learning LOSO)
Excluded groups P01 ({", ".join(present["P01"])}), P09 ({", ".join(present["P09"])}),
P02 ({", ".join(present["P02"])}; its other record x14 is in the test set and not in this file).

| variant | rows | records | subject groups | AUC [95% CI] |
|---|---|---|---|---|
| original | {len(learn)} | {learn.record.nunique()} | {learn.subject.nunique()} | {ci(orig_auc, orig_lo, orig_hi)} |
| without P01, P09, P02 | {len(k)} | {k.record.nunique()} | {k.subject.nunique()} | {ci(a2_auc, a2_lo, a2_hi)} |

## 3. Within-record ranking (pooled learning LOSO)
| predictions | AUC [95% CI] |
|---|---|
| original probabilities | {ci(orig_auc, orig_lo, orig_hi)} |
| within-record ranks scaled to [0, 1] | {ci(a3_auc, a3_lo, a3_hi)} |

Replacing probabilities with within-record ranks removes every difference in probability level and
calibration between records, keeping only how each record's minutes are ordered. A change in pooled
AUC shows how much the pooled number depends on between-record differences. It does not show that
calibration caused the learning-vs-x gap: the rank transform also discards real between-record
information (a mostly-apnea night ranked against a mostly-normal night), and it changes nothing
about the x predictions.

## Caveats
- The two sets differ in more than evaluation scheme: LOSO models were trained on 33 or fewer
  records while the x model used all 34, and the records, subjects and apnea prevalence differ.
- Subject groups other than the documented copies are inferred from identical demographics
  (suspected same-subject groups, not identities).
- P01, P09 and P02 were named for exclusion before this analysis; analysis 2 is a sensitivity check,
  not a claim that these subjects should be dropped.
- Within-record AUCs rest on one record each and are unstable for records with few apnea or
  normal minutes; see `analysis1_within_record_auc_per_record.csv` for per-record counts.
- Minute-wise (leaky) predictions in the learning file were not used.
"""
    (out / "report.md").write_text(report, encoding="utf-8")
    print("\nSaved:")
    for f in sorted(out.iterdir()):
        print(f"  {f.resolve()}")


if __name__ == "__main__":
    main()
