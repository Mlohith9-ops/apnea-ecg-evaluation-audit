"""The official Apnea-ECG benchmark split, examined for subject overlap.

Train on all 35 learning records (a/b/c), exactly as in the PhysioNet/CinC 2000 challenge,
then test on the 35 x records, reported separately for:
  * seen   - x records whose suspected same-subject group also appears in training
  * unseen - x records whose group never appears in training
If 'seen' scores clearly higher, the official benchmark partly rewards recognising people.

Run this ONCE per model configuration, after all design choices are fixed. The x set is the
final test set: do not tune anything on it.

Night classes for x records (approximating PhysioNet's A/C definitions from the labels):
  apnea night   >= 100 apnea minutes
  control night  <   5 apnea minutes
  otherwise borderline (excluded from the binary night task)

Usage:
  python src/build_apnea_features.py --sets abcx --out data/processed/apnea_minutes_all.csv
  python src/official_split_apnea.py
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 0
N_BOOT = 2000


def make_model(name):
    if name == "logreg":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(max_iter=2000, class_weight="balanced"))
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                          class_weight="balanced", random_state=SEED)


def minute_metrics(y, prob):
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"minute_auc": roc_auc_score(y, prob) if len(set(y)) == 2 else np.nan,
            "minute_acc": (tp + tn) / len(y),
            "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
            "specificity": tn / (tn + fp) if tn + fp else np.nan}


def night_metrics(d):
    t = d.groupby("record").agg(mean_prob=("prob", "mean"), apnea_min=("label", "sum"),
                                frac=("label", "mean"))
    ac = t[(t.apnea_min >= 100) | (t.apnea_min < 5)]
    yy = (ac.apnea_min >= 100).to_numpy()
    return {"night_auc": roc_auc_score(yy, ac.mean_prob) if 0 < yy.sum() < len(yy) else np.nan,
            "night_spearman": spearmanr(t.mean_prob, t.frac).statistic if len(t) > 2 else np.nan}


def all_metrics(d):
    return {**minute_metrics(d.label.to_numpy(), d.prob.to_numpy()), **night_metrics(d)}


def boot(d, rng):
    groups = d.subject.unique()
    draws = []
    for _ in range(N_BOOT):
        pick = rng.choice(groups, len(groups), replace=True)
        b = pd.concat([d[d.subject == g].assign(record=lambda x, i=i: x.record + f"_{i}")
                       for i, g in enumerate(pick)])
        if b.label.nunique() == 2:
            draws.append(all_metrics(b))
    cols = ["minute_auc", "minute_acc", "sensitivity", "specificity", "night_auc", "night_spearman"]
    return pd.DataFrame(draws, columns=cols)   # empty (all NaN CIs) if a part has one class only


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="data/processed/apnea_minutes_all.csv")
    p.add_argument("--out", default="results/apnea_official_split")
    p.add_argument("--models", default="logreg,hist_gb")
    args = p.parse_args()

    df = pd.read_csv(args.features)
    feat = [c for c in df.columns if c.startswith(("m1_", "m5_"))]
    train = df[df.record.str[0].isin(list("abc"))]
    test = df[df.record.str[0] == "x"].copy()
    if test.empty:
        raise SystemExit("No x records in the feature file - rebuild it with --sets abcx.")
    seen_groups = set(train.inferred_subject)
    test["group_status"] = np.where(test.inferred_subject.isin(seen_groups), "seen", "unseen")
    test = test.rename(columns={"inferred_subject": "subject"})
    st = test.groupby("group_status").record.nunique()
    print(f"train: {train.record.nunique()} records | test: {test.record.nunique()} records "
          f"({st.get('seen', 0)} seen, {st.get('unseen', 0)} unseen suspected subjects)")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    rows = []
    for name in args.models.split(","):
        m = make_model(name).fit(train[feat].to_numpy(float), train.label.to_numpy())
        test["prob"] = m.predict_proba(test[feat].to_numpy(float))[:, 1]
        test[["record", "minute", "label", "subject", "group_status", "prob"]].to_csv(
            out / f"x_predictions_{name}.csv", index=False)

        parts = {"all x": test, "seen": test[test.group_status == "seen"],
                 "unseen": test[test.group_status == "unseen"]}
        boots = {}
        for part, d in parts.items():
            point, b = all_metrics(d), boot(d, rng)
            boots[part] = b
            lo, hi = b.quantile(0.025), b.quantile(0.975)
            rows.append({"model": name, "test_part": part, "records": d.record.nunique(),
                         **{k: f"{point[k]:.3f} [{lo[k]:.3f}, {hi[k]:.3f}]" for k in point}})
        # seen minus unseen, from independent bootstraps of the two disjoint parts
        n = min(len(boots["seen"]), len(boots["unseen"]))
        diff = boots["seen"].iloc[:n].reset_index(drop=True) - boots["unseen"].iloc[:n].reset_index(drop=True)
        pt = {k: all_metrics(parts["seen"])[k] - all_metrics(parts["unseen"])[k] for k in diff}
        rows.append({"model": name, "test_part": "seen - unseen", "records": "",
                     **{k: f"{pt[k]:+.3f} [{diff[k].quantile(.025):+.3f}, {diff[k].quantile(.975):+.3f}]"
                        for k in diff}})
        print(f"{name} done")

    s = pd.DataFrame(rows).set_index(["model", "test_part"])
    s.to_csv(out / "summary.csv")
    pd.set_option("display.width", 250)
    print("\nOfficial split (train a/b/c, test x), point [95% CI, bootstrap over subjects]:\n")
    print(s.to_string())


if __name__ == "__main__":
    main()
