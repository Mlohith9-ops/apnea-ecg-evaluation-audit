"""Where do the apnea labels of duplicate recordings disagree, and is it only boundary shift?

For each pair rec1:rec2:offset_s (rec2 time = rec1 time + offset), every apnea minute in
either copy is placed on rec1's timeline and compared with the other copy's labels:

  consistent   the other copy also has apnea covering >= 30 s of this minute
  shift_only   the other copy has apnea covering 1-29 s of it: explainable by the shifted
               minute grid (one event split differently between minutes)
  unmatched    the other copy has NO apnea anywhere in this minute: a genuine annotation
               difference that boundary shift cannot explain

Times are printed as hh:mm:ss from the start of each record.

Usage:
  python src/label_disagreement_report.py
"""
import argparse
from pathlib import Path

import pandas as pd
import wfdb


def hms(sec):
    sec = int(round(sec))
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def apnea_minutes(root, rec):
    ann = wfdb.rdann(str(root / rec), "apn")
    fs = int(wfdb.rdheader(str(root / rec)).fs)
    starts = [s / fs for s, sym in zip(ann.sample, ann.symbol)]
    labels = [sym == "A" for sym in ann.symbol]
    return starts, labels


def overlap(a0, a1, intervals):
    return sum(max(0.0, min(a1, b1) - max(a0, b0)) for b0, b1 in intervals)


def report(root, r1, r2, off):
    s1, l1 = apnea_minutes(root, r1)
    s2, l2 = apnea_minutes(root, r2)
    # everything on rec1's timeline
    apnea1 = [(t, t + 60) for t, a in zip(s1, l1) if a]
    apnea2 = [(t - off, t - off + 60) for t, a in zip(s2, l2) if a]
    cover1 = (s1[0], s1[-1] + 60)
    cover2 = (s2[0] - off, s2[-1] + 60 - off)
    rows = []
    for src, mine, other, other_cover in ((r1, apnea1, apnea2, cover2), (r2, apnea2, apnea1, cover1)):
        for a0, a1 in mine:
            if a1 <= other_cover[0] or a0 >= other_cover[1]:
                status, ov = "outside_other_copy", 0.0
            else:
                ov = overlap(a0, a1, other)
                status = "consistent" if ov >= 30 else "shift_only" if ov > 0 else "unmatched"
            rows.append({"pair": f"{r1}/{r2}", "apnea_minute_in": src,
                         f"{r1}_time": f"{hms(a0)}-{hms(a1)}",
                         f"{r2}_time": f"{hms(a0 + off)}-{hms(a1 + off)}",
                         "other_copy_apnea_s": round(ov, 1), "status": status})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", nargs="+", default=["c05:c06:80", "x33:x34:70", "x22:x35:40"])
    p.add_argument("--data-dir", default="data/apnea_ecg")
    p.add_argument("--out", default="results/apnea_diagnostics/duplicates/label_disagreements.csv")
    args = p.parse_args()
    root = Path(args.data_dir)
    pd.set_option("display.width", 200)

    all_rows, summary = [], []
    for spec in args.pairs:
        r1, r2, off = spec.split(":")
        df = report(root, r1, r2, float(off))
        df.columns = ["pair", "apnea_minute_in", "time_rec1", "time_rec2", "other_copy_apnea_s", "status"]
        all_rows.append(df)
        print(f"\n=== {r1} / {r2} (offset {off} s; times from the start of each record) ===")
        print(df.to_string(index=False) if len(df) else "no apnea minutes in either copy")
        counts = df.status.value_counts().to_dict()
        summary.append({"pair": f"{r1}/{r2}", "apnea_min_" + "rec1": int((df.apnea_minute_in == r1).sum()),
                        "apnea_min_rec2": int((df.apnea_minute_in == r2).sum()), **counts})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(all_rows).to_csv(out, index=False)
    print("\n=== Summary ===")
    print(pd.DataFrame(summary).fillna(0).to_string(index=False))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
