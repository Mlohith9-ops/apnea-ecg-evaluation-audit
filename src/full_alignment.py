"""Full-record comparison of duplicate candidates after alignment.

For each pair rec1:rec2:offset_s (rec2 sample = rec1 sample + offset), over the WHOLE overlap:
  * overlap length, and the length of each record outside the overlap
  * fraction of digital samples that are exactly identical (raw ADC integers)
  * correlation and max absolute difference of the physical signal
  * agreement of the minute-by-minute apnea labels, compared second by second
    (minute boundaries are shifted by the offset, so labels are compared at each second)
  * whether the .qrs beat times line up (fraction of beats matched within 2 samples)

Usage:
  python src/full_alignment.py --pairs c05:c06:80 x33:x34:70 x22:x35:40
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb


def compare(root, r1, r2, off_s):
    h1, h2 = wfdb.rdrecord(str(root / r1), physical=False), wfdb.rdrecord(str(root / r2), physical=False)
    fs = int(h1.fs)
    d1, d2 = h1.d_signal[:, 0].astype(np.int64), h2.d_signal[:, 0].astype(np.int64)
    off = int(round(off_s * fs))
    i0, j0 = max(0, -off), max(0, off)                 # rec1 index i <-> rec2 index i + off
    n = min(len(d1) - i0, len(d2) - j0)
    a, b = d1[i0:i0 + n], d2[j0:j0 + n]
    p1 = wfdb.rdrecord(str(root / r1)).p_signal[i0:i0 + n, 0]
    p2 = wfdb.rdrecord(str(root / r2)).p_signal[j0:j0 + n, 0]

    # labels per second
    def per_second(rec, length):
        ann = wfdb.rdann(str(root / rec), "apn")
        lab = np.full(length // fs, -1)
        for s, sym in zip(ann.sample, ann.symbol):
            lab[s // fs: s // fs + 60] = int(sym == "A")
        return lab
    l1 = per_second(r1, len(d1))[i0 // fs: (i0 + n) // fs]
    l2 = per_second(r2, len(d2))[j0 // fs: (j0 + n) // fs]
    m = min(len(l1), len(l2))
    l1, l2 = l1[:m], l2[:m]
    both = (l1 >= 0) & (l2 >= 0)
    disagree = both & (l1 != l2)

    q1 = np.asarray(wfdb.rdann(str(root / r1), "qrs").sample) + off
    q2 = np.asarray(wfdb.rdann(str(root / r2), "qrs").sample)
    q1 = q1[(q1 >= j0) & (q1 < j0 + n)]
    idx = np.searchsorted(q2, q1).clip(1, len(q2) - 1)
    nearest = np.minimum(np.abs(q2[idx] - q1), np.abs(q2[idx - 1] - q1))

    return {
        "pair": f"{r1}-{r2}", "offset_s": off_s,
        "overlap_min": round(n / fs / 60, 1),
        f"outside_overlap_min_rec1": round((len(d1) - n) / fs / 60, 1),
        f"outside_overlap_min_rec2": round((len(d2) - n) / fs / 60, 1),
        "identical_samples_frac": round(float((a == b).mean()), 5),
        "corr_full": round(float(np.corrcoef(p1, p2)[0, 1]), 5),
        "max_abs_diff_mV": round(float(np.abs(p1 - p2).max()), 4),
        "label_seconds_compared": int(both.sum()),
        "label_disagree_frac": round(float(disagree.sum() / max(1, both.sum())), 4),
        "label_disagree_min_approx": round(disagree.sum() / 60, 1),
        "apnea_min_rec1_overlap": round((l1[both] == 1).sum() / 60, 1),
        "apnea_min_rec2_overlap": round((l2[both] == 1).sum() / 60, 1),
        "qrs_matched_within_2_samples": round(float((nearest <= 2).mean()), 4),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", nargs="+", default=["c05:c06:80", "x33:x34:70", "x22:x35:40"])
    p.add_argument("--data-dir", default="data/apnea_ecg")
    p.add_argument("--out", default="results/apnea_diagnostics/duplicates/full_alignment.csv")
    args = p.parse_args()
    root = Path(args.data_dir)
    rows = [compare(root, *s.split(":")[:2], float(s.split(":")[2])) for s in args.pairs]
    res = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out, index=False)
    print(res.set_index("pair").T.to_string())


if __name__ == "__main__":
    main()
