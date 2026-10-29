"""Check whether two Apnea-ECG records contain the same raw ECG (a copied recording).

For each pair: take several 5-minute windows from record 1 and search record 2 for the
best-matching position (normalised cross-correlation of the raw ECG, search +-SEARCH_MIN
minutes around the lag found by diagnose_overlap.py). A copy of the same recording gives
correlations near 1 at a consistent offset; two different nights give low, inconsistent values.

Also saves an overlay plot of 10 s of both signals at the best offset.

Usage:
  python src/verify_duplicates.py --pairs c05:c06:-1 x33:x34:-1 x22:x35:-1 a04:a12:35
  (format rec1:rec2:lag_minutes; c05:c06 is the documented positive control)
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import correlate

WIN_MIN = 5
SEARCH_MIN = 6
N_WINDOWS = 5


def load(root, rec):
    r = wfdb.rdrecord(str(root / rec), channels=[0])
    return r.p_signal[:, 0].astype(float), int(r.fs)


def ncc_search(window, region):
    """Max normalised cross-correlation of window slid over region; returns (corr, offset)."""
    w = (window - window.mean()) / (window.std() + 1e-12)
    n = len(w)
    num = correlate(region, w, mode="valid", method="fft")
    # rolling std/mean of region under the window
    c1 = np.cumsum(np.insert(region, 0, 0.0))
    c2 = np.cumsum(np.insert(region ** 2, 0, 0.0))
    s1 = c1[n:] - c1[:-n]
    s2 = c2[n:] - c2[:-n]
    var = np.maximum(s2 / n - (s1 / n) ** 2, 1e-12)
    ncc = (num - s1 / n * w.sum()) / (n * np.sqrt(var))
    k = int(np.nanargmax(ncc))
    return float(ncc[k]), k


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", nargs="+", default=["c05:c06:-1", "x33:x34:-1", "x22:x35:-1", "a04:a12:35"])
    p.add_argument("--data-dir", default="data/apnea_ecg")
    p.add_argument("--out", default="results/apnea_diagnostics/duplicates")
    args = p.parse_args()
    root, out = Path(args.data_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    rows = []
    for spec in args.pairs:
        r1, r2, lag = spec.split(":")
        lag = int(lag)
        s1, fs = load(root, r1)
        s2, _ = load(root, r2)
        win, search = WIN_MIN * 60 * fs, SEARCH_MIN * 60 * fs
        results = []
        lo_start = max(0, -lag * 60 * fs)
        hi_start = min(len(s1), len(s2) - lag * 60 * fs) - win
        if hi_start <= lo_start:
            rows.append({"pair": f"{r1}-{r2}", "note": "no overlap at this lag"})
            continue
        for start in np.sort(rng.integers(lo_start, hi_start, N_WINDOWS)):
            centre = start + lag * 60 * fs
            a, b = max(0, centre - search), min(len(s2), centre + win + search)
            if b - a <= win:
                continue
            corr, k = ncc_search(s1[start:start + win], s2[a:b])
            results.append((start, corr, (a + k - start) / fs))
        if not results:
            rows.append({"pair": f"{r1}-{r2}", "note": "windows out of range"})
            continue
        corrs = np.array([c for _, c, _ in results])
        offs = np.array([o for _, _, o in results])
        rows.append({"pair": f"{r1}-{r2}", "windows": len(results),
                     "median_corr": round(float(np.median(corrs)), 3),
                     "min_corr": round(float(corrs.min()), 3),
                     "offset_s_median": round(float(np.median(offs)), 2),
                     "offset_s_spread": round(float(offs.max() - offs.min()), 2)})

        start, _, off = results[int(np.argmax(corrs))]
        j = start + int(round(off * fs))
        t = np.arange(10 * fs) / fs
        plt.figure(figsize=(14, 3))
        plt.plot(t, s1[start:start + 10 * fs], lw=0.9, label=r1)
        plt.plot(t, s2[j:j + 10 * fs], lw=0.9, alpha=0.7, label=f"{r2} (offset {off:.2f} s)")
        plt.legend(loc="upper right")
        plt.title(f"{r1} vs {r2}: best-matching 10 s")
        plt.xlabel("s")
        plt.tight_layout()
        plt.savefig(out / f"overlay_{r1}_{r2}.png", dpi=120)
        plt.close()
        print(f"{r1}-{r2} done")

    res = pd.DataFrame(rows)
    res.to_csv(out / "duplicate_check.csv", index=False)
    pd.set_option("display.width", 200)
    print("\nRaw-ECG match (copy of one recording: corr near 1, offset spread near 0):\n")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
