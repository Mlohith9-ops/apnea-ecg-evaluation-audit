"""Signal-quality audit for one Apnea-ECG record, compared with records of the same class.

Writes:
  results/audit_<rec>/quality_vs_peers.csv   beat/RR quality stats, this record vs its class
  results/audit_<rec>/tachogram.png          whole-night RR intervals (raw .qrs beats)
  results/audit_<rec>/ecg_snippets.png       4 random 10-s ECG windows with .qrs marks

Usage:
  python src/audit_record.py --record c03
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wfdb

RR_MIN, RR_MAX = 0.3, 2.0


def quality(root, rec):
    hdr = wfdb.rdheader(str(root / rec))
    fs = int(hdr.fs)
    r = np.asarray(wfdb.rdann(str(root / rec), "qrs").sample)
    rr = np.diff(r) / fs
    ok = (rr >= RR_MIN) & (rr <= RR_MAX)
    d = np.diff(rr[ok])
    return {
        "record": rec,
        "hours": hdr.sig_len / fs / 3600,
        "beats": len(r),
        "beats_per_min": len(r) / (hdr.sig_len / fs / 60),
        "rr_rejected_frac": 1 - ok.mean(),
        "rr_mean_s": rr[ok].mean(),
        "rr_std_s": rr[ok].std(),
        "big_jumps_frac": (np.abs(d) > 0.3).mean(),   # successive-RR jumps > 300 ms: ectopics / missed beats
        "longest_gap_s": rr.max(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--record", default="c03")
    p.add_argument("--data-dir", default="data/apnea_ecg")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    root = Path(args.data_dir)
    rec = args.record
    out = Path(args.out or f"results/audit_{rec}")
    out.mkdir(parents=True, exist_ok=True)

    peers = [r.strip() for r in (root / "RECORDS").read_text().split()
             if len(r.strip()) == 3 and r.strip()[0] == rec[0]]
    q = pd.DataFrame([quality(root, r) for r in peers]).set_index("record")
    q.loc["peer_median"] = q.drop(index=rec).median()
    q.round(4).to_csv(out / "quality_vs_peers.csv")
    pd.set_option("display.width", 200)
    print(q.loc[[rec, "peer_median"]].round(4).T.to_string())

    hdr = wfdb.rdheader(str(root / rec))
    fs = int(hdr.fs)
    r = np.asarray(wfdb.rdann(str(root / rec), "qrs").sample)
    rr = np.diff(r) / fs
    plt.figure(figsize=(14, 3.5))
    plt.plot(r[1:] / fs / 3600, rr, ".", ms=1)
    plt.axhspan(RR_MIN, RR_MAX, color="green", alpha=0.06, label="kept range")
    plt.ylim(0, 3)
    plt.xlabel("hours")
    plt.ylabel("RR interval (s)")
    plt.title(f"{rec}: RR tachogram from .qrs beats")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out / "tachogram.png", dpi=130)

    sig = wfdb.rdrecord(str(root / rec), channels=[0]).p_signal[:, 0]
    rng = np.random.default_rng(0)
    starts = np.sort(rng.integers(0, len(sig) - 10 * fs, 4))
    fig, axes = plt.subplots(4, 1, figsize=(14, 9))
    for ax, s in zip(axes, starts):
        t = np.arange(s, s + 10 * fs)
        ax.plot(t / fs, sig[t], lw=0.8)
        beats = r[(r >= s) & (r < s + 10 * fs)]
        ax.plot(beats / fs, sig[beats], "rv", ms=6)
        ax.set_title(f"{rec}, minute {s // (60 * fs)}: {len(beats)} detected beats in 10 s", fontsize=9)
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(out / "ecg_snippets.png", dpi=130)
    print(f"\nPlots and table saved to {out}")


if __name__ == "__main__":
    main()
