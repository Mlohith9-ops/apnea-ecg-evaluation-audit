"""Build a minute-level feature table for the PhysioNet Apnea-ECG database.

One row per labelled minute. Rules:
  * a minute is kept only if a full 60 s of ECG exists (sample + 60*fs <= sig_len)
  * unlabelled tails are never labelled
  * R-peaks come from the database's .qrs files (machine-generated; may contain errors)
  * features use RR intervals from the minute itself and from a centred 5-minute
    window (same record only), so no information crosses between records

Usage:
  python src/build_apnea_features.py --data-dir data/apnea_ecg --map splits/apnea_ecg_subject_map.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from scipy.interpolate import interp1d
from scipy.signal import welch

RR_MIN, RR_MAX = 0.3, 2.0     # seconds; physiologically plausible beat-to-beat range
INTERP_FS = 4.0               # Hz, resampling rate for frequency-domain HRV


def rr_stats(rr, prefix):
    """Time-domain statistics of an RR series (seconds)."""
    out = {f"{prefix}n_beats": len(rr)}
    if len(rr) < 5:
        return out  # left as NaN downstream; minute is flagged
    d = np.diff(rr)
    out.update({
        f"{prefix}rr_mean": rr.mean(),
        f"{prefix}rr_median": np.median(rr),
        f"{prefix}rr_std": rr.std(ddof=1),
        f"{prefix}rr_iqr": np.subtract(*np.percentile(rr, [75, 25])),
        f"{prefix}rr_min": rr.min(),
        f"{prefix}rr_max": rr.max(),
        f"{prefix}rmssd": np.sqrt(np.mean(d ** 2)),
        f"{prefix}pnn50": np.mean(np.abs(d) > 0.05),
        f"{prefix}hr_mean": 60.0 / rr.mean(),
    })
    return out


def rr_spectral(times, rr, prefix):
    """VLF/LF/HF power of an RR series resampled to 4 Hz."""
    out = {}
    if len(rr) < 30 or times[-1] - times[0] < 120:
        return out
    t_new = np.arange(times[0], times[-1], 1.0 / INTERP_FS)
    rr_i = interp1d(times, rr, kind="linear")(t_new)
    f, p = welch(rr_i - rr_i.mean(), fs=INTERP_FS, nperseg=min(256, len(rr_i)))

    def band(lo, hi):
        m = (f >= lo) & (f < hi)
        return np.trapezoid(p[m], f[m]) if m.any() else np.nan

    vlf, lf, hf = band(0.003, 0.04), band(0.04, 0.15), band(0.15, 0.4)
    out.update({f"{prefix}vlf": vlf, f"{prefix}lf": lf, f"{prefix}hf": hf,
                f"{prefix}lf_hf": lf / hf if hf and hf > 0 else np.nan})
    return out


def process_record(root, rec):
    hdr = wfdb.rdheader(str(root / rec))
    fs = int(hdr.fs)
    win = 60 * fs

    apn = wfdb.rdann(str(root / rec), "apn")
    qrs = wfdb.rdann(str(root / rec), "qrs")
    sig = wfdb.rdrecord(str(root / rec), channels=[0]).p_signal[:, 0]

    r = np.asarray(qrs.sample)
    r = r[(r >= 0) & (r < len(sig))]
    rr = np.diff(r) / fs                    # RR interval ending at beat r[1:]
    rr_t = r[1:] / fs                       # time (s) of the beat that ends each RR
    ok = (rr >= RR_MIN) & (rr <= RR_MAX)
    rr, rr_t = rr[ok], rr_t[ok]
    r_amp = sig[r]                          # R-peak amplitude (simple EDR proxy)
    r_time = r / fs

    rows = []
    for minute, (start, sym) in enumerate(zip(apn.sample, apn.symbol)):
        if start + win > hdr.sig_len:       # incomplete final minute -> drop
            continue
        t0, t1 = start / fs, (start + win) / fs
        row = {"record": rec, "minute": minute, "label": int(sym == "A")}

        m1 = (rr_t >= t0) & (rr_t < t1)
        row.update(rr_stats(rr[m1], "m1_"))
        row["m1_rejected_frac"] = 1 - m1.sum() / max(1, ((r_time >= t0) & (r_time < t1)).sum())

        m5 = (rr_t >= t0 - 120) & (rr_t < t1 + 120)       # centred 5-minute context
        row.update(rr_stats(rr[m5], "m5_"))
        row.update(rr_spectral(rr_t[m5], rr[m5], "m5_"))

        a = (r_time >= t0) & (r_time < t1)
        if a.sum() >= 5:
            row.update({"m1_ramp_mean": r_amp[a].mean(), "m1_ramp_std": r_amp[a].std(ddof=1)})
        rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/apnea_ecg")
    p.add_argument("--map", default="splits/apnea_ecg_subject_map.csv")
    p.add_argument("--out", default="data/processed/apnea_minutes.csv")
    p.add_argument("--sets", default="abc", help="record prefixes to include, e.g. abc or abcx")
    p.add_argument("--exclude", default="",
                   help="records to leave out, e.g. c05,x34,x35 (verified duplicate copies)")
    args = p.parse_args()
    root = Path(args.data_dir)
    excluded = {r for r in args.exclude.split(",") if r}

    records = [r.strip() for r in (root / "RECORDS").read_text().split()
               if len(r.strip()) == 3 and r.strip()[0] in args.sets and r.strip() not in excluded]
    if excluded:
        print("excluded as duplicate copies:", ", ".join(sorted(excluded)))

    all_rows = []
    for rec in records:
        rows = process_record(root, rec)
        all_rows.extend(rows)
        print(f"{rec}: {len(rows)} minutes kept")

    df = pd.DataFrame(all_rows)
    subj = pd.read_csv(args.map)[["record", "inferred_subject"]]
    df = df.merge(subj, on="record", how="left", validate="many_to_one")
    if df["inferred_subject"].isna().any():
        raise SystemExit("Some records have no subject group in the map - fix the map first.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"\nRows: {len(df)} | records: {df.record.nunique()} | subject groups: {df.inferred_subject.nunique()}")
    print(f"Apnea minutes: {df.label.sum()} ({df.label.mean():.1%})")
    print(f"Minutes with <5 clean beats: {(df['m1_n_beats'] < 5).sum()}")
    print(f"Median fraction of beats rejected per minute: {df['m1_rejected_frac'].median():.3f}")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
