import argparse
from pathlib import Path
import pandas as pd
import wfdb

p = argparse.ArgumentParser()
p.add_argument("--data-dir", default="data/apnea_ecg")
p.add_argument("--out", default="results/apnea_label_check.csv")
args = p.parse_args()
root = Path(args.data_dir)

records = [
    r.strip()
    for r in (root / "RECORDS").read_text().split()
    if len(r.strip()) == 3 and r.strip()[0] in "abcx"
]

rows = []
for rec in records:
    row = {"record": rec}
    try:
        hdr = wfdb.rdheader(str(root / rec))
        row["fs"] = hdr.fs
        row["signal_minutes"] = hdr.sig_len // (hdr.fs * 60)

        ann_path = root / f"{rec}.apn"
        if not ann_path.exists():
            row["status"] = "NO_LOCAL_APN_FILE"
        else:
            ann = wfdb.rdann(str(root / rec), "apn")
            syms = pd.Series(ann.symbol)
            row.update({
                "status": "OK",
                "labelled_minutes": len(syms),
                "apnea_A": int((syms == "A").sum()),
                "normal_N": int((syms == "N").sum()),
                "other_symbols": sorted(set(syms) - {"A", "N"}),
                "first_sample": int(ann.sample[0]) if len(ann.sample) else None,
                "regular_60s_steps": bool(
                    (pd.Series(ann.sample).diff().dropna()
                     == hdr.fs * 60).all()
                ),
            })
    except Exception as e:
        row["status"] = "ERROR"
        row["error"] = str(e)
    rows.append(row)

df = pd.DataFrame(rows)
if "labelled_minutes" in df.columns:
    df["count_gap"] = df["signal_minutes"] - df["labelled_minutes"]

out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)

print(df.to_string(index=False))
print("\nRecords checked:", len(df))
print("Rows with errors:", int((df["status"] == "ERROR").sum()))
print("Rows with local annotations:", int((df["status"] == "OK").sum()))
print("CSV saved to:", out)
