# Data

This repository does not include the dataset. The analysis uses the **PhysioNet Apnea-ECG
Database, version 1.0.0** (70 single-lead ECG recordings at 100 Hz, with per-minute apnea
annotations).

## Download

Install the AWS CLI. No AWS account is needed for this public bucket. Then run from the
repository root:

```powershell
aws s3 sync --no-sign-request s3://physionet-open/apnea-ecg/1.0.0/ data/apnea_ecg/
```

`download_apnea.ps1` in the repository root runs the same command. The scripts expect the
files in `data/apnea_ecg/` (for example `data/apnea_ecg/a01.dat`, `a01.hea`, `a01.apn`,
`a01.qrs`, plus `RECORDS` and `additional-information.txt`). To check file integrity, compare
against the `SHA256SUMS.txt` file that comes with the download.

Derived feature tables are written to `data/processed/` by `src/build_apnea_features.py`. They
are not tracked in git.

## Licence

The Apnea-ECG Database is distributed under the **Open Data Commons Attribution License v1.0**
(ODC-By 1.0). See the dataset page for the licence terms:
https://physionet.org/content/apnea-ecg/1.0.0/

## Citation

If you use the dataset, cite:

- T. Penzel, G. B. Moody, R. G. Mark, A. L. Goldberger, J. H. Peter. "The Apnea-ECG Database."
  *Computers in Cardiology*, 2000, 27:255-258.
- T. Pollard, B. E. Moody, L. Lehman, G. Gow, C. Fernandes, C. Xie, A. Johnson, R. G. Mark,
  T. Heldt. "PhysioNet as a global platform for biomedical research." *Nature Health*, 2026.

Also cite the dataset itself with its DOI. Copy the DOI from the dataset page,
https://physionet.org/content/apnea-ecg/1.0.0/

The additional per-record information (`additional-information.txt`) was provided to PhysioNet
by Dr. Thomas Penzel, as stated in that file.
