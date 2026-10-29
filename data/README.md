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

The downloaded files do not include a licence file. Check the PhysioNet page for the current
licence and terms of use before redistributing anything derived from the data:
https://physionet.org/content/apnea-ecg/1.0.0/

## Citation

Check the PhysioNet page above for the required citation; the downloaded files do not state
one. The database is usually cited with its original paper and the PhysioNet resource paper:

- Penzel T, Moody GB, Mark RG, Goldberger AL, Peter JH. The Apnea-ECG Database. Computers in
  Cardiology 2000;27:255-258.
- Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG, Mietus JE, Moody GB,
  Peng C-K, Stanley HE. PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research
  resource for complex physiologic signals. Circulation 2000;101(23):e215-e220.

The additional per-record information (`additional-information.txt`) was provided to PhysioNet
by Dr. Thomas Penzel, as stated in that file.
