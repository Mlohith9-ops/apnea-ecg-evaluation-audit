# =====================================================================================
# WARNING: running this script performs NEW experiments.
#   * It retrains every model and runs a NEW evaluation on the x test set (the official
#     Apnea-ECG test records). The x set is the final test set: run the official split
#     once per fixed configuration and never tune anything on it.
#   * All output goes to NEW folders, results\repro_<tag>\ and data\processed\repro_<tag>\.
#     The reported results folders are never touched. The script stops if a target
#     folder already exists.
#   * Runtime: about 15-25 minutes (the baseline step alone takes 10-15 minutes).
# =====================================================================================
#
# Usage (from anywhere; the script runs from the repository root):
#   .\reproduce.ps1                 # asks for confirmation; tag = current timestamp
#   .\reproduce.ps1 -Tag run2 -Yes  # no prompt; writes results\repro_run2\
#
# Requires the dataset in data\apnea_ecg\ (see data\README.md) and requirements.txt installed.

param(
    [string]$Tag = (Get-Date -Format "yyyyMMdd_HHmmss"),
    [switch]$Yes
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$R = "results\repro_$Tag"           # new results root
$D = "data\processed\repro_$Tag"    # new feature tables

foreach ($p in $R, $D) {
    if (Test-Path $p) { throw "$p already exists; choose a new -Tag (earlier runs are never overwritten)" }
}
if (-not (Test-Path "data\apnea_ecg\RECORDS")) { throw "data\apnea_ecg\ not found; download it first (see data\README.md)" }

Write-Warning "This runs NEW experiments, including a new evaluation on the x test set. Output: $R and $D"
if (-not $Yes) {
    $answer = Read-Host "Type YES to continue"
    if ($answer -ne "YES") { Write-Host "Cancelled."; exit 1 }
}

function Invoke-Step([string]$Title, [string[]]$PyArgs) {
    Write-Host "`n=== $Title" -ForegroundColor Cyan
    Write-Host "python $($PyArgs -join ' ')"
    & python @PyArgs
    if ($LASTEXITCODE -ne 0) { throw "Step failed: $Title" }
}

function Assert-Counts([string]$Csv, [int]$Records, [int]$Groups, [int]$Rows = -1, [int]$Apnea = -1) {
    # Stop if a feature table does not have the expected records/groups/rows/apnea minutes.
    $code = @"
import sys, pandas as pd
df = pd.read_csv(sys.argv[1], usecols=['record', 'label', 'inferred_subject'])
got = (df.record.nunique(), df.inferred_subject.nunique(), len(df), int(df.label.sum()))
want = tuple(int(v) for v in sys.argv[2:])
print(f'{sys.argv[1]}: {got[0]} records, {got[1]} subject groups, {got[2]} rows, {got[3]} apnea minutes')
sys.exit(0 if all(w < 0 or g == w for g, w in zip(got, want)) else 1)
"@
    & python -c $code $Csv $Records $Groups $Rows $Apnea
    if ($LASTEXITCODE -ne 0) { throw "Unexpected counts in $Csv (expected $Records records, $Groups groups, rows $Rows, apnea $Apnea; -1 = not checked)" }
}

New-Item -ItemType Directory -Path $R | Out-Null

# 1. Label/annotation check for all 70 records
Invoke-Step "Label check" @("src\verify_apnea_labels.py", "--out", "$R\apnea_label_check.csv")

# 2. Feature tables, keeping one copy of each duplicate recording
Invoke-Step "Features: learning set without c05" @(
    "src\build_apnea_features.py", "--exclude", "c05", "--out", "$D\apnea_minutes.csv")
Assert-Counts "$D\apnea_minutes.csv" 34 23 16558 6508

Invoke-Step "Features: all sets without c05, x34, x35" @(
    "src\build_apnea_features.py", "--sets", "abcx", "--exclude", "c05,x34,x35",
    "--out", "$D\apnea_minutes_all.csv")
Assert-Counts "$D\apnea_minutes_all.csv" 67 29

# 3. Learning-set cross-validation (minute-wise leaky diagnostic, LORO, LOSO)
Invoke-Step "Baseline" @(
    "src\baseline_apnea.py", "--features", "$D\apnea_minutes.csv", "--out", "$R\apnea_baseline")

# 4. Official split: train a/b/c, test x (NEW evaluation on the test set)
Invoke-Step "Official split" @(
    "src\official_split_apnea.py", "--features", "$D\apnea_minutes_all.csv",
    "--out", "$R\apnea_official")

# 5. LOSO vs x gap analysis on the predictions just produced
Invoke-Step "Gap analysis" @(
    "src\explain_gap.py",
    "--learning", "$R\apnea_baseline\oof_predictions.csv",
    "--test", "$R\apnea_official\x_predictions_hist_gb.csv",
    "--out", "$R\apnea_gap_analysis")

# 6. Duplicate checks. The RR-series screen needs all 70 records, so build a separate
#    table without exclusions (used only here).
Invoke-Step "Features: all 70 records (duplicate screen only)" @(
    "src\build_apnea_features.py", "--sets", "abcx", "--out", "$D\apnea_minutes_all70.csv")
Assert-Counts "$D\apnea_minutes_all70.csv" 70 29

Invoke-Step "Duplicate screen (RR series, all pairs)" @(
    "src\diagnose_overlap.py", "--features", "$D\apnea_minutes_all70.csv",
    "--preds", "$R\apnea_official\x_predictions_logreg.csv", "--out", "$R\apnea_diagnostics")
Invoke-Step "Raw-ECG duplicate check" @(
    "src\verify_duplicates.py", "--out", "$R\apnea_diagnostics\duplicates")
Invoke-Step "Full alignment of copies" @(
    "src\full_alignment.py", "--out", "$R\apnea_diagnostics\duplicates\full_alignment.csv")
Invoke-Step "Label disagreement between copies" @(
    "src\label_disagreement_report.py", "--out", "$R\apnea_diagnostics\duplicates\label_disagreements.csv")

Write-Host "`nDone. New results in $R ; new feature tables in $D" -ForegroundColor Green
