$ErrorActionPreference = "Stop"

aws s3 sync --no-sign-request `
  s3://physionet-open/apnea-ecg/1.0.0/ `
  "data/apnea_ecg/"

if ($LASTEXITCODE -ne 0) {
    throw "Download failed. Run this script again to retry."
}

Write-Host "`nDownload command finished."
Get-ChildItem "data\apnea_ecg" -Recurse -File |
    Select-Object Name, Length
