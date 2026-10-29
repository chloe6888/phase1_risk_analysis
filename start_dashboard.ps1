$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    py -3 -m venv .venv
}

& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv\Scripts\python.exe" -m pip install -r requirements-dashboard.txt
New-Item -ItemType Directory -Force -Path "data\private" | Out-Null

if (-not (Test-Path "data\private\holdings_snapshot.csv")) {
    Write-Host "No private holdings CSV found."
    Write-Host "Copy your personal CSV to data\private\holdings_snapshot.csv or upload it in the dashboard."
}

& ".\.venv\Scripts\python.exe" -m streamlit run dashboard.py --server.address 127.0.0.1
