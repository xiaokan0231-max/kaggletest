$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = "D:\Users\49662\miniconda3\envs\rogii\python.exe"
$Log = Join-Path $Root "run_local_repro.log"

if (-not (Test-Path $Python)) {
    throw "Python env not found at $Python"
}

Set-Location $Root

& $Python ".\local_rogii_repro.py" 2>&1 | Tee-Object -FilePath $Log
if ($LASTEXITCODE -ne 0) {
    throw "local_rogii_repro.py failed with exit code $LASTEXITCODE"
}

Write-Host "Run log: $Log"
Write-Host "Submission: $(Join-Path $Root 'submission.csv')"
