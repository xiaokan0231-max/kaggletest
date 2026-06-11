$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Kaggle = "D:\Users\49662\miniconda3\envs\rogii\Scripts\kaggle.exe"

if (-not (Test-Path $Kaggle)) {
    throw "Kaggle CLI not found at $Kaggle"
}

Set-Location $Root

New-Item -ItemType Directory -Force -Path "data", "koolbox-offline", "artifacts", "nb" | Out-Null

& $Kaggle competitions download -c rogii-wellbore-geology-prediction -p ".\data"
$competitionZip = Get-ChildItem ".\data" -Filter "*.zip" | Select-Object -First 1
if ($competitionZip) {
    Expand-Archive -LiteralPath $competitionZip.FullName -DestinationPath ".\data" -Force
}

& $Kaggle datasets download -d phongnguyn23021656/koolbox-offline -p ".\koolbox-offline"
$koolboxZip = Get-ChildItem ".\koolbox-offline" -Filter "*.zip" | Select-Object -First 1
if ($koolboxZip) {
    Expand-Archive -LiteralPath $koolboxZip.FullName -DestinationPath ".\koolbox-offline" -Force
}

& $Kaggle datasets download -d ravaghi/wellbore-geology-prediction-artifacts -p ".\artifacts"
$artifactsZip = Get-ChildItem ".\artifacts" -Filter "*.zip" | Select-Object -First 1
if ($artifactsZip) {
    Expand-Archive -LiteralPath $artifactsZip.FullName -DestinationPath ".\artifacts" -Force
}

& $Kaggle kernels pull lightningv08/lb-7-776-rogii-ridge-sp -p ".\nb" -m

Write-Host "Downloaded and extracted Kaggle assets under $Root"
