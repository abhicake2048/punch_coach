param(
    [string]$PythonCommand = "python"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $ProjectRoot

if (-not (Test-Path -LiteralPath ".venv\Scripts\python.exe")) {
    Write-Host "Creating .venv..."
    & $PythonCommand -m venv .venv
}

$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
Write-Host "Installing CornerCoach dependencies..."
& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r requirements.txt

$RequiredWeights = @(
    "weights\yolo11s-pose.pt",
    "weights\stgcn\best_checkpoint.pt"
)
$MissingWeights = @($RequiredWeights | Where-Object {
    -not (Test-Path -LiteralPath (Join-Path $ProjectRoot $_))
})
if ($MissingWeights.Count -gt 0) {
    throw "Missing production weights: $($MissingWeights -join ', ')"
}

Write-Host "Setup complete. Start the app with: .\run.ps1"
