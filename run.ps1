param(
    [string]$GeminiApiKey = "",
    [int]$Port = 8501
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $VenvPython)) {
    throw "CornerCoach is not set up. Run .\setup.ps1 first."
}
if ($GeminiApiKey.Trim()) {
    $env:GEMINI_API_KEY = $GeminiApiKey.Trim()
}
$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS = "false"

Set-Location -LiteralPath $ProjectRoot
& $VenvPython -m streamlit run app.py --server.port $Port

