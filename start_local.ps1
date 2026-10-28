Set-Location -Path $PSScriptRoot
Write-Host "Starting FinDoc AI locally..." -ForegroundColor Cyan
& "$PSScriptRoot\venv\Scripts\python.exe" "$PSScriptRoot\run_app.py"

