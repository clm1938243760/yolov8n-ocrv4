param(
  [int]$Port = 5002
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Root ".venv311\Scripts\python.exe"
$App = Join-Path $Root "window_detect_all_api.py"
$Log = Join-Path $Root "api_server.log"
$ErrLog = Join-Path $Root "api_server.err.log"

if (-not (Test-Path $Python)) {
  throw "Python venv not found: $Python"
}
if (-not (Test-Path $App)) {
  throw "API script not found: $App"
}

Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue | ForEach-Object {
  try {
    Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue
  } catch {}
}

Remove-Item $Log, $ErrLog -ErrorAction SilentlyContinue

$env:VISION_API_PORT = [string]$Port
Start-Process `
  -WindowStyle Hidden `
  -WorkingDirectory $Root `
  -FilePath $Python `
  -ArgumentList $App `
  -RedirectStandardOutput $Log `
  -RedirectStandardError $ErrLog

Start-Sleep -Seconds 8
curl.exe -sS "http://127.0.0.1:$Port/health"
