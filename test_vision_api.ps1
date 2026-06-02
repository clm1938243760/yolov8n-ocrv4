param(
  [string]$Image = "D:\Documents\New project\vision_probe\current.jpg",
  [string]$BaseUrl = "http://127.0.0.1:5002",
  [string]$Software = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Request = Join-Path $Root "api_test_request.json"
$WindowResponse = Join-Path $Root "api_test_window_response.json"
$IconResponse = Join-Path $Root "api_test_icon_response.json"

if (-not (Test-Path $Image)) {
  throw "Image not found: $Image"
}
if (-not $Software) {
  throw "Software is required for icon locate test. Pass -Software '<name>'."
}

$b64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes($Image))

@{ image_base64 = $b64 } |
  ConvertTo-Json -Compress |
  Set-Content -Encoding UTF8 $Request

"window:"
curl.exe -sS -X POST "$BaseUrl/window/detect" `
  -H "Content-Type: application/json" `
  --data-binary "@$Request" |
  Tee-Object $WindowResponse

""
@{ image_base64 = $b64; software = $Software } |
  ConvertTo-Json -Compress |
  Set-Content -Encoding UTF8 $Request

"icon:"
curl.exe -sS -X POST "$BaseUrl/icon/locate" `
  -H "Content-Type: application/json" `
  --data-binary "@$Request" |
  Tee-Object $IconResponse

""
"window_response=$WindowResponse"
"icon_response=$IconResponse"
