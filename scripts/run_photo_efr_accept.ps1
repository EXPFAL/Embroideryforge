$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root
$py = Join-Path $Root '.venv\Scripts\python.exe'
$exp = Join-Path $Root 'exports\photo_exp'

Write-Host '=== portrait_naive ==='
& $py -m embroideryforge stitch-auto (Join-Path $Root 'samples\golden-portrait.jpg') `
  --out (Join-Path $exp 'portrait_naive') --domain portrait --no-efr --quality dense --bg auto --k 8
if ($LASTEXITCODE -ne 0) { throw "naive failed: $LASTEXITCODE" }

Write-Host '=== portrait_efr ==='
& $py -m embroideryforge stitch-auto (Join-Path $Root 'samples\golden-portrait.jpg') `
  --out (Join-Path $exp 'portrait_efr') --domain portrait --efr --quality dense --bg auto --k 8
if ($LASTEXITCODE -ne 0) { throw "efr failed: $LASTEXITCODE" }

Write-Host '=== landscape_efr ==='
& $py -m embroideryforge stitch-auto (Join-Path $Root 'samples\golden-landscape.jpg') `
  --out (Join-Path $exp 'landscape_efr') --domain portrait --efr --quality dense --bg auto --k 8
if ($LASTEXITCODE -ne 0) { throw "landscape failed: $LASTEXITCODE" }

Write-Host '=== compare-photo ==='
& $py -m embroideryforge compare-photo `
  (Join-Path $exp 'portrait_naive') (Join-Path $exp 'portrait_efr') `
  --out (Join-Path $exp 'compare_portrait.json')
if ($LASTEXITCODE -ne 0) { throw "compare failed: $LASTEXITCODE" }

Write-Host '=== chibi evaluate ==='
$chibiOut = Join-Path $Root 'exports\accept_flat_v3'
$ref = Join-Path $Root 'exports\inkscape_smoke\chibi_multicolor.dst'
if (-not (Test-Path -LiteralPath $chibiOut)) {
  Write-Host 'accept_flat_v3 missing; regenerating chibi baseline'
  & $py -m embroideryforge stitch-auto (Join-Path $Root 'samples\golden-chibi.jpg') `
    --out $chibiOut --quality dense --bg auto
  if ($LASTEXITCODE -ne 0) { throw "chibi regen failed: $LASTEXITCODE" }
}
$evalArgs = @('-m', 'embroideryforge', 'evaluate', $chibiOut, '--image', (Join-Path $Root 'samples\golden-chibi.jpg'))
if (Test-Path -LiteralPath $ref) { $evalArgs += @('--ref-dst', $ref) }
& $py @evalArgs
if ($LASTEXITCODE -ne 0) { throw "evaluate failed: $LASTEXITCODE" }

Write-Host 'ACCEPT_OK'
