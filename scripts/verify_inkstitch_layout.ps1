$ErrorActionPreference = 'Continue'
$extDir = Join-Path $env:APPDATA 'inkscape\extensions'
Write-Host "extDir: $extDir"
Write-Host '=== top-level ==='
Get-ChildItem -LiteralPath $extDir -Force | Select-Object Name, Mode | Format-Table -AutoSize

Write-Host '=== top-level inx ==='
@(Get-ChildItem -LiteralPath $extDir -Filter '*.inx' -File -ErrorAction SilentlyContinue).Count

Write-Host '=== inkstitch tree (depth 2) ==='
$root = Join-Path $extDir 'inkstitch'
if (Test-Path -LiteralPath $root) {
  Get-ChildItem -LiteralPath $root -Force | Select-Object Name, Mode | Format-Table -AutoSize
  $inner = Join-Path $root 'inkstitch'
  if (Test-Path -LiteralPath $inner) {
    Write-Host 'inner inkstitch:'
    Get-ChildItem -LiteralPath $inner -Force | Select-Object Name, Mode | Format-Table -AutoSize
  }
}

# Inkscape 1.x discovers .inx in user extensions; nested may need inx at known places
# Official installer usually places files so that inkstitch/*.inx or root inx exist
$sampleInx = Get-ChildItem -LiteralPath $extDir -Recurse -Filter 'inkstitch_params.inx' -ErrorAction SilentlyContinue |
  Select-Object -First 3 FullName
$sampleInx | ForEach-Object { Write-Host "params:" $_.FullName }

# Quick CLI: ask inkscape to open and quit; extension load hard to query.
# Instead check preferences system path via inkscape --system-data-directory etc.
$ink = 'D:\TOOL\Inkscape\bin\inkscape.com'
if (Test-Path -LiteralPath $ink) {
  Write-Host '=== inkscape user-data ==='
  & $ink --user-data-directory 2>&1
  & $ink --system-data-directory 2>&1
}

# Log silent install
$log = 'D:\WORK\TEST\Project 1\exports\inkstitch_setup.log'
if (Test-Path -LiteralPath $log) {
  Write-Host '=== setup log (selected) ==='
  Select-String -LiteralPath $log -Pattern 'Error|Dest|folder|extensions|Failed|Successfully' |
    Select-Object -First 30 Line
}
Write-Host 'DONE'
