$ErrorActionPreference = 'Stop'
$extDir = Join-Path $env:APPDATA 'inkscape\extensions'
$inxDir = Join-Path $extDir 'inkstitch\inx'

Write-Host "extDir: $extDir"
Write-Host "inxDir exists:" (Test-Path -LiteralPath $inxDir)

if (-not (Test-Path -LiteralPath $inxDir)) {
    throw "missing $inxDir — reinstall inkstitch first"
}

# Inkscape loads *.inx from the user extensions folder (and typically immediate content).
# Official layout often also needs loader scripts beside inx. Check for inkstitch.py / next to inx.
Write-Host '=== sample of inx dir ==='
Get-ChildItem -LiteralPath $inxDir | Select-Object -First 15 Name

# Many packaged builds put a thin wrapper: extensions/inkstitch*.inx + inkstitch/ package
# If inx are only under inkstitch/inx, copy/link them to extensions root OR one level down.

# Check whether .inx reference paths assume being in extensions root
$sample = Get-ChildItem -LiteralPath $inxDir -Filter '*.inx' | Select-Object -First 1
$content = Get-Content -LiteralPath $sample.FullName -Raw -Encoding UTF8
Write-Host "sample_inx:" $sample.Name
# show script/command hints
if ($content -match '<command[^>]*>([^<]+)</command>') { Write-Host "command:" $Matches[1] }
if ($content -match 'location=["'']([^"'']+)["'']') { Write-Host "location:" $Matches[1] }
Select-String -Path $sample.FullName -Pattern 'command|script|inkstitch' | Select-Object -First 8 Line

# Prefer: create junctions/copies of all inx into extensions root pointing content is wrong for path.
# Better approach used by installer: place inx files in extensions/ AND keep package in inkstitch/
# Inspect a few inx for relative path to bin\inkstitch.exe or python

$params = Join-Path $inxDir 'inkstitch_params.inx'
$raw = Get-Content -LiteralPath $params -Raw -Encoding UTF8
Write-Host '--- params.inx head ---'
($raw -split "`n" | Select-Object -First 40) -join "`n" | Write-Host

# Copy all inx to user extensions root (standard discovery), and also copy any .py next to them if present in inx folder
Write-Host 'Copying *.inx to extensions root for discovery...'
Copy-Item -LiteralPath (Join-Path $inxDir '*.inx') -Destination $extDir -Force

$rootInx = @(Get-ChildItem -LiteralPath $extDir -Filter 'inkstitch*.inx' -File)
Write-Host "root_inkstitch_inx_count:" $rootInx.Count

# Ensure inkstitch executable exists
$bin = Join-Path $extDir 'inkstitch\bin'
Write-Host "bin exists:" (Test-Path -LiteralPath $bin)
if (Test-Path -LiteralPath $bin) {
    Get-ChildItem -LiteralPath $bin | Select-Object -First 10 Name | Format-Table -AutoSize
}

Write-Host 'DONE — fully quit Inkscape (all windows), reopen, create new document, then check 扩展 menu'
