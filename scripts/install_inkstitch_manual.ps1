$ErrorActionPreference = 'Stop'

$installer = 'C:\Users\Administrator\Downloads\inkstitch-v3.3.0-windows-64bit.exe'
$seven = 'C:\Program Files\7-Zip\7z.exe'
$extDir = Join-Path $env:APPDATA 'inkscape\extensions'
$stage = 'D:\WORK\TEST\Project 1\exports\_inkstitch_extract'

if (-not (Test-Path -LiteralPath $installer)) { throw "missing installer: $installer" }
if (-not (Test-Path -LiteralPath $seven)) { throw "missing 7-Zip" }
if (-not (Test-Path -LiteralPath $extDir)) {
    New-Item -ItemType Directory -Path $extDir -Force | Out-Null
}

if (Test-Path -LiteralPath $stage) { Remove-Item -LiteralPath $stage -Recurse -Force }
New-Item -ItemType Directory -Path $stage -Force | Out-Null

Write-Host "Extracting installer with 7-Zip..."
& $seven x $installer "-o$stage" -y | Out-Host

Write-Host "=== staged top ==="
Get-ChildItem -LiteralPath $stage -Force | Select-Object Name, Mode, Length | Format-Table -AutoSize

# Find payload: usually a folder with inkstitch.inx / inkstitch/
$payload = $null
$inx = Get-ChildItem -LiteralPath $stage -Recurse -Filter '*inkstitch*.inx' -ErrorAction SilentlyContinue |
    Select-Object -First 5
$inx | ForEach-Object { Write-Host "inx:" $_.FullName }
if ($inx) {
    # parent that looks like extension root (contains inx + inkstitch folder or py)
    $payload = $inx[0].Directory.FullName
}

# Also check for {app} style inno extract
$appDir = Get-ChildItem -LiteralPath $stage -Directory -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match 'app|inkstitch|files' } |
    Select-Object -First 5
$appDir | ForEach-Object { Write-Host "dir:" $_.FullName }

Write-Host "payload_guess: $payload"

if (-not $payload) {
    # dump tree depth 3
    Get-ChildItem -LiteralPath $stage -Recurse -Depth 2 | Select-Object -First 80 FullName
    throw 'Could not locate Ink/Stitch payload in extracted installer'
}

# Copy contents of payload into user extensions
Write-Host "Installing into: $extDir"
Copy-Item -LiteralPath (Join-Path $payload '*') -Destination $extDir -Recurse -Force

$checkInx = @(Get-ChildItem -LiteralPath $extDir -Filter '*inkstitch*.inx' -ErrorAction SilentlyContinue)
$checkDir = @(Get-ChildItem -LiteralPath $extDir -Directory -ErrorAction SilentlyContinue | Where-Object { $_.Name -match 'inkstitch' })
Write-Host "installed_inx:" $checkInx.Count
Write-Host "installed_dirs:" ($checkDir | ForEach-Object { $_.Name } -join ', ')

if ($checkInx.Count -lt 1 -and $checkDir.Count -lt 1) {
    throw 'Copy finished but Ink/Stitch files not detected'
}

# Verify Inkscape can list extension (optional CLI)
$ink = 'D:\TOOL\Inkscape\bin\inkscape.com'
if (Test-Path -LiteralPath $ink) {
    Write-Host "Inkscape OK at $ink"
}

Write-Host 'RESULT: OK — restart Inkscape, then check Extensions > Ink/Stitch'
Write-Host 'DONE'
