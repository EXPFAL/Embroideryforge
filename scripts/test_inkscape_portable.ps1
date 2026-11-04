$ErrorActionPreference = 'Continue'

$toolRoot = 'D:\TOOL\InkScape'
$zipPath = 'C:\Users\Administrator\Downloads\inkscape-1.4.4_2026-05-05_dcaf3e7-x64.7z'
$golden = 'D:\WORK\TEST\Project 1\samples\golden-logo.jpg'
$outDir = 'D:\WORK\TEST\Project 1\exports\inkscape_smoke'
$outPng = Join-Path $outDir 'golden_from_inkscape.png'
$outSvg = Join-Path $outDir 'golden_trace_probe.svg'

Write-Host "=== paths ==="
Write-Host "toolRoot exists:" (Test-Path -LiteralPath $toolRoot)
Write-Host "zip exists:" (Test-Path -LiteralPath $zipPath)
if (Test-Path -LiteralPath $zipPath) {
    $z = Get-Item -LiteralPath $zipPath
    Write-Host "zip_mb:" ([math]::Round($z.Length / 1MB, 2))
}

$exeCandidates = @()
if (Test-Path -LiteralPath $toolRoot) {
    $exeCandidates = @(
        Get-ChildItem -LiteralPath $toolRoot -Recurse -Filter 'inkscape.exe' -ErrorAction SilentlyContinue |
            Select-Object -ExpandProperty FullName
    )
}
Write-Host "inkscape.exe count:" $exeCandidates.Count
$exeCandidates | ForEach-Object { Write-Host "  $_" }

if ($exeCandidates.Count -eq 0) {
    Write-Host "FAIL: no inkscape.exe under $toolRoot"
    # show top-level layout for diagnosis
    if (Test-Path -LiteralPath $toolRoot) {
        Get-ChildItem -LiteralPath $toolRoot -Force | Select-Object Name, Mode, Length
    }
    exit 1
}

$ink = $exeCandidates[0]
Write-Host "using:" $ink

Write-Host "=== --version ==="
& $ink --version
Write-Host "version_exit:" $LASTEXITCODE

New-Item -ItemType Directory -Path $outDir -Force | Out-Null

if (-not (Test-Path -LiteralPath $golden)) {
    Write-Host "WARN: golden sample missing:" $golden
    exit 2
}

# Export/convert smoke: import raster and write PNG via Inkscape CLI
Write-Host "=== convert golden jpg -> png ==="
& $ink $golden --export-type=png --export-filename=$outPng
Write-Host "convert_exit:" $LASTEXITCODE
Write-Host "out_png_exists:" (Test-Path -LiteralPath $outPng)
if (Test-Path -LiteralPath $outPng) {
    Write-Host "out_png_kb:" ([math]::Round((Get-Item -LiteralPath $outPng).Length / 1KB, 1))
}

# Also try writing SVG wrapper/export if supported
Write-Host "=== export-type svg probe ==="
& $ink $golden --export-type=svg --export-filename=$outSvg
Write-Host "svg_exit:" $LASTEXITCODE
Write-Host "out_svg_exists:" (Test-Path -LiteralPath $outSvg)

if ((Test-Path -LiteralPath $outPng) -and ($LASTEXITCODE -eq 0 -or (Test-Path -LiteralPath $outPng))) {
    Write-Host "RESULT: OK portable Inkscape usable"
    exit 0
}

Write-Host "RESULT: PARTIAL/FAIL — see exits above"
exit 3
