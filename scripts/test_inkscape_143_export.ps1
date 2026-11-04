$ErrorActionPreference = 'Continue'
$bin = 'D:\TOOL\Inkscape\bin'
$inkCom = Join-Path $bin 'inkscape.com'
$inkExe = Join-Path $bin 'inkscape.exe'
$ink = if (Test-Path -LiteralPath $inkCom) { $inkCom } else { $inkExe }

$golden = 'D:\WORK\TEST\Project 1\samples\golden-logo.jpg'
$outDir = 'D:\WORK\TEST\Project 1\exports\inkscape_smoke'
New-Item -ItemType Directory -Path $outDir -Force | Out-Null
$outPng = Join-Path $outDir 'golden_from_inkscape.png'
$outSvg = Join-Path $outDir 'golden_probe.svg'
Remove-Item -LiteralPath $outPng, $outSvg -Force -ErrorAction SilentlyContinue

Write-Host "using: $ink"
Write-Host '=== version ==='
$ver = & $ink --version 2>&1 | Out-String
Write-Host $ver.Trim()
Write-Host "ver_exit: $LASTEXITCODE"

Write-Host '=== export with -o ==='
& $ink $golden -o $outPng 2>&1 | ForEach-Object { Write-Host $_ }
Write-Host "o_exit: $LASTEXITCODE  exists:" (Test-Path -LiteralPath $outPng)

if (-not (Test-Path -LiteralPath $outPng)) {
    Write-Host '=== export-type fallback ==='
    & $ink --export-type=png --export-filename=$outPng $golden 2>&1 | ForEach-Object { Write-Host $_ }
    Write-Host "type_exit: $LASTEXITCODE  exists:" (Test-Path -LiteralPath $outPng)
}

if (Test-Path -LiteralPath $outPng) {
    Write-Host "png_kb:" ([math]::Round((Get-Item -LiteralPath $outPng).Length / 1KB, 1))
    @"
INKSCAPE_EXE=$ink
VERSION=$($ver.Trim())
TESTED=$(Get-Date -Format o)
"@ | Set-Content -LiteralPath (Join-Path $outDir 'inkscape_path.env') -Encoding UTF8
    Write-Host 'RESULT: OK'
    exit 0
}

Write-Host 'RESULT: FAIL'
exit 1
