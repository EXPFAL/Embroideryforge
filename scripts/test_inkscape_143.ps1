$ErrorActionPreference = 'Continue'

$roots = @('D:\TOOL\Inkscape', 'D:\TOOL\InkScape')
$toolRoot = $roots | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $toolRoot) {
    Write-Host 'FAIL: D:\TOOL\Inkscape not found'
    exit 1
}
Write-Host "toolRoot: $toolRoot"

$exes = @(Get-ChildItem -LiteralPath $toolRoot -Recurse -Filter 'inkscape.exe' -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName)
Write-Host "inkscape.exe count:" $exes.Count
$exes | ForEach-Object { Write-Host "  $_" }
if ($exes.Count -eq 0) { exit 1 }
$ink = $exes[0]

function Get-PeMachine([string]$path) {
    $fs = [System.IO.File]::OpenRead($path)
    $br = New-Object System.IO.BinaryReader($fs)
    $fs.Position = 0x3C
    $peOffset = $br.ReadUInt32()
    $fs.Position = $peOffset
    $null = $br.ReadBytes(4)
    $machine = $br.ReadUInt16()
    $br.Close(); $fs.Close()
    switch ($machine) {
        0x014c { 'i386' }
        0x8664 { 'x64' }
        0xAA64 { 'ARM64' }
        default { ('0x{0:X4}' -f $machine) }
    }
}

$osArch = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture
$pe = Get-PeMachine $ink
Write-Host "OSArch: $osArch"
Write-Host "PE: $pe"
Write-Host "exe_mb:" ([math]::Round((Get-Item -LiteralPath $ink).Length / 1MB, 2))

if ($pe -eq 'ARM64' -and "$osArch" -eq 'X64') {
    Write-Host 'FAIL: still ARM64 on x64 OS'
    exit 2
}

Write-Host '=== --version ==='
& $ink --version
Write-Host "version_exit: $LASTEXITCODE"

$golden = 'D:\WORK\TEST\Project 1\samples\golden-logo.jpg'
$outDir = 'D:\WORK\TEST\Project 1\exports\inkscape_smoke'
New-Item -ItemType Directory -Path $outDir -Force | Out-Null
$outPng = Join-Path $outDir 'golden_from_inkscape.png'
if (Test-Path -LiteralPath $outPng) { Remove-Item -LiteralPath $outPng -Force }

Write-Host '=== export png ==='
& $ink $golden --export-type=png --export-filename=$outPng
Write-Host "export_exit: $LASTEXITCODE"
Write-Host "png_exists:" (Test-Path -LiteralPath $outPng)
if (Test-Path -LiteralPath $outPng) {
    Write-Host "png_kb:" ([math]::Round((Get-Item -LiteralPath $outPng).Length / 1KB, 1))
}

if ((Test-Path -LiteralPath $outPng) -and $pe -eq 'x64') {
    Write-Host 'RESULT: OK'
    # write a small pointer file for later scripts
    @"
INKSCAPE_EXE=$ink
VERSION_PE=$pe
TESTED=$(Get-Date -Format o)
"@ | Set-Content -LiteralPath (Join-Path $outDir 'inkscape_path.env') -Encoding UTF8
    exit 0
}

Write-Host 'RESULT: FAIL'
exit 3
