$ErrorActionPreference = 'Continue'
$installer = 'C:\Users\Administrator\Downloads\inkstitch-v3.3.0-windows-64bit.exe'
$extDir = Join-Path $env:APPDATA 'inkscape\extensions'
$log = 'D:\WORK\TEST\Project 1\exports\inkstitch_setup.log'

# Ensure user extensions exists and Inkscape has been "seen"
New-Item -ItemType Directory -Path $extDir -Force | Out-Null

# Also create a junction/symlink hint? No — instead create Program Files junction if admin
# Prefer silent DIR into user extensions
Write-Host "extDir: $extDir"
Write-Host "Running silent installer..."

$args = @(
  '/VERYSILENT',
  '/SUPPRESSMSGBOXES',
  '/NORESTART',
  "/DIR=$extDir",
  "/LOG=$log"
)
$p = Start-Process -FilePath $installer -ArgumentList $args -Wait -PassThru
Write-Host "exit:" $p.ExitCode

if (Test-Path -LiteralPath $log) {
    Write-Host '=== log tail ==='
    Get-Content -LiteralPath $log -Tail 40
}

$inx = @(Get-ChildItem -LiteralPath $extDir -Filter '*inkstitch*.inx' -Recurse -ErrorAction SilentlyContinue)
$dir = @(Get-ChildItem -LiteralPath $extDir -Directory -ErrorAction SilentlyContinue | Where-Object { $_.Name -match 'inkstitch' })
Write-Host "inx_count:" $inx.Count
Write-Host "dir_count:" $dir.Count
$inx | Select-Object -First 10 | ForEach-Object { Write-Host $_.FullName }
$dir | ForEach-Object { Write-Host $_.FullName }
