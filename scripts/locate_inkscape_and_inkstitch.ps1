$ErrorActionPreference = 'Continue'

Write-Host '=== inkscape.exe search (common roots) ==='
$roots = @(
  'D:\TOOL\Inkscape',
  'D:\TOOL\InkScape',
  'C:\Program Files\Inkscape',
  'C:\Program Files (x86)\Inkscape',
  "$env:LOCALAPPDATA\Programs\Inkscape"
)
foreach ($r in $roots) {
  $exe = Join-Path $r 'bin\inkscape.exe'
  $com = Join-Path $r 'bin\inkscape.com'
  Write-Host ("root {0} exists={1} exe={2} com={3}" -f $r, (Test-Path -LiteralPath $r), (Test-Path -LiteralPath $exe), (Test-Path -LiteralPath $com))
  if (Test-Path -LiteralPath $exe) {
    try {
      $vi = [System.Diagnostics.FileVersionInfo]::GetVersionInfo($exe)
      Write-Host "  product:" $vi.ProductVersion " file:" $vi.FileVersion
    } catch {}
  }
}

Write-Host '=== where.exe / Get-Command ==='
Get-Command inkscape -ErrorAction SilentlyContinue | Format-List Source, Version
cmd /c "where inkscape 2>nul"

Write-Host '=== extension dirs ==='
$exts = @(
  "$env:APPDATA\inkscape\extensions",
  'C:\Program Files\Inkscape\share\inkscape\extensions',
  'D:\TOOL\Inkscape\share\inkscape\extensions'
)
foreach ($e in $exts) {
  $exists = Test-Path -LiteralPath $e
  Write-Host ("ext {0} exists={1}" -f $e, $exists)
  if ($exists) {
    $inkstitch = Get-ChildItem -LiteralPath $e -Directory -ErrorAction SilentlyContinue |
      Where-Object { $_.Name -match 'inkstitch|ink.?stitch' }
    $inx = Get-ChildItem -LiteralPath $e -Filter '*inkstitch*.inx' -ErrorAction SilentlyContinue
    Write-Host "  inkstitch_dirs:" @($inkstitch).Count " inx:" @($inx).Count
    if ($inkstitch) { $inkstitch | ForEach-Object { Write-Host "   " $_.FullName } }
  }
}

Write-Host '=== installer files ==='
@(
  'C:\Users\Administrator\Downloads\inkscape-1.4.3.exe',
  'C:\Users\Administrator\Downloads\inkstitch-v3.3.0-windows-64bit.exe'
) | ForEach-Object {
  if (Test-Path -LiteralPath $_) {
    $i = Get-Item -LiteralPath $_
    Write-Host ("{0} mb={1}" -f $i.Name, [math]::Round($i.Length/1MB,2))
  } else { Write-Host "missing $_" }
}

# Uninstall registry clues
Write-Host '=== uninstall registry Inkscape ==='
$keys = @(
  'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\*',
  'HKLM:\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\*',
  'HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\*'
)
Get-ItemProperty $keys -ErrorAction SilentlyContinue |
  Where-Object { $_.DisplayName -match 'Inkscape' } |
  Select-Object DisplayName, DisplayVersion, InstallLocation, UninstallString |
  Format-List

Write-Host 'DONE'
