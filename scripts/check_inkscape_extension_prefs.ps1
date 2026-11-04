$ErrorActionPreference = 'Continue'
$pref = Join-Path $env:APPDATA 'inkscape\preferences.xml'
Write-Host "pref exists:" (Test-Path -LiteralPath $pref)
if (Test-Path -LiteralPath $pref) {
  Select-String -LiteralPath $pref -Pattern 'extension|org.inkscape.extensions' -SimpleMatch:$false |
    Select-Object -First 40 LineNumber, Line
}

$ext = Join-Path $env:APPDATA 'inkscape\extensions'
Write-Host "user_ext: $ext"
Write-Host "inx_count_nested:" @(Get-ChildItem -LiteralPath (Join-Path $ext 'inkstitch\inx') -Filter '*.inx' -ErrorAction SilentlyContinue).Count
Write-Host "bin_exe:" (Test-Path -LiteralPath (Join-Path $ext 'inkstitch\bin\inkstitch.exe'))

# Probe: does nested inx get seen? Inkscape 1.4 may only load *.inx directly under extensions or one level.
# Official packaged layout from zip: extract creates folder `inkstitch` with `inx` inside — same as ours.
# Some builds ship flat: copy inx to inkstitch/*.inx with command path bin/inkstitch.exe

# Create a test: list first inx command path existence from nested location
$cmd = Join-Path $ext 'inkstitch\bin\inkstitch.exe'
Write-Host "inkstitch_bin_exe_exists:" (Test-Path -LiteralPath $cmd)
if (Test-Path -LiteralPath $cmd) {
  $vi = [System.Diagnostics.FileVersionInfo]::GetVersionInfo($cmd)
  Write-Host "bin_desc:" $vi.FileDescription $vi.FileVersion
}

# Try running inkscape extension list via GUI-less if available
$ink = 'D:\TOOL\Inkscape\bin\inkscape.com'
& $ink --help 2>&1 | Select-String -Pattern 'extension' | Select-Object -First 20

Write-Host 'DONE'
