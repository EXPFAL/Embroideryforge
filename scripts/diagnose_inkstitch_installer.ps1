$ErrorActionPreference = 'Continue'
$path = 'C:\Users\Administrator\Downloads\inkstitch-v3.3.0-windows-64bit.exe'
Write-Host "exists:" (Test-Path -LiteralPath $path)
if (-not (Test-Path -LiteralPath $path)) { exit 1 }
$i = Get-Item -LiteralPath $path
Write-Host "size_mb:" ([math]::Round($i.Length/1MB,2))
try {
  $vi = [System.Diagnostics.FileVersionInfo]::GetVersionInfo($path)
  Write-Host "desc:" $vi.FileDescription
  Write-Host "product:" $vi.ProductName
  Write-Host "company:" $vi.CompanyName
  Write-Host "original:" $vi.OriginalFilename
} catch {}
try {
  $sig = Get-AuthenticodeSignature -LiteralPath $path
  Write-Host "sig:" $sig.Status
} catch {}

# Portable Inkscape layout
$inkRoot = 'D:\TOOL\Inkscape'
Write-Host "inkscape_root:" (Test-Path -LiteralPath $inkRoot)
$candidates = @(
  (Join-Path $inkRoot 'share\inkscape\extensions'),
  (Join-Path $inkRoot 'share\extensions'),
  (Join-Path $env:APPDATA 'inkscape\extensions'),
  (Join-Path $env:APPDATA 'Inkscape\extensions')
)
foreach ($c in $candidates) {
  Write-Host ("ext_dir {0} exists={1}" -f $c, (Test-Path -LiteralPath $c))
}

# Common Program Files inkscape
@(
  'C:\Program Files\Inkscape\share\inkscape\extensions',
  'C:\Program Files\Inkscape\share\extensions'
) | ForEach-Object { Write-Host ("pf {0} exists={1}" -f $_, (Test-Path -LiteralPath $_)) }

# Is Inno Setup? Check magic
$bytes = [System.IO.File]::ReadAllBytes($path)[0..1]
Write-Host ("mz:" + [char]$bytes[0] + [char]$bytes[1])
Write-Host 'DONE'
