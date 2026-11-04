$ErrorActionPreference = 'Stop'
$seven = 'C:\Program Files\7-Zip\7z.exe'
$zip = 'C:\Users\Administrator\Downloads\inkscape-1.4.4_2026-05-05_dcaf3e7-x64.7z'
$tmp = 'D:\WORK\TEST\Project 1\exports\_inkscape_pe_probe'
if (Test-Path -LiteralPath $tmp) { Remove-Item -LiteralPath $tmp -Recurse -Force }
New-Item -ItemType Directory -Path $tmp -Force | Out-Null

& $seven e $zip "-o$tmp" 'inkscape.exe' -r -y | Out-Null
$exe = Join-Path $tmp 'inkscape.exe'
if (-not (Test-Path -LiteralPath $exe)) { throw "extract failed" }

$fs = [System.IO.File]::OpenRead($exe)
$br = New-Object System.IO.BinaryReader($fs)
$fs.Position = 0x3C
$peOffset = $br.ReadUInt32()
$fs.Position = $peOffset
$sig = [System.Text.Encoding]::ASCII.GetString($br.ReadBytes(4))
$machine = $br.ReadUInt16()
$br.Close(); $fs.Close()

$name = switch ($machine) {
    0x014c { 'i386' }
    0x8664 { 'x64/AMD64' }
    0xAA64 { 'ARM64' }
    default { "unknown_$machine" }
}
Write-Host "archive_exe_PE:" $sig ('0x{0:X4}' -f $machine) $name
Write-Host "archive_exe_bytes:" (Get-Item -LiteralPath $exe).Length

# Also check installed again
$inst = 'D:\TOOL\InkScape\bin\inkscape.exe'
$fs2 = [System.IO.File]::OpenRead($inst)
$br2 = New-Object System.IO.BinaryReader($fs2)
$fs2.Position = 0x3C
$pe2 = $br2.ReadUInt32()
$fs2.Position = $pe2
$null = $br2.ReadBytes(4)
$m2 = $br2.ReadUInt16()
$br2.Close(); $fs2.Close()
$n2 = switch ($m2) { 0x8664 { 'x64/AMD64' } 0xAA64 { 'ARM64' } default { "unknown_$m2" } }
Write-Host "installed_exe_PE:" ('0x{0:X4}' -f $m2) $n2
Write-Host "installed_exe_bytes:" (Get-Item -LiteralPath $inst).Length

# Hash compare
$h1 = (Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash
$h2 = (Get-FileHash -LiteralPath $inst -Algorithm SHA256).Hash
Write-Host "same_file:" ($h1 -eq $h2)
Write-Host "DONE"
