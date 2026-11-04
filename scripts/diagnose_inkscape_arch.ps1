$ErrorActionPreference = 'Continue'

Write-Host "OS:" ([System.Environment]::OSVersion.VersionString)
Write-Host "ProcessArch:" ([System.Runtime.InteropServices.RuntimeInformation]::ProcessArchitecture)
Write-Host "OSArch:" ([System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture)
Write-Host "Is64BitOS:" [Environment]::Is64BitOperatingSystem

$exe = 'D:\TOOL\InkScape\bin\inkscape.exe'
$fs = [System.IO.File]::OpenRead($exe)
$br = New-Object System.IO.BinaryReader($fs)
$null = $br.ReadBytes(0x3C)
$fs.Position = 0x3C
$peOffset = $br.ReadUInt32()
$fs.Position = $peOffset
$peSig = [System.Text.Encoding]::ASCII.GetString($br.ReadBytes(4))
$machine = $br.ReadUInt16()
$br.Close(); $fs.Close()

$machineName = switch ($machine) {
    0x014c { 'i386' }
    0x8664 { 'x64/AMD64' }
    0xAA64 { 'ARM64' }
    0x01c4 { 'ARM' }
    default { "unknown_$machine" }
}
Write-Host "PE_signature:" $peSig
Write-Host "PE_machine:" ('0x{0:X4}' -f $machine) $machineName
Write-Host "exe_size_mb:" ([math]::Round((Get-Item -LiteralPath $exe).Length / 1MB, 2))

# Check a few sibling DLLs exist
$bin = 'D:\TOOL\InkScape\bin'
@('libinkscape_base.dll','libstdc++-6.dll','libcairo-2.dll') | ForEach-Object {
    $p = Join-Path $bin $_
    Write-Host ("dll {0}: {1}" -f $_, (Test-Path -LiteralPath $p))
}

# 7z listing if 7z available
$zip = 'C:\Users\Administrator\Downloads\inkscape-1.4.4_2026-05-05_dcaf3e7-x64.7z'
$seven = @(
    'C:\Program Files\7-Zip\7z.exe',
    'C:\Program Files (x86)\7-Zip\7z.exe',
    'D:\TOOL\7-Zip\7z.exe'
) | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1

if ($seven) {
    Write-Host "7z:" $seven
    & $seven l $zip | Select-Object -First 25
} else {
    Write-Host "7z.exe not found; skip archive listing"
}

Write-Host "DONE"
