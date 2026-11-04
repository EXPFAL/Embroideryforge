$ErrorActionPreference = 'Continue'
function Get-PeMachine([string]$path) {
    $fs = [System.IO.File]::OpenRead($path)
    $br = New-Object System.IO.BinaryReader($fs)
    $fs.Position = 0x3C
    $peOffset = $br.ReadUInt32()
    $fs.Position = $peOffset
    $null = $br.ReadBytes(4)
    $machine = $br.ReadUInt16()
    $br.Close(); $fs.Close()
    $name = switch ($machine) {
        0x014c { 'i386' }
        0x8664 { 'x64' }
        0xAA64 { 'ARM64' }
        default { "0x{0:X4}" -f $machine }
    }
    [pscustomobject]@{ Name = Split-Path $path -Leaf; Machine = $name; MB = [math]::Round((Get-Item $path).Length/1MB,2) }
}

$bin = 'D:\TOOL\InkScape\bin'
Get-ChildItem -LiteralPath $bin -Include *.exe,*.dll -File -ErrorAction SilentlyContinue |
    Select-Object -First 30 |
    ForEach-Object { Get-PeMachine $_.FullName } |
    Format-Table -AutoSize

# Prefer largest dlls
Get-ChildItem -LiteralPath $bin -Filter *.dll |
    Sort-Object Length -Descending |
    Select-Object -First 8 |
    ForEach-Object { Get-PeMachine $_.FullName } |
    Format-Table -AutoSize
