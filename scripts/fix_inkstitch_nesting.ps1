$ErrorActionPreference = 'Stop'
$extDir = Join-Path $env:APPDATA 'inkscape\extensions'
$outer = Join-Path $extDir 'inkstitch'
$inner = Join-Path $outer 'inkstitch'

if (-not (Test-Path -LiteralPath $inner)) {
    Write-Host 'No double nesting; layout may already be OK'
    if (Test-Path -LiteralPath (Join-Path $outer 'inx')) {
        Write-Host 'OK: inx under extensions\inkstitch\inx'
        exit 0
    }
    throw "Unexpected layout under $outer"
}

$tmp = Join-Path $extDir '_inkstitch_flat_tmp'
if (Test-Path -LiteralPath $tmp) { Remove-Item -LiteralPath $tmp -Recurse -Force }

Write-Host "Moving inner package out..."
Move-Item -LiteralPath $inner -Destination $tmp

# Keep uninstaller if present at outer, then replace outer with flat package
Get-ChildItem -LiteralPath $outer -Force | ForEach-Object {
    Write-Host "removing leftover:" $_.Name
    Remove-Item -LiteralPath $_.FullName -Recurse -Force
}
Remove-Item -LiteralPath $outer -Force
Move-Item -LiteralPath $tmp -Destination $outer

Write-Host '=== fixed top of inkstitch ==='
Get-ChildItem -LiteralPath $outer | Select-Object Name, Mode | Format-Table -AutoSize
Write-Host 'params exists:' (Test-Path -LiteralPath (Join-Path $outer 'inx\inkstitch_params.inx'))
Write-Host 'VERSION:' (Get-Content -LiteralPath (Join-Path $outer 'VERSION') -ErrorAction SilentlyContinue)
Write-Host 'DONE'
