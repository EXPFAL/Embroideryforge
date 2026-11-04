$ErrorActionPreference = 'Continue'
$inxDir = Join-Path $env:APPDATA 'inkscape\extensions\inkstitch\inx'
Get-ChildItem -LiteralPath $inxDir -Filter '*.inx' |
  Where-Object { $_.Name -match 'zip|export|dst|output|save|visualize|print|simulator|stitch.plan' } |
  ForEach-Object {
    $raw = Get-Content -LiteralPath $_.FullName -Raw -Encoding UTF8
    $name = if ($raw -match '<name>([^<]+)</name>') { $Matches[1] } else { '?' }
    $submenu = ([regex]::Matches($raw, '<submenu name="([^"]+)"')).Value -join ' | '
    Write-Host ("{0}`n  name={1}`n  {2}`n" -f $_.Name, $name, $submenu)
  }

Write-Host '=== all inx with Zip/zip/DST in body ==='
Select-String -Path (Join-Path $inxDir '*.inx') -Pattern 'Zip|zip folder|DST|embroidery' -SimpleMatch:$false |
  Select-Object -First 40 Filename, Line
