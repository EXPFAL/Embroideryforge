$dir = 'D:\WORK\TEST\Project 1\exports\inkscape_smoke'
Get-ChildItem -LiteralPath $dir | ForEach-Object {
  Write-Host ("{0}`t{1}`t{2}" -f $_.LastWriteTime.ToString('s'), $_.Length, $_.Name)
}
