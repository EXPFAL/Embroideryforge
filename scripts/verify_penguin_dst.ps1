$ErrorActionPreference = 'Stop'
$root = 'D:\WORK\TEST\Project 1'
$dst = Join-Path $root 'exports\inkscape_smoke\penguin.dst'
$py = Join-Path $root '.venv\Scripts\python.exe'
$outPng = Join-Path $root 'exports\inkscape_smoke\penguin_pyembroidery_preview.png'

if (-not (Test-Path -LiteralPath $dst)) { throw "missing $dst" }
if (-not (Test-Path -LiteralPath $py)) { throw "missing venv python" }

Write-Host "dst_kb:" ([math]::Round((Get-Item -LiteralPath $dst).Length / 1KB, 1))

$code = @'
from pathlib import Path
import json
from pyembroidery import read, write_png

dst = Path(r"D:\WORK\TEST\Project 1\exports\inkscape_smoke\penguin.dst")
png = Path(r"D:\WORK\TEST\Project 1\exports\inkscape_smoke\penguin_pyembroidery_preview.png")
pattern = read(str(dst))
info = {
    "stitches": pattern.count_stitches(),
    "threads": pattern.count_color_changes() + (1 if pattern.count_stitches() else 0),
    "bounds": pattern.bounds(),
    "num_threads_meta": len(pattern.threadlist) if hasattr(pattern, "threadlist") else None,
}
write_png(pattern, str(png))
info["preview"] = str(png)
print(json.dumps(info, ensure_ascii=False, indent=2))
'@

$tmpPy = Join-Path $root 'exports\inkscape_smoke\_verify_dst.py'
Set-Content -LiteralPath $tmpPy -Value $code -Encoding UTF8
& $py $tmpPy
Write-Host "RESULT: OK"
