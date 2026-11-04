$ErrorActionPreference = 'Stop'
$py = 'D:\WORK\TEST\Project 1\.venv\Scripts\python.exe'
$code = @'
import json
from pathlib import Path
from pyembroidery import read

files = {
  "manual_inkstitch": Path(r"D:\WORK\TEST\Project 1\exports\inkscape_smoke\chibi_multicolor.dst"),
  "auto_forge": Path(r"D:\WORK\TEST\Project 1\exports\segment_chibi_v4\auto.dst"),
  "penguin_single": Path(r"D:\WORK\TEST\Project 1\exports\inkscape_smoke\penguin.dst"),
}
out = {}
for k, p in files.items():
    if not p.exists():
        out[k] = {"exists": False}
        continue
    pat = read(str(p))
    out[k] = {
        "exists": True,
        "size_kb": round(p.stat().st_size/1024, 1),
        "stitches": pat.count_stitches(),
        "color_changes": pat.count_color_changes(),
        "bounds": pat.bounds(),
    }
print(json.dumps(out, ensure_ascii=False, indent=2))
Path(r"D:\WORK\TEST\Project 1\exports\segment_chibi_v4\dst_compare.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
)
'@
$tmp = 'D:\WORK\TEST\Project 1\exports\segment_chibi_v4\_cmp.py'
Set-Content -LiteralPath $tmp -Value $code -Encoding UTF8
& $py $tmp
