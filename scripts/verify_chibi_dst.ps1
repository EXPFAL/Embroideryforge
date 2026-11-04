$ErrorActionPreference = 'Stop'
$py = 'D:\WORK\TEST\Project 1\.venv\Scripts\python.exe'
$dst = 'D:\WORK\TEST\Project 1\exports\inkscape_smoke\chibi_multicolor.dst'
$png = 'D:\WORK\TEST\Project 1\exports\inkscape_smoke\chibi_multicolor_preview.png'
$code = @"
from pathlib import Path
import json
from pyembroidery import read, write_png
dst = Path(r'$dst')
png = Path(r'$png')
p = read(str(dst))
info = {
  'file': dst.name,
  'size_kb': round(dst.stat().st_size/1024, 1),
  'stitches': p.count_stitches(),
  'color_changes': p.count_color_changes(),
  'bounds': p.bounds(),
}
write_png(p, str(png))
info['preview'] = png.name
print(json.dumps(info, ensure_ascii=False, indent=2))
"@
$tmp = 'D:\WORK\TEST\Project 1\exports\inkscape_smoke\_verify_chibi.py'
Set-Content -LiteralPath $tmp -Value $code -Encoding UTF8
& $py $tmp
