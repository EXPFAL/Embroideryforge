"""从 DST 渲染清洁预览；颜色优先读同目录 report/plan（DST 无 RGB）。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from embroideryforge.preview import render_clean_preview


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dst", type=Path)
    parser.add_argument("out", type=Path, nargs="?", default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--plan", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or args.dst.with_name("auto_clean_preview.png")
    info = render_clean_preview(
        args.dst,
        out,
        report_path=args.report,
        plan_path=args.plan,
    )
    print(json.dumps(info, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
