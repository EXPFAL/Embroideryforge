from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2


def write_run_artifacts(
    out_dir: Path,
    source_name: str,
    analysis: dict[str, Any],
    stitch_plan: list[dict[str, Any]],
    export_meta: dict[str, Any],
    overlay_bgr,
    swatch_bgr,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = out_dir / "contour_overlay.png"
    swatch_path = out_dir / "color_swatch.png"
    cv2.imwrite(str(overlay_path), overlay_bgr)
    cv2.imwrite(str(swatch_path), swatch_bgr)

    report = {
        "product": "EmbroideryForge",
        "version": "0.1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": source_name,
        "analysis": analysis,
        "stitch_plan": stitch_plan,
        "export": {
            "dst": str(export_meta["dst"].name),
            "png": str(export_meta["png"].name),
            "stitches": export_meta["stitches"],
        },
        "roles": {
            "model_or_rules": "主色/轮廓分析 + 确定性针法建议表（可替换为 LLM）",
            "tools": "OpenCV, pyembroidery",
            "human": "选择输入、确认分区取舍、验收预览",
        },
    }
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md = out_dir / "design_notes.md"
    lines = [
        f"# 设计说明 — {source_name}",
        "",
        f"- 时间（UTC）：{report['created_at']}",
        f"- 尺寸：{analysis['width']}×{analysis['height']}",
        f"- 外轮廓数：{analysis['external_contours']}",
        f"- 针数：{export_meta['stitches']}",
        "",
        "## 主色",
    ]
    for c in analysis["colors"]:
        lines.append(f"- {c['hex']}（占比 {c['ratio']}）")
    lines.extend(["", "## 针法建议", ""])
    for p in stitch_plan:
        lines.append(
            f"- `{p['region_id']}` {p.get('hex')} → {p['suggested_stitch']}（{p['role']}）"
        )
    lines.extend(
        [
            "",
            "## 取舍（草案）",
            "1. 当前导出以主轮廓走针为主，填针分区将在后续迭代按主色 mask 实现。",
            "2. 白底被抑制，避免背景占色干扰分区。",
            "",
        ]
    )
    md.write_text("\n".join(lines), encoding="utf-8")
    return {
        "overlay": overlay_path,
        "swatch": swatch_path,
        "report": report_path,
        "notes": md,
    }
