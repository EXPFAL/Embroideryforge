from __future__ import annotations

from pathlib import Path
from typing import Any

from pyembroidery import EmbPattern, write_dst, write_png


def outline_to_pattern(
    points: list[list[int]],
    scale: float = 0.4,
    color: str = "black",
) -> EmbPattern:
    pattern = EmbPattern()
    if len(points) < 2:
        pattern.add_block([(0, 0), (100, 0), (100, 100), (0, 100), (0, 0)], color)
        return pattern
    block = [(int(x * scale), int(-y * scale)) for x, y in points]
    if block[0] != block[-1]:
        block.append(block[0])
    pattern.add_block(block, color)
    return pattern


def export_pattern(pattern: EmbPattern, out_dir: str | Path, stem: str = "design") -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{stem}.dst"
    png = out_dir / f"{stem}.png"
    write_dst(pattern, str(dst))
    write_png(pattern, str(png))
    return {"dst": dst, "png": png, "stitches": pattern.count_stitches()}
