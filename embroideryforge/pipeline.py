from __future__ import annotations

from pathlib import Path
from typing import Any

from embroideryforge.analyze import analyze_image, suggest_stitch_plan
from embroideryforge.report import write_run_artifacts
from embroideryforge.stitch import export_pattern, outline_to_pattern


def run_pipeline(image_path: str | Path, out_dir: str | Path, scale: float = 0.4) -> dict[str, Any]:
    image_path = Path(image_path)
    out_dir = Path(out_dir)
    result = analyze_image(image_path)
    plan = suggest_stitch_plan(result)
    pattern = outline_to_pattern(result.outline_points, scale=scale)
    export_meta = export_pattern(pattern, out_dir, stem="embroidery")
    artifacts = write_run_artifacts(
        out_dir=out_dir,
        source_name=image_path.name,
        analysis=result.to_dict(),
        stitch_plan=plan,
        export_meta=export_meta,
        overlay_bgr=result.overlay_bgr,
        swatch_bgr=result.swatch_bgr,
    )
    return {
        "analysis": result.to_dict(),
        "stitch_plan": plan,
        "export": export_meta,
        "artifacts": artifacts,
        "ok": export_meta["stitches"] > 0,
    }
