from __future__ import annotations

import argparse
import json
from pathlib import Path

from embroideryforge.autofill import run_auto_dst
from embroideryforge.evaluate import evaluate_export_dir
from embroideryforge.pipeline import run_pipeline
from embroideryforge.plan import PLAN_FILENAME, build_rules_plan, save_plan
from embroideryforge.segment import run_segment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="embroideryforge", description="EmbroideryForge CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_an = sub.add_parser("analyze", help="分析图片并导出刺绣文件")
    p_an.add_argument("image", type=Path)
    p_an.add_argument("--out", type=Path, required=True)
    p_an.add_argument("--scale", type=float, default=0.4)

    p_seg = sub.add_parser("segment", help="自动分色分区并导出分层 SVG")
    p_seg.add_argument("image", type=Path)
    p_seg.add_argument("--out", type=Path, required=True)
    p_seg.add_argument("--mode", choices=["palette", "kmeans"], default="palette")
    p_seg.add_argument("--k", type=int, default=10)
    p_seg.add_argument("--min-area-ratio", type=float, default=0.0012)
    p_seg.add_argument("--merge-dist", type=float, default=30.0)
    p_seg.add_argument("--upscale", type=int, default=2)

    p_plan = sub.add_parser("plan", help="从 layers.json 生成 stitch_plan.json")
    p_plan.add_argument("layers_json", type=Path)
    p_plan.add_argument("--out", type=Path, default=None)
    p_plan.add_argument("--height-mm", type=float, default=70.0)
    p_plan.add_argument("--quality", choices=["draft", "standard", "dense"], default="dense")
    p_plan.add_argument("--llm", action="store_true")
    p_plan.add_argument("--source", default=None)

    p_auto = sub.add_parser("stitch-auto", help="自动分区并生成多色 DST")
    p_auto.add_argument("image", type=Path)
    p_auto.add_argument("--out", type=Path, required=True)
    p_auto.add_argument("--mode", choices=["palette", "kmeans"], default="palette")
    p_auto.add_argument("--k", type=int, default=12)
    p_auto.add_argument("--upscale", type=int, default=2)
    p_auto.add_argument("--height-mm", type=float, default=70.0)
    p_auto.add_argument("--plan", type=Path, default=None)
    p_auto.add_argument("--llm", action="store_true")
    p_auto.add_argument(
        "--bg",
        choices=["auto", "off", "grabcut", "white"],
        default="auto",
        help="去背景：auto 白底跳过，复杂背景 GrabCut",
    )
    p_auto.add_argument(
        "--quality",
        choices=["draft", "standard", "dense"],
        default="dense",
        help="针密度档位，dense 逼近手工对照",
    )

    p_eval = sub.add_parser("evaluate", help="对导出目录打自动评估分（含课题五维）")
    p_eval.add_argument("out_dir", type=Path)
    p_eval.add_argument("--image", type=Path, default=None)
    p_eval.add_argument("--ref-dst", type=Path, default=None)

    args = parser.parse_args(argv)
    if args.cmd == "analyze":
        result = run_pipeline(args.image, args.out, scale=args.scale)
        print(json.dumps({"ok": result["ok"], "out": str(args.out)}, ensure_ascii=False))
        return 0 if result["ok"] else 1
    if args.cmd == "segment":
        result = run_segment(
            args.image,
            args.out,
            mode=args.mode,
            k=args.k,
            min_area_ratio=args.min_area_ratio,
            merge_dist=args.merge_dist,
            upscale=args.upscale,
        )
        print(
            json.dumps(
                {
                    "ok": result["ok"],
                    "out": str(result["out_dir"]),
                    "svg": str(result["svg"]),
                    "layers": [x["tag"] for x in result["report"]["layers"]],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if result["ok"] else 1
    if args.cmd == "plan":
        payload = json.loads(args.layers_json.read_text(encoding="utf-8"))
        layers = payload.get("layers")
        if not isinstance(layers, list):
            print(json.dumps({"ok": False, "error": "layers.json missing layers[]"}, ensure_ascii=False))
            return 1
        if args.llm:
            from embroideryforge.llm import build_llm_plan

            plan = build_llm_plan(
                layers,
                target_height_mm=args.height_mm,
                fallback=True,
                quality=args.quality,
            )
        else:
            plan = build_rules_plan(
                layers,
                target_height_mm=args.height_mm,
                source=args.source or "rules",
                quality=args.quality,
            )
        out = args.out or (args.layers_json.parent / PLAN_FILENAME)
        save_plan(plan, out)
        print(
            json.dumps(
                {
                    "ok": True,
                    "plan": str(out),
                    "source": plan["source"],
                    "quality": plan.get("quality"),
                    "layers": [L["tag"] for L in plan["layers"]],
                    "llm_meta": plan.get("llm_meta"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.cmd == "stitch-auto":
        result = run_auto_dst(
            args.image,
            args.out,
            mode=args.mode,
            k=args.k,
            upscale=args.upscale,
            target_height_mm=args.height_mm,
            plan_path=args.plan,
            use_llm=bool(args.llm) and args.plan is None,
            bg_mode=args.bg,
            quality=args.quality,
        )
        print(
            json.dumps(
                {
                    "ok": result["ok"],
                    "dst": str(result["dst"]),
                    "plan": str(result["plan"]),
                    "plan_source": result["report"].get("plan_source"),
                    "quality": result["report"].get("quality"),
                    "bg_applied": (result["report"].get("bg") or {}).get("applied"),
                    "stitches": result["report"]["stitches"],
                    "color_changes": result["report"]["color_changes"],
                    "layers": [x["tag"] for x in result["report"]["layers"]],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if result["ok"] else 1
    if args.cmd == "evaluate":
        result = evaluate_export_dir(
            args.out_dir,
            source_image=args.image,
            reference_dst=args.ref_dst,
        )
        print(
            json.dumps(
                {
                    "ok": result.get("rubric_pass_95", False) or result["rubric_overall"] >= 0.95,
                    "rubric_pass_95": result["rubric_pass_95"],
                    "rubric_overall": result["rubric_overall"],
                    "rubric": result["rubric"],
                    "overall": result["overall"],
                    "scores": result["scores"],
                    "report": result["report"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if result["rubric_overall"] >= 0.95 else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
