from __future__ import annotations

import argparse
import json
from pathlib import Path

from embroideryforge.autofill import run_auto_dst
from embroideryforge.evaluate import evaluate_export_dir
from embroideryforge.pipeline import run_pipeline
from embroideryforge.plan import PLAN_FILENAME, build_rules_plan, save_plan
from embroideryforge.segment import run_segment
from embroideryforge.vectorize import vectorize_image


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
    p_seg.add_argument("--domain", choices=["auto", "chibi", "portrait"], default="auto")

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
        choices=["auto", "off", "grabcut", "white", "keep", "scene"],
        default="auto",
        help="去背景：auto/rembg；keep|scene=保留全帧（与 stitch-photo 一致，插画少用）",
    )
    p_auto.add_argument(
        "--quality",
        choices=["draft", "standard", "dense"],
        default="dense",
        help="针密度档位，dense 逼近手工对照",
    )
    p_auto.add_argument(
        "--domain",
        choices=["auto", "chibi", "portrait"],
        default="auto",
        help="调色板域：插画 chibi / 人像 portrait / 自动",
    )
    efr_g = p_auto.add_mutually_exclusive_group()
    efr_g.add_argument("--efr", action="store_true", help="强制启用照片 EFR")
    efr_g.add_argument("--no-efr", action="store_true", help="强制关闭 EFR")
    p_auto.add_argument("--efr-colors", type=int, default=7, help="EFR 目标色数")

    p_eval = sub.add_parser("evaluate", help="对导出目录打自动评估分（含课题五维）")
    p_eval.add_argument("out_dir", type=Path)
    p_eval.add_argument("--image", type=Path, default=None)
    p_eval.add_argument("--ref-dst", type=Path, default=None)

    p_cmp = sub.add_parser("compare-photo", help="对照 naive vs EFR 两个导出目录")
    p_cmp.add_argument("naive_dir", type=Path)
    p_cmp.add_argument("efr_dir", type=Path)
    p_cmp.add_argument("--out", type=Path, default=None)

    p_j = sub.add_parser(
        "judge-photo",
        help="DeepSeek 识图裁判：原图+预览 → vision_judge.json（辅评，不改 dens）",
    )
    p_j.add_argument("export_dir", type=Path)
    p_j.add_argument("--image", type=Path, default=None, help="原图；缺省尝试从导出推断")
    p_j.add_argument(
        "--rubric",
        choices=["auto", "portrait", "animal"],
        default="auto",
        help="评分表；auto 读 meta.domain",
    )
    p_j.add_argument("--no-compare", action="store_true", help="不附带 compare_preview")
    p_j.add_argument("--max-side", type=int, default=1280, help="送模前缩边")

    p_jp = sub.add_parser(
        "judge-photo-pair",
        help="DeepSeek 识图 A/B 对照 → vision_judge_pair.json",
    )
    p_jp.add_argument("dir_a", type=Path)
    p_jp.add_argument("dir_b", type=Path)
    p_jp.add_argument("--image", type=Path, default=None)
    p_jp.add_argument(
        "--rubric",
        choices=["auto", "portrait", "animal"],
        default="auto",
    )
    p_jp.add_argument("--max-side", type=int, default=1280)

    p_ps = sub.add_parser("stitch-photo", help="照片绣 Photo-Stitch（色调密度短填，不替代 stitch-auto）")
    p_ps.add_argument("image", type=Path)
    p_ps.add_argument("--out", type=Path, required=True)
    p_ps.add_argument("--colors", type=int, default=9, help="针位/色数，默认 9（对齐常见照片绣）")
    p_ps.add_argument("--resolution", choices=["low", "mid", "high"], default="high")
    p_ps.add_argument(
        "--mode",
        choices=["face", "hybrid", "full"],
        default="face",
        help="face=脸ROI；hybrid=脸照片绣+衣帽tatami；full=整图",
    )
    p_ps.add_argument("--height-mm", type=float, default=70.0)
    p_ps.add_argument(
        "--bg",
        choices=["auto", "off", "grabcut", "white", "keep", "scene"],
        default="auto",
        help="keep/scene=保留全帧背景，主体加密、场景疏针（V6）",
    )
    p_ps.add_argument(
        "--full-frame",
        action="store_true",
        help="同 --mode full",
    )
    p_ps.add_argument("--no-face-roi", action="store_true", help="禁用脸部 ROI（改为整图）")
    p_ps.add_argument(
        "--poster",
        action="store_true",
        help="软实填档（遗留 v3）+ 假厚预览",
    )
    p_ps.add_argument(
        "--density-field",
        action="store_true",
        help="密度场短针；近脸默认 v4.1.1，半身自动 V6 unified",
    )
    p_ps.add_argument(
        "--density-unified",
        action="store_true",
        help="强制 V6 整主体统一彩色密度（无衣 tatami/叠针）",
    )
    p_ps.add_argument(
        "--no-density-unified",
        action="store_true",
        help="强制 v4.1.1（禁止半身自动切 V6）",
    )
    p_ps.add_argument(
        "--domain",
        choices=["auto", "figure", "animal"],
        default="auto",
        help="animal=跳过 dens_face（宠物/多宠）；figure=保留人脸密轨；auto=有 MediaPipe 人脸才开",
    )
    p_ps.add_argument(
        "--portrait-p0",
        action="store_true",
        help="人像 P0：Reserve 肤唇眼 + 灰垫底序（默认关；不做全图线表贴色，避免蒙红）",
    )
    p_ps.add_argument(
        "--no-portrait-p0",
        action="store_true",
        help="显式关闭人像 P0（与默认相同）",
    )
    p_ps.add_argument(
        "--streamlines-patch",
        nargs="?",
        const="AUTO",
        default=None,
        metavar="MASK",
        help="A2 流线单连通域 spike；MASK=罩图路径，省略则用主体最大连通域（不改 dens）",
    )
    p_ps.add_argument(
        "--stitches-per-mm2",
        type=float,
        default=None,
        help="人像密度预算（针/mm²）；映射 step，不换填法",
    )
    p_ps.add_argument(
        "--density-contrast",
        type=float,
        default=None,
        help="密度对比度 γ（默认沿用 dens gamma；>1 更拉开疏密）",
    )
    p_ps.add_argument(
        "--stitch-mask",
        type=Path,
        default=None,
        help="只绣这张罩（与输入同尺寸）。方案 B 单动物构图，不改 dens",
    )
    p_ps.add_argument(
        "--photo-blend",
        action="store_true",
        help="商用叠色：中间调再绣次近线色，密度约主色四成",
    )

    p_vec = sub.add_parser("vectorize", help="VTracer 位图→SVG 旁路（不替代 stitch-auto）")
    p_vec.add_argument("image", type=Path)
    p_vec.add_argument("--out", type=Path, required=True)

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
            domain=args.domain,
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
        use_efr = True if args.efr else (False if args.no_efr else None)
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
            domain=args.domain,
            use_efr=use_efr,
            efr_colors=args.efr_colors,
        )
        print(
            json.dumps(
                {
                    "ok": result["ok"],
                    "dst": str(result["dst"]),
                    "plan": str(result["plan"]),
                    "plan_source": result["report"].get("plan_source"),
                    "quality": result["report"].get("quality"),
                    "domain": result["report"].get("domain"),
                    "efr": result["report"].get("efr"),
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
    if args.cmd == "compare-photo":
        from embroideryforge.photo_eval import compare_photo_exports

        report = compare_photo_exports(args.naive_dir, args.efr_dir, out_json=args.out)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report.get("ok") else 1
    if args.cmd == "judge-photo":
        from embroideryforge.vision_judge import judge_photo_export

        report = judge_photo_export(
            args.export_dir,
            image=args.image,
            rubric=args.rubric,
            include_compare=not args.no_compare,
            max_side=int(args.max_side),
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report.get("ok") else 1
    if args.cmd == "judge-photo-pair":
        from embroideryforge.vision_judge import judge_photo_pair

        report = judge_photo_pair(
            args.dir_a,
            args.dir_b,
            image=args.image,
            rubric=args.rubric,
            max_side=int(args.max_side),
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report.get("ok") else 1
    if args.cmd == "stitch-photo":
        from embroideryforge.photo_stitch import run_photo_stitch

        mode = args.mode
        if args.full_frame or args.no_face_roi:
            mode = "full"
        stream_arg = getattr(args, "streamlines_patch", None)
        if stream_arg is not None:
            from embroideryforge.streamlines_patch import run_streamlines_patch

            mask_p = None if stream_arg == "AUTO" else Path(stream_arg)
            result = run_streamlines_patch(
                args.image,
                args.out,
                mask_path=mask_p,
                n_colors=args.colors,
                resolution=args.resolution,
                target_height_mm=args.height_mm,
                bg_mode=args.bg if args.bg != "auto" else "keep",
            )
        elif args.density_field or args.density_unified:
            from embroideryforge.density_field import run_density_photo

            if args.density_unified:
                unified: bool | None = True
            elif args.no_density_unified:
                unified = False
            else:
                unified = None  # 半身自动 V6
            if getattr(args, "no_portrait_p0", False):
                p0_flag: bool | None = False
            elif getattr(args, "portrait_p0", False):
                p0_flag = True
            else:
                p0_flag = None
            result = run_density_photo(
                args.image,
                args.out,
                n_colors=args.colors,
                resolution=args.resolution,
                target_height_mm=args.height_mm,
                bg_mode=args.bg,
                hybrid_body=mode != "face",
                unified=unified,
                domain=str(args.domain),
                portrait_p0=p0_flag,
                stitches_per_mm2=getattr(args, "stitches_per_mm2", None),
                density_contrast=getattr(args, "density_contrast", None),
                stitch_mask=getattr(args, "stitch_mask", None),
                photo_blend=bool(getattr(args, "photo_blend", False)),
            )
        else:
            result = run_photo_stitch(
                args.image,
                args.out,
                n_colors=args.colors,
                resolution=args.resolution,
                target_height_mm=args.height_mm,
                bg_mode=args.bg,
                face_roi=mode == "face",
                full_frame=mode == "full",
                mode=mode,
                poster=bool(args.poster),
            )
        print(
            json.dumps(
                {
                    "ok": result["ok"],
                    "dst": str(result["dst"]),
                    "engine": result["meta"].get("engine"),
                    "mode": result["meta"].get("mode"),
                    "poster": result["meta"].get("poster"),
                    "density_field": "density" in str(result["meta"].get("engine") or ""),
                    "resolution": result["meta"].get("resolution"),
                    "n_colors": result["meta"].get("n_colors"),
                    "roi": result["meta"].get("roi"),
                    "stitches": result["report"]["stitches"],
                    "color_changes": result["report"]["color_changes"],
                    "palette": result["meta"].get("palette"),
                    "layers": [L.get("tag") for L in result["meta"].get("layers") or []],
                    "compare": str(Path(result["out_dir"]) / "compare_preview.png"),
                    "preview": result["meta"].get("preview"),
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
    if args.cmd == "vectorize":
        meta = vectorize_image(args.image, args.out)
        print(json.dumps(meta, ensure_ascii=False, indent=2))
        return 0 if meta.get("ok") else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
