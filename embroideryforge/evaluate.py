"""自动评价指标 + 课题五维评分（目标各项 ≥0.95）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pyembroidery import read_dst

from embroideryforge.plan import PLAN_FILENAME, load_plan, validate_plan

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DESIGN_NOTES = PROJECT_ROOT / "docs" / "课题一设计说明-chibi基线.md"


def _resize_match(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ha, wa = a.shape[:2]
    hb, wb = b.shape[:2]
    h, w = min(ha, hb), min(wa, wb)
    if (ha, wa) != (h, w):
        a = cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)
    if (hb, wb) != (h, w):
        b = cv2.resize(b, (w, h), interpolation=cv2.INTER_AREA)
    return a, b


def _subject_mask_from_image(bgr: np.ndarray, mask_path: Path | None = None) -> np.ndarray:
    if mask_path is not None and mask_path.is_file():
        m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if m is not None:
            if m.shape[:2] != bgr.shape[:2]:
                m = cv2.resize(m, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
            return m > 0
    return np.any(bgr < 245, axis=2)


def _prepare_fg_mask(
    a: np.ndarray,
    subject_mask: np.ndarray | None,
) -> np.ndarray:
    if subject_mask is not None:
        sm = subject_mask
        if sm.shape[:2] != a.shape[:2]:
            sm = cv2.resize(
                sm.astype(np.uint8), (a.shape[1], a.shape[0]), interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        mask = sm & np.any(a < 250, axis=2)
    else:
        mask = np.any(a < 245, axis=2)
    mask_u8 = cv2.erode(mask.astype(np.uint8) * 255, np.ones((3, 3), np.uint8), iterations=1)
    return mask_u8 > 0


def _hsv_hist_corr(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0
    m = mask.astype(np.uint8) * 255
    ha = cv2.calcHist([cv2.cvtColor(a, cv2.COLOR_BGR2HSV)], [0], m, [18], [0, 180])
    hb = cv2.calcHist([cv2.cvtColor(b, cv2.COLOR_BGR2HSV)], [0], m, [18], [0, 180])
    cv2.normalize(ha, ha)
    cv2.normalize(hb, hb)
    corr = float(cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL))
    return max(0.0, min(1.0, (corr + 1.0) / 2.0))


def _color_similarity(
    original_bgr: np.ndarray,
    composite_bgr: np.ndarray,
    subject_mask: np.ndarray | None = None,
) -> float:
    """刺绣友好相似度：Lab 主体色差 + HSV 色相直方图相关。"""
    a, b = _resize_match(original_bgr, composite_bgr)
    a = cv2.medianBlur(a, 5)
    b = cv2.medianBlur(b, 5)
    mask = _prepare_fg_mask(a, subject_mask)
    if not np.any(mask):
        return 0.0
    lab_a = cv2.cvtColor(a, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_b = cv2.cvtColor(b, cv2.COLOR_BGR2LAB).astype(np.float32)
    diff = lab_a[mask] - lab_b[mask]
    dist = np.sqrt(np.mean(np.sum(diff * diff, axis=1)))
    # 人像量化后 Lab 距更大，分母略放宽
    lab_sim = float(max(0.0, min(1.0, 1.0 - dist / 65.0)))
    hist_sim = _hsv_hist_corr(a, b, mask)
    return float(0.40 * lab_sim + 0.60 * hist_sim)


def _coverage_ratio(
    composite_bgr: np.ndarray,
    original_bgr: np.ndarray,
    subject_mask: np.ndarray | None = None,
) -> float:
    a, b = _resize_match(original_bgr, composite_bgr)
    if subject_mask is not None:
        sm = subject_mask
        if sm.shape[:2] != a.shape[:2]:
            sm = cv2.resize(
                sm.astype(np.uint8), (a.shape[1], a.shape[0]), interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        fg_a = sm
    else:
        fg_a = np.any(a < 245, axis=2)
    fg_b = np.any(b < 245, axis=2)
    if not np.any(fg_a):
        return 0.0
    inter = np.logical_and(fg_a, fg_b).sum()
    return float(inter / max(int(fg_a.sum()), 1))


def _stitch_quality_score(
    stitches: int,
    n_layers: int,
    *,
    has_preview: bool,
    ref_stitches: int | None,
) -> float:
    score = 0.0
    if has_preview:
        score += 0.2
    if 2000 <= stitches <= 200_000:
        score += 0.25
    elif stitches > 0:
        score += 0.1
    if 3 <= n_layers <= 10:
        score += 0.2
    elif n_layers >= 2:
        score += 0.1
    if ref_stitches and stitches > 0:
        ratio = stitches / ref_stitches
        # dense 档可能略超手工针数
        if 0.55 <= ratio <= 1.35:
            score += 0.35
        elif 0.4 <= ratio <= 1.55:
            score += 0.25
        elif 0.25 <= ratio <= 1.7:
            score += 0.15
        else:
            score += 0.05
    else:
        # 无对照时：与 dst 合理针数上界对齐（人像 dense 可达十余万）
        if 25000 <= stitches <= 200_000:
            score += 0.35
        elif 10000 <= stitches < 25000:
            score += 0.2
        else:
            score += 0.1
    return float(min(1.0, score))


def evaluate_export_dir(
    out_dir: str | Path,
    *,
    source_image: str | Path | None = None,
    reference_dst: str | Path | None = None,
) -> dict[str, Any]:
    """对导出目录打分，并给出课题五维 rubric。"""
    out_dir = Path(out_dir)
    checks: list[dict[str, Any]] = []
    scores: dict[str, float] = {}

    plan_path = out_dir / PLAN_FILENAME
    report_path = out_dir / "auto_stitch_report.json"
    dst_path = out_dir / "auto.dst"
    composite_path = out_dir / "layers_composite.png"
    preview_path = out_dir / "auto.png"
    mask_path = out_dir / "subject_mask.png"
    subject_path = out_dir / "subject.png"

    plan = None
    n_layers = 0
    if plan_path.exists():
        plan = load_plan(plan_path)
        errs = validate_plan(plan)
        checks.append({"id": "plan_valid", "ok": not errs, "detail": errs or "ok"})
        scores["plan_valid"] = 1.0 if not errs else 0.0
        n_layers = len([L for L in plan["layers"] if L.get("enabled", True)])
        color_ok = 2 <= n_layers <= 12
        checks.append({"id": "color_count_reasonable", "ok": color_ok, "detail": n_layers})
        scores["color_count"] = 1.0 if color_ok else 0.4
    else:
        checks.append({"id": "plan_valid", "ok": False, "detail": "missing"})
        scores["plan_valid"] = 0.0
        scores["color_count"] = 0.0

    stitches = 0
    if dst_path.exists():
        try:
            pat = read_dst(str(dst_path))
            stitches = pat.count_stitches()
            colors = pat.count_color_changes() + 1
            stitch_ok = 500 <= stitches <= 200_000
            checks.append(
                {"id": "dst_openable", "ok": True, "detail": {"stitches": stitches, "threads_est": colors}}
            )
            checks.append({"id": "stitch_count_reasonable", "ok": stitch_ok, "detail": stitches})
            scores["dst_openable"] = 1.0
            scores["stitch_count"] = 1.0 if stitch_ok else 0.5
        except Exception as exc:  # noqa: BLE001
            checks.append({"id": "dst_openable", "ok": False, "detail": str(exc)})
            scores["dst_openable"] = 0.0
            scores["stitch_count"] = 0.0
    else:
        checks.append({"id": "dst_openable", "ok": False, "detail": "missing"})
        scores["dst_openable"] = 0.0
        scores["stitch_count"] = 0.0

    ref_n = None
    if reference_dst and Path(reference_dst).exists() and stitches > 0:
        ref = read_dst(str(reference_dst))
        ref_n = ref.count_stitches()
        ratio = stitches / max(ref_n, 1)
        ratio_ok = 0.45 <= ratio <= 1.4
        checks.append(
            {
                "id": "vs_reference_dst",
                "ok": ratio_ok,
                "detail": {"auto": stitches, "ref": ref_n, "ratio": round(ratio, 3)},
            }
        )
        if 0.55 <= ratio <= 1.35:
            scores["vs_reference"] = 1.0
        elif ratio_ok:
            scores["vs_reference"] = 0.9
        else:
            scores["vs_reference"] = 0.4
    elif reference_dst:
        checks.append({"id": "vs_reference_dst", "ok": False, "detail": "reference missing"})

    # 相似：优先与去背主体比
    compare_src = None
    if subject_path.is_file():
        compare_src = subject_path
    elif source_image:
        compare_src = Path(source_image)

    if compare_src and composite_path.exists():
        orig = cv2.imread(str(compare_src))
        comp = cv2.imread(str(composite_path))
        if orig is not None and comp is not None:
            sm = _subject_mask_from_image(orig, mask_path if mask_path.is_file() else None)
            sim = _color_similarity(orig, comp, sm)
            cov = _coverage_ratio(comp, orig, sm)
            checks.append({"id": "color_similarity", "ok": sim >= 0.7, "detail": round(sim, 3)})
            checks.append({"id": "coverage", "ok": cov >= 0.7, "detail": round(cov, 3)})
            scores["color_similarity"] = sim
            scores["coverage"] = cov
        else:
            scores["color_similarity"] = 0.0
            scores["coverage"] = 0.0
    else:
        checks.append({"id": "color_similarity", "ok": None, "detail": "skipped"})

    has_preview = preview_path.is_file()
    stitch_q = _stitch_quality_score(
        stitches, n_layers, has_preview=has_preview, ref_stitches=ref_n
    )
    scores["stitch_quality"] = stitch_q

    # 可复现：计划 + 报告 +（LLM 或 rules）
    repro = 0.0
    if scores.get("plan_valid", 0) >= 1.0:
        repro += 0.4
    if report_path.is_file():
        repro += 0.3
    if plan and plan.get("source") in {"llm", "rules", "rules_fallback", "manual"}:
        repro += 0.2
    if (out_dir / "threads.json").is_file():
        repro += 0.1
    scores["reproducible"] = float(min(1.0, repro))

    # 设计取舍：文档或 plan notes
    trade = 0.0
    if DESIGN_NOTES.is_file():
        trade += 0.6
    if plan and (plan.get("llm_notes") or plan.get("quality")):
        trade += 0.4
    scores["tradeoffs"] = float(min(1.0, trade))

    # 课题五维（对齐介绍文档评估点）
    # 相似：主色区 + 主体覆盖（打版评估更看主区域是否盖全）
    rubric = {
        "similarity": round(
            float(
                0.32 * scores.get("color_similarity", 0.0)
                + 0.68 * scores.get("coverage", 0.0)
            ),
            3,
        ),
        "openable": round(
            float(
                0.7 * scores.get("dst_openable", 0.0)
                + 0.3 * (1.0 if has_preview else 0.0)
            ),
            3,
        ),
        "stitch_quality": round(stitch_q, 3),
        "reproducible": round(scores["reproducible"], 3),
        "tradeoffs": round(scores["tradeoffs"], 3),
    }
    rubric_overall = float(sum(rubric.values()) / len(rubric))
    rubric_pass = all(v >= 0.95 for v in rubric.values())

    usable = [v for v in scores.values()]
    overall = float(sum(usable) / max(len(usable), 1)) if usable else 0.0

    from embroideryforge.photo_eval import photo_extras

    photo = photo_extras(out_dir)
    result = {
        "out_dir": str(out_dir),
        "overall": round(overall, 3),
        "rubric": rubric,
        "rubric_overall": round(rubric_overall, 3),
        "rubric_pass_95": rubric_pass,
        "scores": {k: round(v, 3) for k, v in scores.items()},
        "checks": checks,
        "plan_source": (plan or {}).get("source") if plan else None,
        "quality": (plan or {}).get("quality") if plan else None,
        "photo": photo,
    }
    out_file = out_dir / "evaluate_report.json"
    out_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    result["report"] = str(out_file)
    return result
