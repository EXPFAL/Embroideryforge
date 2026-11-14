"""DeepSeek 识图比对裁判：照片绣结构化评分（不改 dens）。

输出 vision_judge.json / vision_judge_pair.json；辅评人评清单，不替代实绣。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from embroideryforge.llm import (
    DEFAULT_MODEL,
    _extract_json_object,
    chat_completion,
    encode_image_data_url,
    get_api_key,
)

PORTRAIT_DIMS = (
    "facial_features",
    "head_body_color",
    "background",
    "seams",
    "stitch_count",
    "color_accuracy",
)

ANIMAL_DIMS = (
    "species_pose",
    "fur_color",
    "fur_silhouette",
    "foliage_bg",
    "stitch_count",
)

RUBRIC_LABELS = {
    "portrait": {
        "facial_features": "五官可辨（眼/嘴是否糊死）",
        "head_body_color": "头身同色（脸与衣领是否两套色阶）",
        "background": "背景可读（建筑/树叶是否一团）",
        "seams": "缝隙/接缝（轮廓露底或硬边光晕）",
        "stitch_count": "针数合理（约 15–40 万场景）",
        "color_accuracy": "色准（肤/衣主色是否偏灰串色）",
    },
    "animal": {
        "species_pose": "物种/姿态可辨",
        "fur_color": "毛色大致对（勿发灰串肤粉）",
        "fur_silhouette": "毛发轮廓/明暗（不要求丝理）",
        "foliage_bg": "绿叶/背景可读",
        "stitch_count": "针数合理（约 15–45 万）",
    },
}


def load_photo_meta(export_dir: Path) -> dict[str, Any]:
    p = export_dir / "photo_stitch_meta.json"
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def resolve_rubric(rubric: str, meta: dict[str, Any]) -> str:
    r = (rubric or "auto").strip().lower()
    if r in ("portrait", "animal"):
        return r
    domain = str(meta.get("domain") or "").lower()
    if domain == "animal":
        return "animal"
    return "portrait"


def resolve_source_image(
    export_dir: Path,
    meta: dict[str, Any],
    image: Path | None,
) -> Path | None:
    """比对原图：优先导出目录 subject.png（成绣实际输入），再 --image / meta。"""
    sub = export_dir / "subject.png"
    if sub.is_file():
        return sub
    if image is not None and image.is_file():
        return image
    for key in ("source_image", "image", "input"):
        v = meta.get(key)
        if isinstance(v, str) and Path(v).is_file():
            return Path(v)
    return None


def resolve_preview(export_dir: Path) -> Path:
    for name in (
        "auto_stitch_thick.png",
        "auto_stitch_thick_white.png",
        "compare_preview.png",
        "auto_clean_preview.png",
    ):
        p = export_dir / name
        if p.is_file():
            return p
    raise FileNotFoundError(f"no stitch preview in {export_dir}")


def meta_brief(meta: dict[str, Any]) -> dict[str, Any]:
    v6 = meta.get("v6") if isinstance(meta.get("v6"), dict) else {}
    return {
        "domain": meta.get("domain"),
        "stitches": meta.get("stitches"),
        "color_changes": meta.get("color_changes"),
        "v66_portrait_p0": v6.get("v66_portrait_p0"),
        "v66_animal_a1": v6.get("v66_animal_a1"),
        "note": (meta.get("note") or "")[:160],
    }


def _dim_schema(dims: tuple[str, ...]) -> dict[str, Any]:
    return {
        d: {"score": "1-5 int", "evidence": "one short Chinese sentence pointing at the image"}
        for d in dims
    }


def build_judge_prompt(
    *,
    rubric: str,
    brief: dict[str, Any],
    pair: bool = False,
) -> str:
    labels = RUBRIC_LABELS[rubric]
    dims = PORTRAIT_DIMS if rubric == "portrait" else ANIMAL_DIMS
    lines = [f"- {k}: {labels[k]}" for k in dims]
    schema = {
        "rubric": rubric,
        "dimensions": _dim_schema(dims),
        "mean_score": "float average of dimension scores",
        "demo_ok": "bool: true if mean>=3 and no dimension is 1",
        "hard_fails": ["optional short tags e.g. face_blob"],
        "summary": "2-3 Chinese sentences",
        "suggested_levers": ["optional next experiment knobs, no code"],
    }
    if pair:
        schema = {
            "rubric": rubric,
            "dir_a": {
                "dimensions": _dim_schema(dims),
                "mean_score": "float",
                "demo_ok": "bool",
            },
            "dir_b": {
                "dimensions": _dim_schema(dims),
                "mean_score": "float",
                "demo_ok": "bool",
            },
            "winner": "a|b|tie",
            "delta_mean": "float mean_b - mean_a",
            "summary": "2-3 Chinese sentences comparing A vs B",
            "suggested_levers": ["optional"],
        }
    task = (
        "Compare ORIGINAL photo vs EMBROIDERY stitch preview. "
        if not pair
        else "You see ORIGINAL, then embroidery A, then embroidery B. Score A and B separately, pick winner. "
    )
    return (
        task
        + "You are a strict embroidery photo-stitch judge. "
        "Image labeled ORIGINAL is subject.png (the processed frame fed to stitch). "
        "Score only what is visible; do not invent sew-out physics. "
        "Reply with ONE JSON object only (no markdown). "
        f"Rubric={rubric}. Dimensions:\n"
        + "\n".join(lines)
        + "\nExport meta (may help stitch_count):\n"
        + json.dumps(brief, ensure_ascii=False)
        + "\nRequired JSON shape:\n"
        + json.dumps(schema, ensure_ascii=False)
    )


def validate_single_scores(obj: dict[str, Any], dims: tuple[str, ...]) -> list[str]:
    errors: list[str] = []
    dims_obj = obj.get("dimensions")
    if not isinstance(dims_obj, dict):
        return ["missing dimensions object"]
    scores: list[float] = []
    for d in dims:
        cell = dims_obj.get(d)
        if not isinstance(cell, dict):
            errors.append(f"missing dimension {d}")
            continue
        sc = cell.get("score")
        ev = cell.get("evidence")
        try:
            si = int(sc)
        except (TypeError, ValueError):
            errors.append(f"{d}.score not int")
            continue
        if si < 1 or si > 5:
            errors.append(f"{d}.score out of 1-5")
        else:
            scores.append(float(si))
        if not isinstance(ev, str) or not ev.strip():
            errors.append(f"{d}.evidence empty")
    if scores and len(scores) == len(dims):
        mean = sum(scores) / len(scores)
        obj["mean_score"] = round(mean, 3)
        hard = any(s <= 1 for s in scores)
        obj["demo_ok"] = bool(mean >= 3.0 and not hard)
    return errors


def validate_judge_payload(obj: dict[str, Any], *, rubric: str, pair: bool) -> list[str]:
    dims = PORTRAIT_DIMS if rubric == "portrait" else ANIMAL_DIMS
    if pair:
        errors: list[str] = []
        for key in ("dir_a", "dir_b"):
            block = obj.get(key)
            if not isinstance(block, dict):
                errors.append(f"missing {key}")
                continue
            errors.extend(validate_single_scores(block, dims))
        w = obj.get("winner")
        if w not in ("a", "b", "tie"):
            errors.append("winner must be a|b|tie")
        if "summary" not in obj or not str(obj.get("summary") or "").strip():
            errors.append("missing summary")
        # delta
        try:
            ma = float((obj.get("dir_a") or {}).get("mean_score") or 0)
            mb = float((obj.get("dir_b") or {}).get("mean_score") or 0)
            obj["delta_mean"] = round(mb - ma, 3)
        except (TypeError, ValueError):
            pass
        return errors
    errors = validate_single_scores(obj, dims)
    if "summary" not in obj or not str(obj.get("summary") or "").strip():
        errors.append("missing summary")
    obj["rubric"] = rubric
    return errors


def _image_block(data_url: str, *, detail: str = "high") -> dict[str, Any]:
    return {
        "type": "image_url",
        "image_url": {"url": data_url, "detail": detail},
    }


def judge_photo_export(
    export_dir: str | Path,
    *,
    image: str | Path | None = None,
    rubric: str = "auto",
    include_compare: bool = True,
    max_side: int = 1280,
    api_key: str | None = None,
    timeout_s: float = 150.0,
) -> dict[str, Any]:
    export_dir = Path(export_dir)
    if not export_dir.is_dir():
        raise FileNotFoundError(export_dir)
    meta = load_photo_meta(export_dir)
    rubric_s = resolve_rubric(rubric, meta)
    src = resolve_source_image(export_dir, meta, Path(image) if image else None)
    preview = resolve_preview(export_dir)
    if src is None:
        raise FileNotFoundError(
            f"subject.png / source image not found for {export_dir}; pass --image"
        )

    brief = meta_brief(meta)
    brief["export_dir"] = str(export_dir)
    brief["preview"] = preview.name
    brief["source"] = str(src)
    brief["source_role"] = "subject.png" if src.name.lower() == "subject.png" else "image"

    content: list[dict[str, Any]] = [
        {"type": "text", "text": build_judge_prompt(rubric=rubric_s, brief=brief, pair=False)},
        {
            "type": "text",
            "text": f"Image 1 = ORIGINAL subject frame ({src.name}; stitch input / subject.png).",
        },
        _image_block(encode_image_data_url(src, max_side=max_side)),
        {"type": "text", "text": f"Image 2 = EMBROIDERY preview ({preview.name})."},
        _image_block(encode_image_data_url(preview, max_side=max_side)),
    ]
    cmp = export_dir / "compare_preview.png"
    if include_compare and cmp.is_file() and cmp.resolve() != preview.resolve():
        content.append({"type": "text", "text": "Image 3 = compare_preview (optional context)."})
        content.append(_image_block(encode_image_data_url(cmp, max_side=max_side), detail="low"))

    report: dict[str, Any] = {
        "ok": False,
        "provider": "deepseek",
        "model": DEFAULT_MODEL,
        "rubric": rubric_s,
        "export_dir": str(export_dir),
        "source_image": str(src),
        "preview": str(preview),
        "meta_brief": brief,
    }
    key = api_key or get_api_key()
    if not key:
        report["error"] = "DEEPSEEK_API_KEY not set"
        _write_json(export_dir / "vision_judge.json", report)
        return report

    try:
        raw = chat_completion(
            [
                {
                    "role": "system",
                    "content": "Return only valid JSON for embroidery vision judging.",
                },
                {"role": "user", "content": content},
            ],
            api_key=key,
            temperature=0.15,
            timeout_s=timeout_s,
        )
        report["raw_preview"] = raw[:800]
        obj = _extract_json_object(raw)
        errors = validate_judge_payload(obj, rubric=rubric_s, pair=False)
        report["scores"] = obj
        report["validation_errors"] = errors
        report["ok"] = len(errors) == 0
        if errors:
            report["error"] = "; ".join(errors)
    except Exception as exc:  # noqa: BLE001
        report["error"] = str(exc)
        report["ok"] = False

    _write_json(export_dir / "vision_judge.json", report)
    report["path"] = str(export_dir / "vision_judge.json")
    return report


def judge_photo_pair(
    dir_a: str | Path,
    dir_b: str | Path,
    *,
    image: str | Path | None = None,
    rubric: str = "auto",
    max_side: int = 1280,
    api_key: str | None = None,
    timeout_s: float = 180.0,
) -> dict[str, Any]:
    dir_a = Path(dir_a)
    dir_b = Path(dir_b)
    meta_a = load_photo_meta(dir_a)
    meta_b = load_photo_meta(dir_b)
    rubric_s = resolve_rubric(rubric, meta_a if meta_a else meta_b)
    # A/B 对照：原图固定用 subject.png（优先 dir_a，其次 dir_b）
    src = resolve_source_image(dir_a, meta_a, None)
    if src is None:
        src = resolve_source_image(dir_b, meta_b, None)
    if src is None and image is not None and Path(image).is_file():
        src = Path(image)
    if src is None:
        raise FileNotFoundError(
            "subject.png not found in dir_a/dir_b; pass --image as fallback"
        )

    prev_a = resolve_preview(dir_a)
    prev_b = resolve_preview(dir_b)
    brief = {
        "dir_a": {**meta_brief(meta_a), "dir": str(dir_a), "preview": prev_a.name},
        "dir_b": {**meta_brief(meta_b), "dir": str(dir_b), "preview": prev_b.name},
        "source": str(src),
        "source_role": "subject.png",
    }
    content: list[dict[str, Any]] = [
        {"type": "text", "text": build_judge_prompt(rubric=rubric_s, brief=brief, pair=True)},
        {
            "type": "text",
            "text": f"Image 1 = ORIGINAL subject frame ({src.name}; stitch input / subject.png).",
        },
        _image_block(encode_image_data_url(src, max_side=max_side)),
        {"type": "text", "text": f"Image 2 = embroidery A ({dir_a.name} / {prev_a.name})."},
        _image_block(encode_image_data_url(prev_a, max_side=max_side)),
        {"type": "text", "text": f"Image 3 = embroidery B ({dir_b.name} / {prev_b.name})."},
        _image_block(encode_image_data_url(prev_b, max_side=max_side)),
    ]

    out_path = dir_b / "vision_judge_pair.json"
    report: dict[str, Any] = {
        "ok": False,
        "provider": "deepseek",
        "model": DEFAULT_MODEL,
        "rubric": rubric_s,
        "dir_a": str(dir_a),
        "dir_b": str(dir_b),
        "source_image": str(src),
        "meta_brief": brief,
    }
    key = api_key or get_api_key()
    if not key:
        report["error"] = "DEEPSEEK_API_KEY not set"
        _write_json(out_path, report)
        report["path"] = str(out_path)
        return report

    try:
        raw = chat_completion(
            [
                {
                    "role": "system",
                    "content": "Return only valid JSON for pairwise embroidery vision judging.",
                },
                {"role": "user", "content": content},
            ],
            api_key=key,
            temperature=0.15,
            timeout_s=timeout_s,
        )
        report["raw_preview"] = raw[:800]
        obj = _extract_json_object(raw)
        errors = validate_judge_payload(obj, rubric=rubric_s, pair=True)
        report["scores"] = obj
        report["validation_errors"] = errors
        report["ok"] = len(errors) == 0
        if errors:
            report["error"] = "; ".join(errors)
    except Exception as exc:  # noqa: BLE001
        report["error"] = str(exc)
        report["ok"] = False

    _write_json(out_path, report)
    report["path"] = str(out_path)
    return report


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
