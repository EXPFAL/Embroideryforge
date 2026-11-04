"""DeepSeek 生成 stitch_plan.json（失败回退规则版）。

密钥：环境变量 DEEPSEEK_API_KEY，或项目根目录 .env（勿提交）。
文档：https://api-docs.deepseek.com/zh-cn/
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from embroideryforge.plan import SCHEMA_VERSION, apply_quality, build_rules_plan, validate_plan

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_dotenv(path: Path | None = None) -> None:
    """把 .env 里未设置的键载入 os.environ（不覆盖已有环境变量）。"""
    candidates = [path] if path else [_project_root() / ".env", Path.cwd() / ".env"]
    for env_path in candidates:
        if env_path is None or not env_path.is_file():
            continue
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip("'").strip('"')
            if key and key not in os.environ:
                os.environ[key] = val
        break


def get_api_key() -> str | None:
    load_dotenv()
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    return key or None


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        obj = json.loads(text[start : end + 1])
        if isinstance(obj, dict):
            return obj
    raise ValueError("LLM response does not contain a JSON object")


def chat_completion(
    messages: list[dict[str, str]],
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    temperature: float = 0.2,
    timeout_s: float = 90.0,
) -> str:
    """调用 DeepSeek Chat Completions，返回 assistant content。"""
    key = api_key or get_api_key()
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY not set")
    base = (base_url or os.environ.get("DEEPSEEK_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    model_name = model or os.environ.get("DEEPSEEK_MODEL") or DEFAULT_MODEL
    url = f"{base}/chat/completions"
    body = {
        "model": model_name,
        "messages": messages,
        "stream": False,
        "temperature": temperature,
        # 结构化针法不需要长思维链
        "thinking": {"type": "disabled"},
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"DeepSeek HTTP {exc.code}: {detail[:500]}") from exc
    try:
        return str(payload["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"unexpected DeepSeek response: {payload!r}") from exc


def _layers_brief(layers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    brief = []
    for L in layers:
        brief.append(
            {
                "tag": L.get("tag"),
                "hex": L.get("hex"),
                "area_px": L.get("area_px"),
                "role_hint": L.get("role_hint"),
                "contour_count": L.get("contour_count"),
            }
        )
    return brief


def merge_llm_into_rules(
    rules: dict[str, Any],
    llm_obj: dict[str, Any],
) -> dict[str, Any]:
    """以规则 plan 为底，用 LLM 覆盖 stitch/angle/spacing/order/enabled。"""
    llm_layers = llm_obj.get("layers")
    if not isinstance(llm_layers, list):
        raise ValueError("LLM plan missing layers[]")
    by_tag = {str(L.get("tag")): L for L in llm_layers if isinstance(L, dict) and L.get("tag")}
    merged_layers: list[dict[str, Any]] = []
    for base in rules["layers"]:
        tag = str(base["tag"])
        overlay = by_tag.get(tag, {})
        entry = dict(base)
        for key in (
            "stitch",
            "angle_deg",
            "row_spacing_px",
            "stitch_len_px",
            "order",
            "fill_erode",
            "enabled",
            "role_hint",
        ):
            if key in overlay and overlay[key] is not None:
                entry[key] = overlay[key]
        # 颜色始终以分割结果为准
        entry["hex"] = base.get("hex", entry.get("hex"))
        if entry.get("stitch") == "running":
            entry["row_spacing_px"] = None
            entry["fill_erode"] = int(entry.get("fill_erode") or 0)
        merged_layers.append(entry)
    plan = {
        "schema_version": SCHEMA_VERSION,
        "source": "llm",
        "target_height_mm": float(
            llm_obj.get("target_height_mm", rules.get("target_height_mm", 70.0))
        ),
        "layers": merged_layers,
        "llm_notes": llm_obj.get("notes") or llm_obj.get("rationale"),
    }
    errors = validate_plan(plan)
    if errors:
        raise ValueError("; ".join(errors))
    return plan


def build_llm_plan(
    layers: list[dict[str, Any]],
    *,
    target_height_mm: float = 70.0,
    api_key: str | None = None,
    model: str | None = None,
    fallback: bool = True,
    quality: str = "dense",
) -> dict[str, Any]:
    """调用 DeepSeek 生成 plan；校验失败时可选回退 rules。"""
    rules = build_rules_plan(
        layers, target_height_mm=target_height_mm, source="rules", quality="standard"
    )
    system = (
        "You are an embroidery stitch planner. "
        "Reply with ONE JSON object only (no markdown). "
        f"schema_version must be {SCHEMA_VERSION}. "
        "For each input layer tag, set: tag, stitch (tatami|running), angle_deg, "
        "row_spacing_px (null if running), stitch_len_px, order (number), "
        "fill_erode (0-2), enabled (bool). "
        "Rules of thumb: outline/shoes -> running; large fills -> tatami; "
        "vary fill angles between adjacent regions; smaller regions denser spacing; "
        "embroider fills before outline. Keep all given tags."
    )
    user = {
        "task": "produce stitch_plan",
        "target_height_mm": target_height_mm,
        "layers": _layers_brief(layers),
        "example_shape": {
            "schema_version": SCHEMA_VERSION,
            "source": "llm",
            "target_height_mm": target_height_mm,
            "layers": [
                {
                    "tag": "clothes",
                    "stitch": "tatami",
                    "angle_deg": 0,
                    "row_spacing_px": 2.0,
                    "stitch_len_px": 3.2,
                    "order": 10,
                    "fill_erode": 1,
                    "enabled": True,
                }
            ],
            "notes": "short rationale",
        },
    }
    meta: dict[str, Any] = {"provider": "deepseek", "model": model or DEFAULT_MODEL}
    try:
        content = chat_completion(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
            ],
            api_key=api_key,
            model=model,
        )
        meta["raw_preview"] = content[:400]
        llm_obj = _extract_json_object(content)
        plan = merge_llm_into_rules(rules, llm_obj)
        plan = apply_quality(plan, quality)
        plan["llm_meta"] = meta
        return plan
    except Exception as exc:  # noqa: BLE001 — 明确回退
        if not fallback:
            raise
        rules = build_rules_plan(
            layers,
            target_height_mm=target_height_mm,
            source="rules_fallback",
            quality=quality,
        )
        rules["llm_meta"] = {**meta, "error": str(exc), "fallback": True}
        return rules
