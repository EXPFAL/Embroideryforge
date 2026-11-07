from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from embroideryforge.autofill import run_auto_dst
from embroideryforge.evaluate import evaluate_export_dir
from embroideryforge.photo_stitch import run_photo_stitch

st.set_page_config(page_title="EmbroideryForge", layout="wide")
st.title("EmbroideryForge")
st.caption("Project 1 · stitch-auto（插画） / stitch-photo（照片绣）")

sample_chibi = ROOT / "samples" / "golden-chibi.jpg"
sample_logo = ROOT / "samples" / "golden-logo.jpg"
sample_portrait = ROOT / "samples" / "golden-portrait.jpg"
sample_landscape = ROOT / "samples" / "golden-landscape.jpg"
uploaded = st.file_uploader("上传 Logo / 图案 / 照片", type=["png", "jpg", "jpeg"])
sample_choice = st.selectbox(
    "或使用黄金样例",
    [
        "golden-chibi.jpg（插画主验收）",
        "golden-portrait.jpg（人像）",
        "golden-landscape.jpg（风景）",
        "golden-logo.jpg（历史对照）",
        "仅上传",
    ],
    index=0 if uploaded is None else 4,
)

pipeline = st.radio(
    "管线",
    ["stitch-auto（插画/EFR）", "stitch-photo（照片绣）"],
    horizontal=True,
    index=1 if (uploaded is None and sample_choice.startswith("golden-portrait")) else 0,
)

out_dir = ROOT / "exports" / "latest"
ref_dst = ROOT / "exports" / "inkscape_smoke" / "chibi_multicolor.dst"

if pipeline.startswith("stitch-photo"):
    st.subheader("彩色照片绣（对标 Wilcom 对话框隐喻）")
    c_a, c_b, c_c = st.columns(3)
    with c_a:
        ps_mode = st.selectbox(
            "模式",
            ["hybrid（脸照片绣+衣帽）", "face（仅脸 ROI）", "full（整图）"],
            index=0,
        )
    with c_b:
        ps_res = st.selectbox("分辨率", ["high", "mid", "low"], index=0)
    with c_c:
        ps_colors = st.slider("针位/色数", 8, 12, 9)
    ps_engine = st.selectbox(
        "引擎",
        ["density v4（密度场，推荐）", "poster v3（软实填）", "legacy hybrid"],
        index=0,
    )
    bg_mode = st.selectbox("去背景", ["auto", "off", "grabcut", "white"], index=0)
    run = st.button("生成照片绣", type="primary")
    do_eval = False
else:
    domain = st.selectbox("调色板域", ["auto", "portrait", "chibi"], index=0)
    efr_mode = st.selectbox("照片 EFR", ["auto（照片域默认开）", "强制开", "强制关"], index=0)
    use_llm = st.checkbox("DeepSeek 写 stitch_plan（失败回退规则）", value=False)
    quality = st.selectbox("针密度", ["dense", "standard", "draft"], index=0)
    bg_mode = st.selectbox("去背景", ["auto", "off", "grabcut", "white"], index=0)
    k = st.slider("分色 k", 6, 16, 12)
    col_run, col_eval, _ = st.columns([1, 1, 2])
    with col_run:
        run = st.button("生成刺绣方案", type="primary")
    with col_eval:
        do_eval = st.button("评估 exports/latest")


def _resolve_image(tmp: str) -> Path | None:
    if sample_choice.startswith("golden-chibi") and sample_chibi.exists() and uploaded is None:
        return sample_chibi
    if sample_choice.startswith("golden-portrait") and sample_portrait.exists() and uploaded is None:
        return sample_portrait
    if sample_choice.startswith("golden-landscape") and sample_landscape.exists() and uploaded is None:
        return sample_landscape
    if sample_choice.startswith("golden-logo") and sample_logo.exists() and uploaded is None:
        return sample_logo
    if uploaded is not None:
        image_path = Path(tmp) / uploaded.name
        image_path.write_bytes(uploaded.getvalue())
        return image_path
    return None


if run:
    with tempfile.TemporaryDirectory() as tmp:
        image_path = _resolve_image(tmp)
        if image_path is None:
            st.error("请上传图片或选择黄金样例")
            st.stop()

        if pipeline.startswith("stitch-photo"):
            mode = "hybrid"
            if ps_mode.startswith("face"):
                mode = "face"
            elif ps_mode.startswith("full"):
                mode = "full"
            with st.spinner("Photo-Stitch → DST…"):
                if ps_engine.startswith("density"):
                    from embroideryforge.density_field import run_density_v4

                    result = run_density_v4(
                        image_path,
                        out_dir,
                        n_colors=ps_colors,
                        resolution=ps_res,
                        bg_mode=bg_mode,
                        hybrid_body=mode != "face",
                    )
                else:
                    result = run_photo_stitch(
                        image_path,
                        out_dir,
                        n_colors=ps_colors,
                        resolution=ps_res,
                        bg_mode=bg_mode,
                        mode=mode,
                        poster=ps_engine.startswith("poster"),
                    )
            st.session_state["last_result"] = {
                "ok": result["ok"],
                "stitches": result["report"].get("stitches"),
                "engine": result["meta"].get("engine"),
                "mode": result["meta"].get("mode"),
                "pipeline": "stitch-photo",
                "image": str(image_path),
            }
        else:
            use_efr = None
            if efr_mode.startswith("强制开"):
                use_efr = True
            elif efr_mode.startswith("强制关"):
                use_efr = False
            with st.spinner("EFR → 分区 → plan → DST…"):
                result = run_auto_dst(
                    image_path,
                    out_dir,
                    k=k,
                    use_llm=use_llm,
                    bg_mode=bg_mode,
                    quality=quality,
                    domain=domain,
                    use_efr=use_efr,
                )
            st.session_state["last_result"] = {
                "ok": result["ok"],
                "stitches": result["report"].get("stitches"),
                "plan_source": result["report"].get("plan_source"),
                "efr": result["report"].get("efr"),
                "pipeline": "stitch-auto",
                "image": str(image_path),
            }
        st.session_state["last_eval"] = None

if do_eval:
    if not (out_dir / "auto.dst").is_file():
        st.warning("请先生成，写出 exports/latest/auto.dst")
    else:
        img = None
        report_path = out_dir / "auto_stitch_report.json"
        meta_ps = out_dir / "photo_stitch_meta.json"
        if report_path.is_file():
            src_name = json.loads(report_path.read_text(encoding="utf-8")).get("source")
            cand = ROOT / "samples" / str(src_name)
            if cand.is_file():
                img = cand
        elif meta_ps.is_file():
            src = json.loads(meta_ps.read_text(encoding="utf-8")).get("source_image")
            if src and Path(src).is_file():
                img = Path(src)
        ev = evaluate_export_dir(
            out_dir,
            source_image=img or (sample_chibi if sample_chibi.exists() else None),
            reference_dst=ref_dst if ref_dst.is_file() else None,
        )
        st.session_state["last_eval"] = ev

meta = st.session_state.get("last_result")
if meta:
    st.success(
        f"完成 · {meta.get('pipeline')} · 针数 {meta.get('stitches')} · "
        f"mode/engine={meta.get('mode') or meta.get('engine')} · "
        f"plan={meta.get('plan_source')} · efr={meta.get('efr')} · ok={meta.get('ok')}"
    )

c1, c2, c3 = st.columns(3)
compare = out_dir / "compare_preview.png"
if not compare.is_file():
    compare = out_dir / "layers_compare.png"
preview = out_dir / "auto_clean_preview.png"
efr_img = out_dir / "efr.png"
tone = out_dir / "tone_preview.png"
layout = out_dir / "layout_preview.png"
if not preview.is_file():
    preview = out_dir / "auto.png"
with c1:
    st.subheader("对照 / 布局")
    if compare.is_file():
        st.image(str(compare))
    if layout.is_file():
        st.caption("hybrid 布局（橙=脸照片绣，绿=衣帽）")
        st.image(str(layout))
    if efr_img.is_file():
        st.caption("EFR")
        st.image(str(efr_img))
    if tone.is_file():
        st.caption("色调量化")
        st.image(str(tone))
with c2:
    st.subheader("针迹预览")
    if preview.is_file():
        st.image(str(preview))
with c3:
    st.subheader("计划 / 调色板")
    plan_path = out_dir / "stitch_plan.json"
    meta_path = out_dir / "photo_stitch_meta.json"
    if plan_path.is_file():
        st.json(json.loads(plan_path.read_text(encoding="utf-8")))
    elif meta_path.is_file():
        st.json(json.loads(meta_path.read_text(encoding="utf-8")))

dst = out_dir / "auto.dst"
if dst.is_file():
    st.download_button(
        "下载 DST",
        data=dst.read_bytes(),
        file_name="auto.dst",
        mime="application/octet-stream",
    )

ev = st.session_state.get("last_eval")
if ev:
    st.subheader("课题五维评估")
    st.metric("rubric_overall", ev.get("rubric_overall"))
    st.json({"rubric_pass_95": ev.get("rubric_pass_95"), "rubric": ev.get("rubric")})

st.caption(f"产物：`{out_dir}`")
