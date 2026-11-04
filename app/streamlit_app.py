from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from embroideryforge.pipeline import run_pipeline

st.set_page_config(page_title="EmbroideryForge", layout="wide")
st.title("EmbroideryForge")
st.caption("Project 1 · AI 辅助机器刺绣设计工作台")

sample_default = ROOT / "samples" / "golden-logo.jpg"
uploaded = st.file_uploader("上传 Logo / 图案", type=["png", "jpg", "jpeg"])
use_golden = st.checkbox("使用黄金样例 golden-logo.jpg", value=uploaded is None)
scale = st.slider("针迹缩放", 0.1, 1.0, 0.4, 0.05)

col_run, _ = st.columns([1, 3])
with col_run:
    run = st.button("生成刺绣方案", type="primary")

if run:
    with tempfile.TemporaryDirectory() as tmp:
        if use_golden and sample_default.exists() and uploaded is None:
            image_path = sample_default
        elif uploaded is not None:
            image_path = Path(tmp) / uploaded.name
            image_path.write_bytes(uploaded.getvalue())
        else:
            st.error("请上传图片或勾选黄金样例")
            st.stop()

        out_dir = ROOT / "exports" / "latest"
        result = run_pipeline(image_path, out_dir, scale=scale)

    st.success(f"完成 · 针数 {result['export']['stitches']}")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("轮廓叠加")
        st.image(str(result["artifacts"]["overlay"]))
    with c2:
        st.subheader("主色")
        st.image(str(result["artifacts"]["swatch"]))
    with c3:
        st.subheader("针迹预览")
        st.image(str(result["export"]["png"]))

    st.subheader("针法建议")
    st.json(result["stitch_plan"])
    st.download_button(
        "下载 DST",
        data=Path(result["export"]["dst"]).read_bytes(),
        file_name="embroidery.dst",
        mime="application/octet-stream",
    )
    st.markdown(Path(result["artifacts"]["notes"]).read_text(encoding="utf-8"))
