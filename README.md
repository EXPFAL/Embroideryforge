# EmbroideryForge（Project 1）

AI 辅助机器刺绣设计工作台：图片 → 分析/分区 → 针迹 → DST/预览 → 设计说明。

## 快速开始

```powershell
cd "D:\WORK\TEST\Project 1"
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest -q
python -m streamlit run app\streamlit_app.py
```

或：`.\scripts\run_app.ps1`

## 结构

```
app/           Streamlit GUI
embroideryforge/  核心引擎（可 CLI）
samples/       黄金样例
tests/         冒烟与单元测试
exports/       运行输出（gitignore 大文件可调）
docs/          产品说明与评估对照
```

## CLI

```powershell
python -m embroideryforge.cli analyze samples\golden-logo.jpg --out exports\run1
```
