# EmbroideryForge 产品说明

## 用户主路径

1. 打开 Streamlit 工作台  
2. 上传图案或使用黄金样例  
3. 调整缩放并生成  
4. 查看轮廓/主色/针迹预览与针法建议  
5. 下载 DST，阅读设计说明  

## 与课题评估对齐

| 评估点 | 产品对应 |
|--------|----------|
| 轮廓/颜色相似 | 分析报告 + 轮廓叠加图 |
| 文件可打开预览 | DST + PNG |
| 针迹合理 | 针数与建议表 |
| 可复现 | CLI + report.json |
| 设计取舍 | design_notes.md |

## 成熟度路线

- v0.1：轮廓走针 + 主色建议  
- v0.2：`segment` 自动分色分区 → 分层 SVG + 标签  
- v0.3：`stitch-auto` 按层 Tatami/Running 导出 `auto.dst`（**chibi_v4 已冻结**）  
- v0.4：结构化 `stitch_plan.json`（规则 / DeepSeek）+ `evaluate`  
- v0.5：去背景（rembg/auto）+ `--quality dense` + 课题五维 `rubric`（目标 ≥0.95）  

### 结构化针法计划 + DeepSeek

`stitch_plan.json` 是规则 / LLM / 手工的共用出口。密钥放项目根 `.env`（已 gitignore），变量名 `DEEPSEEK_API_KEY`。

```powershell
# 规则版
python -m embroideryforge plan exports\plan_pipeline_v1\layers.json

# DeepSeek 写 plan（失败自动回退 rules_fallback）
python -m embroideryforge plan exports\plan_pipeline_v1\layers.json --llm --out exports\plan_pipeline_v1\stitch_plan_llm.json
python -m embroideryforge stitch-auto samples\golden-chibi.jpg --out exports\llm_run --llm --k 10
python -m embroideryforge evaluate exports\llm_run --image samples\golden-chibi.jpg --ref-dst exports\inkscape_smoke\chibi_multicolor.dst
```

### 自动分区 CLI

扁平插画用默认 `palette`（先聚类再挂名）。看对比图：`layers_compare.png`（左原图 / 右分层合成）。

```powershell
cd "D:\WORK\TEST\Project 1"
.\.venv\Scripts\Activate.ps1
python -m embroideryforge segment samples\golden-chibi.jpg --out exports\segment_chibi_v4 --mode palette --k 10
```

产物：`layers.svg`、`layers.json`、`layers_compare.png`、（经 stitch-auto）`stitch_plan.json`。若某色仍糊：加大 `--k`，或改 `segment.py` 里 `CHIBI_PALETTE` / 粉系饱和度阈值。
