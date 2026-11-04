# 课题一设计说明 · Chibi 基线（2026-11-04/05）

## 1. 目标

验证「图像 → 可检查刺绣方案」链路：大模型/规则负责理解与编排，开源工具负责表达，自动指标负责验收。  
本基线用扁平插画 `golden-chibi.jpg`，对照两条实现路径。

## 2. 输入与产物

| 项 | 路径 |
|----|------|
| 输入图 | `samples/golden-chibi.jpg` |
| 自动分区 | `exports/segment_chibi_v4/layers.svg` + `layers.json` |
| 自动 DST | `exports/segment_chibi_v4/auto.dst` |
| 手工 Ink/Stitch DST | `exports/inkscape_smoke/chibi_multicolor.dst` |
| 对照 JSON | `exports/segment_chibi_v4/dst_compare.json` |

自动分层标签（v4）：`clothes` / `hair` / `skin` / `eyes` / `outline`。

## 3. 两条路径

### A. 手工（Inkscape + Ink/Stitch）

导入 → 位图描摹 → 解组清理 → 逐块 Params（Tatami / Running）→ 另存 DST。  
证明工业旁路开源链可跑通，但**解组/打 tag 繁琐**，不适合作主劳动力。

### B. 自动（EmbroideryForge）

```text
segment(palette) → 分层 SVG/JSON → stitch-auto（Tatami/Running）→ auto.dst
```

命令：

```powershell
python -m embroideryforge stitch-auto samples\golden-chibi.jpg --out exports\segment_chibi_v4 --mode palette --k 10
```

针法约定：衣 0°、发 45°、肤 90°；outline 走针；绣序填充先、描边后。

## 4. 结果对照（pyembroidery 回读）

| 方案 | 针数 | 换色次数 | 体积 |
|------|------|----------|------|
| 手工 Ink/Stitch | 63013 | 5 | 185 KB |
| 自动 EmbroideryForge | ~1.4e4 | 4–8* | ~42 KB |
| 早期企鹅单色 | 4850 | 0 | 15 KB |

\*换色统计口径与块切分方式有关；自动报告以 `auto_stitch_report.json` 的 layers 列表为准。

## 5. 设计取舍

1. **分区**：盲 k-means 易把发/肤糊在一起；改为「聚类 + 调色板挂名」，v4 可接受。  
2. **针密度**：自动 Tatami 行距更大，针数显著少于手工——先保可检查闭环，再调美学密度。  
3. **职责**：程序做出可打开 DST + 分层说明；人审对比图 `layers_compare.png` 与仿真预览。  
4. **Ink/Stitch**：保留为质量上界与教学对照，不作为批处理主路径。

## 6. 模型 / 工具 / 人工

| 角色 | 工作 |
|------|------|
| 规则/算法 | 分色、打标、生成 `stitch_plan.json`、Tatami/Running、`evaluate` |
| 工具 | OpenCV、pyembroidery；对照用 Inkscape/Ink/Stitch |
| 人工 | 选样例、验收分层、解释取舍、开题问题提炼 |
| LLM（DeepSeek） | 输入 layers 摘要 → 写 `stitch_plan.json`；校验失败 → `rules_fallback` |

中间契约：`stitch_plan.json`（`schema_version: 1`）。`source`：`rules` / `llm` / `rules_fallback` / `manual`。  
密钥：`DEEPSEEK_API_KEY`（`.env`，勿提交）。CLI：`plan --llm` 或 `stitch-auto --llm`。

## 7. 验收命令（五维 ≥0.95）

```powershell
python -m embroideryforge stitch-auto samples\golden-chibi.jpg --out exports\accept_flat_v3 --llm --quality dense --bg auto
python -m embroideryforge evaluate exports\accept_flat_v3 --image samples\golden-chibi.jpg --ref-dst exports\inkscape_smoke\chibi_multicolor.dst
```

复杂背景同理加 `--bg auto`（rembg）。默认 `--quality dense` 逼近手工针数。

## 8. 下一步

- 开题正文固化「LLM 只出 plan → 工具执行 → evaluate」  
- 可选进入 Project 2  
