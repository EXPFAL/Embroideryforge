# 区域 → 针法对照（规则 / LLM / 工业旁路）

| 区域 tag（chibi） | 规则默认 | LLM 提示倾向 | Ink/Stitch 手工常见 | 备注 |
|-------------------|----------|--------------|---------------------|------|
| clothes | tatami 0° | tatami，大填充 | Tatami | 主面积 |
| hair | tatami ~40° | tatami，角度异于衣 | Tatami | 避免同向 |
| skin | tatami 90° | tatami | Tatami | |
| eyes / patches / leaf | **satin** | satin 或密 tatami | Satin 或小块 Tatami | v0.6 起规则用 satin |
| outline / shoes | running | running | Running / Bean | 最后绣 |
| rose | tatami 密一点 | tatami | Tatami | |

中间契约字段：`tag, stitch, angle_deg, row_spacing_px, stitch_len_px, order, fill_erode, enabled`。  
校验失败 → `rules_fallback`（见 `llm.py`）。

## MSEmb / MSEmbGAN（阅读笔记）

- 用途：学习「区域形状/类别 → 针法标签」体系，不是直接替代 DST 导出。
- 本仓库映射：把论文里的 fill / satin / running 粗对应到 `tatami` / `satin` / `running`。
- 已落地启发式：`embroideryforge/portrait_style.py`（脸轻填 + 错开角度 + 肤色暗部 `features` 走针）。
- 准备期不做整网训练；开题可再以 MSEmb 标签作监督。
- 检索：`MSEmbGAN Multi-Stitch Embroidery`（IEEE TVCG）；数据可参考 HF `zubziretta/MSEmb-DATASET`。
