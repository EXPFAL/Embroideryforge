# StarVector 旁路说明

## 定位

Logo/图标 → SVG（[StarVector](https://github.com/joanrod/star-vector)，论文 arXiv:2312.11556）。  
与本仓库关系：**矢量旁路的研究选项**；主验收仍是 `segment` 栅格分区 + `stitch-auto`。

## 本地策略（2026-11）

| 项 | 状态 |
|----|------|
| 权重 | `starvector/starvector-1b-im2svg`（需 GPU / 较大磁盘） |
| 本机默认 | **不强制安装**；用 **VTracer** 覆盖「位图→SVG」证据 |
| 命令 | `python -m embroideryforge vectorize samples\golden-chibi.jpg --out exports\vectorize_chibi` |

若日后有 CUDA：

1. 按官方 README 建独立 env（避免拖垮主 venv）  
2. 对 `golden-logo.jpg` / `golden-chibi.jpg` 出 SVG  
3. 把 SVG 与 `exports/vectorize_*/vector.svg` 并排放进对照笔记  

自然照片效果差——与资源清单结论一致；扁平插画更匹配。
