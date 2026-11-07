# Project 1 冻结基线

**冻结日：** 2026-11-05  
**状态：** 可演示 / 可复现 / 对照完成

## 锁定文件（勿随意覆盖）

| 文件 | 作用 |
|------|------|
| `samples/golden-chibi.jpg` | 主测图 |
| `exports/segment_chibi_v4/` | 自动分区 + `auto.dst` 全套产物 |
| `exports/inkscape_smoke/chibi_multicolor.dst` | 手工 Ink/Stitch 对照 |
| `docs/课题一设计说明-chibi基线.md` | 说明与取舍 |

企鹅样例 `golden-logo.jpg` / `penguin.dst` 仅作历史对照，不再作为主验收。

人像样例：`samples/golden-portrait.jpg` → `exports/portrait_v1/`（`--domain portrait`，evaluate ≥0.95）。

## 复现命令

```powershell
cd "D:\WORK\TEST\Project 1"
.\.venv\Scripts\Activate.ps1
python -m pytest -q
python -m embroideryforge stitch-auto samples\golden-chibi.jpg --out exports\segment_chibi_v4_repro --mode palette --k 10
```

新实验请输出到**新目录**（如 `segment_chibi_v5`），保留 `v4` 不动。
