# Project 1 冻结基线

**冻结日：** 2026-11-07（v0.15 轻量续冻）  
**状态：** 课题一可验收；照片绣可演示 / 可复现（创新轨）

## 课题一 · 锁定文件（勿随意覆盖）

| 文件 | 作用 |
|------|------|
| `samples/golden-chibi.jpg` | 主测图 |
| `exports/segment_chibi_v4/` | 自动分区 + `auto.dst` 全套产物 |
| `exports/accept_flat_v3/` | 五维验收冻结（`evaluate_report.json` → `rubric_pass_95`） |
| `exports/inkscape_smoke/chibi_multicolor.dst` | 手工 Ink/Stitch 对照 |
| `docs/课题一设计说明-chibi基线.md` | 说明与取舍 |
| `docs/评估对照-课题一.md` | 自检表 |

企鹅样例 `golden-logo.jpg` / `penguin.dst` 仅作历史对照，不再作为主验收。

人像样例（EFR/portrait 域）：`samples/golden-portrait.jpg` → `exports/portrait_v1/`（`--domain portrait`）。

### 课题一复现

```powershell
cd "D:\WORK\TEST\Project 1"
.\.venv\Scripts\Activate.ps1
python -m pytest -q
python -m embroideryforge stitch-auto samples\golden-chibi.jpg --out exports\segment_chibi_v4_repro --mode palette --k 10
python -m embroideryforge evaluate exports\accept_flat_v3 --image samples\golden-chibi.jpg --ref-dst exports\inkscape_smoke\chibi_multicolor.dst
```

新实验请输出到**新目录**，保留 `v4` / `accept_flat_v3` 不动。

---

## 照片绣 · 冻结样例（V6.3d 基线；最新实验 V6.4）

**统一命令（场景分轨）：**

```powershell
python -m embroideryforge stitch-photo <样例> --out <新目录> --mode hybrid --density-unified --bg keep --colors 12 --resolution high
```

| 样例 | 输入 | 冻结导出（勿覆盖；新实验递增版本号） | 必看文件 |
|------|------|--------------------------------------|----------|
| 牛仔场景 | `samples/test_figure_cowboy.jpg` | 基线 `…/figure_cowboy_scene_v63d/`；最新 `…/figure_cowboy_scene_v65c/` | `compare_preview.png`、`auto_stitch_thick.png`、`photo_stitch_meta.json`（查 `v6.v66_*`） |
| 合成脸/武侠 | `samples/test_figure_kobe_wuxia.jpg` | 基线 `…/figure_kobe_wuxia_v63d/` | 同上 |
| Homelander 舱内 | `samples/test_figure_kobe_homelander.jpg` | 最新 `…/figure_kobe_homelander_v64d/` | 同上 |
| 皇庭指向 | `samples/test_figure_kobe_emperor.jpg` | 最新 `…/figure_kobe_emperor_v66/` | 红墙金瓦；V6.6 压平色板 |
| 多人院落 | `samples/test_figure_kobe_multi.jpg` | 最新 `…/figure_kobe_multi_v66/` | V6.6 压平色板；多脸检测保留、细节加压已回退 |
| 半身制服（抠底 V6） | `samples/test_figure_uniform.jpg` | `exports/photo_stitch_exp/figure_uniform_v6/`（若本地有） | 同上 |

命名约定：自 `v63d` 起只追加版本后缀（如 `v64`、`v64b`），**不要**原地覆盖冻结目录。

照片绣验收：人评清单见 [`路线-照片绣PhotoStitch.md`](路线-照片绣PhotoStitch.md) 附录；**不用** `rubric_pass_95`。
