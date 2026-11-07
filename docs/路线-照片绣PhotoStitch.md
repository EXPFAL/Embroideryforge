# 路线：照片绣（Photo-Stitch）— 对标 Wilcom 彩色照片绣

**状态**：P0–P1 完成；**P2 hybrid + Streamlit 已接通**  

**对标参照**：Wilcom「彩色照片绣 / Photo Flash」（多针位、分辨率档、色调预览）  
**与 EFR 关系**：EFR（`docs/创新点-照片EFR.md`）保留为**扁平可绣对照轨**；人像五官可辨的主攻改走本路线。

---

## 1. 为什么转轨

| | EFR 轨（已有） | Photo Flash 类（本路线） |
|--|----------------|--------------------------|
| 中间表示 | 少色硬边分区 | 色调 / 亮度场 + 针密 |
| 成绣手段 | 大面 tatami + 事后 features | 多线色短针 / 半调疏密 |
| 五官 | 暗部叠加，易糊成色块 | 暗部自然落到深色高密针 |
| 开题角色 | 证明「相对 naive 可测」 | 追「照片可辨」的开源可复现基线 |

结论：继续拧 EFR 参数**逼近不了** Wilcom 截图观感；那是路线错配，不是参数不够。

---

## 2. 主张 / 非主张

**主张**

- 照片人像（优先脸部 ROI）可用 **Photo-Stitch**：有限针位（约 8–12）+ 密度/半调映射 → DST，相对 EFR/naive 在「可识别五官」上可人评、可复现对照。
- Wilcom 作**上限参照与交互隐喻**（分辨率档、针位数、左右预览），不宣称功能对等或质量反超。

**非主张**

- 不声称已达到 / 将短期达到 Wilcom Ultimate 写真级。
- 不训练 MSEmbGAN；不把 chibi 的 `similarity≥0.95` 套到照片绣上当验收。
- 第一期不强制集成 `embroidery-streamlines` 方向场（可列为二期可选项）。

---

## 3. 目标管线（工程草图）

```text
Photo
  → rembg（可选）
  → face_roi（优先；衣帽可仍走 segment/EFR）
  → tone_quantize（N≈8–12 线色）
  → density_map（暗→密、亮→疏）
  → short_run / stipple stitches
  → DST + preview + photo_stitch_meta.json
```

**并行保留**

```text
插画 chibi：现有 stitch-auto（默认无 EFR）→ 基线 DoD 不变
EFR 对照：stitch-auto --efr → exports/photo_exp/ 继续可跑
```

| 项 | 路径 |
|----|------|
| 模块 | `embroideryforge/photo_stitch.py` |
| CLI | `stitch-photo`（不劫持 `stitch-auto`） |
| 导出 | `exports/photo_stitch_exp/` |
| 对照 | EFR：`exports/photo_exp/`；人评表见 §5 |

---

## 4. 分阶段

### P0 — 路线与对照基准（本文）

- [x] 锁定 Photo-Stitch 为主攻；EFR 降为对照
- [x] 冻结对照样例：`golden-portrait.jpg`（主）
- [x] 开题三句改写（见 §7）

### P1 — 最小可绣原型（已接通）

- [x] 模块 `embroideryforge/photo_stitch.py`；CLI `stitch-photo`
- [x] 人脸/半身 ROI → Lab 少色量化 → 短填针 DST
- [x] 产物：`auto.dst`、`tone_preview.png`、`density_preview.png`、`compare_preview.png`、`photo_stitch_meta.json`
- 验收：DST 可开；`compare_preview` 三列（ROI | 色调 | 针迹）；人评对照 EFR（见 §5）

```powershell
cd "D:\WORK\TEST\Project 1"
.\.venv\Scripts\Activate.ps1
python -m embroideryforge stitch-photo samples\golden-portrait.jpg --out exports\photo_stitch_exp\portrait_p1 --colors 9 --resolution high --bg auto
# 对照 EFR
python -m embroideryforge stitch-auto samples\golden-portrait.jpg --out exports\photo_exp\portrait_efr --domain portrait --efr --quality dense --bg auto
```

### P2 — 彩色照片绣形态（已接通）

- [x] `--mode hybrid`：脸 Photo-Stitch + 衣帽/发 tatami + outline
- [x] 分辨率档 low/mid/high；色数 6–12
- [x] Streamlit：管线切换 + 分辨率/针位/模式（Wilcom 对话框隐喻）
- [x] 产物增补 `layout_preview.png`（橙=脸，绿=衣帽）

```powershell
python -m embroideryforge stitch-photo samples\golden-portrait.jpg --out exports\photo_stitch_exp\portrait_p2 --mode hybrid --colors 9 --resolution high --bg auto
```

### P3 — 可选增强（明确为可选）

- 局部方向场 / streamlines
- underlay、pull compensation
- 与 Wilcom 导出针数同量级的压力测试（只记录差距，不作 KPI 硬门槛）

---

## 5. 评估怎么写

| 维度 | 做法 |
|------|------|
| 客观 | 针数、换色数、是否含 density 产物、ROI 框 |
| 人评 | 可识别 / 不惊悚 1–5；同图三列：naive \| EFR \| Photo-Stitch |
| 禁止 | 用插画 rubric 的 similarity 冒充照片绣已解决 |

人评表（可直接复制）：

| 评分人 | 样例 | naive | EFR | Photo-Stitch | 备注 |
|--------|------|-------|-----|--------------|------|
| | 可识别 1–5 | | | | |
| | 不惊悚 1–5 | | | | |

---

## 6. 相关工作（叙事用）

- **Wilcom Photo Flash / 彩色照片绣**：工业上限与 UI 隐喻（分辨率、针位列表、双预览）。
- **Embird Sfumato 等**：同类「照片→疏密针」闭源参照。
- **Ink/Stitch**：矢量填缝为主，不是 Photo Flash。
- **embroidery-streamlines**：方向场可研，二期。
- **本仓库 EFR**：证明扁平轨边界，衬托本路线必要性。

---

## 7. 开题三句（转轨版）

1. **真问题**：工业照片绣靠闭源密度算法；开源扁平量化直填或 EFR 大面，人像五官系统性不可辨。  
2. **假设**：有限针位 + 色调密度 Photo-Stitch，在脸部 ROI 上相对 EFR/naive 可复现、可人评增益。  
3. **策略**：插画链保交付；EFR 作对照；Photo-Stitch 作照片域主创新，Wilcom 只作上限参照不写已追平。

---

## 8. 决策记录

| 日期 | 决策 |
|------|------|
| 2026-11-05 | 确认以 Wilcom 彩色照片绣为观感对标；主路线从 EFR 转为 Photo-Stitch；先文档后代码。 |

### 观感迭代 v2 → v2c

- [x] v2：Bayer + `face_dark`（`portrait_p2b`）——门控过猛 + 挖空色调 → 脸信息变少
- [x] v2c：仅疏高光；`face_dark` **叠加**不挖空；脸窗回扩（`portrait_p2c`）

```powershell
python -m embroideryforge stitch-photo samples\golden-portrait.jpg --out exports\photo_stitch_exp\portrait_p2c --mode hybrid --colors 9 --resolution high --bg auto
# 对照：portrait_p2（无门控） / portrait_p2b（过疏） / portrait_p2c（修正）
```

### P3 — 放大观感（分块方向 + 边界）

- [x] 分块梯度主方向短填（`block_oriented_polylines`）
- [x] `clean_layer_mask` + 碎段过滤
- [x] `face_dark` 同步定向

```powershell
python -m embroideryforge stitch-photo samples\golden-portrait.jpg --out exports\photo_stitch_exp\portrait_p3 --mode hybrid --colors 9 --resolution high --bg auto
# 对照缩略/放大：portrait_p2c vs portrait_p3
```

### P3b — underlay + 五官核心加密

- [x] `face_underlay` 极疏肤色底
- [x] 绣序：底层 → 色调 → `face_dark`（含 erode 核心再密一层）
- [x] 引擎 `photo_stitch_hybrid_v3b`

```powershell
python -m embroideryforge stitch-photo samples\golden-portrait.jpg --out exports\photo_stitch_exp\portrait_p3b --mode hybrid --colors 9 --resolution high --bg auto
```

### v3d — 假厚预览 + 少色实填

- [x] `render_clean_preview(thick=True)`：主预览假厚；`auto_stitch_thin.png` 验针
- [x] `--poster`：**软实填** 6–8 色 + 轻闭运算 + 假厚封顶（避免 5 色马赛克）
- [x] 样例：`exports/photo_stitch_exp/meme_p3e/`

```powershell
python -m embroideryforge stitch-photo samples\test_meme_portrait.jpg --out exports\photo_stitch_exp\meme_p3e --mode hybrid --poster --colors 7 --resolution high --bg auto
```

### V4 — 密度场短针（破局主路线）

色块 tatami 已到瓶颈；v4 改 **亮度→针密 + Bayer 落点**，脸 ROI 密度场，衣帽仍可 tatami。

- [x] 模块 `embroideryforge/density_field.py`
- [x] CLI：`--density-field`
- [x] 双轨：`plan_preview.png`（主缩略）/ `auto_stitch_*.png`（验针）
- [x] 四列对照：`compare_preview.png` = 原图 | 密度 | 计划 | 针迹

```powershell
python -m embroideryforge stitch-photo samples\golden-portrait.jpg --out exports\photo_stitch_exp\portrait_v4 --mode hybrid --density-field --colors 10 --resolution high --bg auto
python -m embroideryforge stitch-photo samples\test_meme_portrait.jpg --out exports\photo_stitch_exp\meme_v4 --mode hybrid --density-field --colors 10 --resolution high --bg auto
```

### V4.1 — 蛇形串针 + 五官核

在 v4 密度场上减 JUMP、加五官细节：

- [x] `points_to_serpentine_runs`：同行串段 + 隔行反向衔接
- [x] `feature_core_mask`：眼/眉/嘴暗核二次更密 step（`feature_core.png`）
- [x] **v4.1.1**：只并邻接点（禁止跨空拉线），脸 `step×0.85` + 中调/核区抬密，避免比 v4 更糊
- [x] engine：`photo_stitch_density_v4_1`

```powershell
python -m embroideryforge stitch-photo samples\golden-portrait.jpg --out exports\photo_stitch_exp\portrait_v41 --mode hybrid --density-field --colors 10 --resolution high --bg auto
python -m embroideryforge stitch-photo samples\test_meme_portrait.jpg --out exports\photo_stitch_exp\meme_v41 --mode hybrid --density-field --colors 10 --resolution high --bg auto
```
