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

### 脸窗识别 — MediaPipe Face Landmarker

Haar 框常切在嘴下导致下颚进 body。现优先 **MediaPipe** 关键点定 ROI/凸包脸罩（[`embroideryforge/face_detect.py`](../embroideryforge/face_detect.py)），失败回退 Haar + `refine_face_roi_to_chin`。模型见 `models/README.md`。

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

### V4.2 — 高光保点 + 五官结构针（已回退）

试过在 v4.1.1 上叠结构 satin/running + 高光抠空，**面部观感明显差于 v41**（眼窝易涂黑、层次被破坏）。代码留作实验函数，**默认管线不启用**；对照产物可保留 `portrait_v42` / `meme_v42` 作反例。

当前推荐仍用 v4.1.1：

```powershell
python -m embroideryforge stitch-photo samples\golden-portrait.jpg --out exports\photo_stitch_exp\portrait_v41 --mode hybrid --density-field --colors 10 --resolution high --bg auto
```

### P4 / V5 — 人物 hybrid + 叠针（已失败 / 回退）

试过：脸密度 + 衣 tatami 多带 + `clothes_fold` / 指骨 / 道具轮廓 running。  
**结果**：服装出现怪异飞丝，手/枪仍糊。`figure_uniform_v5` 可留作反例。叠针默认已关。

### P4 / V6 — 整主体统一 Color PhotoStitch（当前人物+服饰主路线）

对齐 Wilcom Color PhotoStitch：**整 `subject_mask` 同一套多色密度蛇形短针**，不再 body tatami、不再 fold/指骨/道具轮廓叠针。

| 项 | 策略 |
|----|------|
| 密度罩 | `dens_mask = subject_mask` |
| 脸/手 | MediaPipe 仅作软权重（局部更密）；不发射结构层 |
| 衣褶/枪形 | 靠亮度→针密自然形成 |
| 引擎 | `photo_stitch_density_v6` |
| 近脸图 | 仍默认 `photo_stitch_density_v4_1` |

```powershell
# 强制 V6
python -m embroideryforge stitch-photo samples\test_figure_uniform.jpg --out exports\photo_stitch_exp\figure_uniform_v6 --mode hybrid --density-unified --colors 11 --resolution high --bg auto
# 半身自动 V6（--density-field 且 face_frac<0.45）
python -m embroideryforge stitch-photo samples\test_figure_uniform.jpg --out exports\photo_stitch_exp\figure_uniform_v6 --mode hybrid --density-field --colors 11 --resolution high --bg auto
# 强制留在 v4.1.1
python -m embroideryforge stitch-photo samples\golden-portrait.jpg --out exports\photo_stitch_exp\portrait_v41 --mode hybrid --density-field --no-density-unified --colors 10 --resolution high --bg auto
```

**非主张**：不引入 SegFormer/Torch；不把 nuizu/OpenStitch 嵌进主链。

### P4 / V6.1 — 人物主体 + 场景（浅做）

纯风景不单开路线。人物+环境用 `--bg keep`（或 `scene`）：

- 工作图保留全帧；`subject_mask`（rembg）仅作**主体加权**
- 主体：密密度 / 细 step；背景疏针当氛围
- 引擎仍 `photo_stitch_density_v6`，`mode=density_scene`

```powershell
python -m embroideryforge stitch-photo samples\test_figure_cowboy.jpg --out exports\photo_stitch_exp\figure_cowboy_scene_v6 --mode hybrid --density-unified --bg keep --colors 11 --resolution high
```

### P4 / V6.2 — 主体偏置调色 + 场景疏密校正 + 脸手微密

针对 V6.1 牛仔样例短板（色阶塌成米灰、背景缝隙过大、bust 脸罩稀释五官）：

| 项 | 策略 |
|----|------|
| 调色 | Lab k-means **主体偏置采样**（person + face + hand 加权）；场景色数上限 **16** |
| 背景疏密 | `bg_dens_scale=0.58`、`bg_step_scale=1.40`、`floor_bg=0.07`（缓和缝隙，仍疏于主体） |
| 脸/手 | MediaPipe 优先；bust 去帽冠+肤色收紧；脸/手 **Bayer 全分辨率**留针（绕过 step≥1 网格）；Canny 五官核；手 dens×1.38 + 膨胀 |
| 非主张 | 仍不嵌 SegFormer/Torch、nuizu/OpenStitch；**不**恢复 V5 衣褶/骨骼叠层 |

```powershell
python -m embroideryforge stitch-photo samples\test_figure_cowboy.jpg --out exports\photo_stitch_exp\figure_cowboy_scene_v62b --mode hybrid --density-unified --bg keep --colors 12 --resolution high
```

### P4 / V6.3 — 人物与背景分轨再整合

脸糊成一片：整帧共用一套 Lab 色心 + 蛇形扫描。`--bg keep|scene` 时改为双轨：

| 轨 | 内容 |
|----|------|
| BG | `dens_mask & ~erode(person)`；独立量化；**禁止 bilateral**；轻锐化 + Canny 边 dens；`step×1.22`；tag `dens_bg_*` |
| FG | `dilate(person)`；整体 unsharp + **脸区 CLAHE/再锐化**；脸/手加权量化；全分辨率留针；tag `dens_fg_*` |

针序：先 BG 后 FG（同画布坐标）。`mode=density_split`。非场景仍单轨 `density_unified`。

```powershell
python -m embroideryforge stitch-photo samples\test_figure_cowboy.jpg --out exports\photo_stitch_exp\figure_cowboy_scene_v63b --mode hybrid --density-unified --bg keep --colors 12 --resolution high
```

**V6.3c**：`max_side=720` + 第三轨 `dens_face_*`（仅脸窗，更细 step，后绣压在 FG 上）。

**V6.3d**：脸密轨**复用 FG 的 label/centers**（不再二次 k-means），避免头身两截色；脸区 CLAHE 减弱并与原体混合。

**V6.4（人物细节）**：去帽冠后 MediaPipe 重试；脸罩肤色收紧；脸内 Laplacian dens 抬；`dens_face` step×0.72 + 肤色脸窗；新增 `dens_hand_*`（同色复用）。人评：相对 v63d 几乎无感（脸区 MAE≈2.5）。

**V6.4b**：FG 挖空脸/手 → 密轨独占；实心脸罩；强 CLAHE + 同色 Lab 重映射。人评：细节升，但面部偏白。

**V6.4c**：保留挖空/重映射；CLAHE/锐化回调 + 高光压暗，减轻漂白。人评：仍略白于 v64。

**V6.4d**：轻 CLAHE；**重映射仅结构区**，大面肤色沿用 FG label；挖空细节保留。

**V6.4e（背景色准）**：BG 色数 12–16；Lab 色度加权采样（红联/金瓦占色槽）；禁高光软化 + centers 色度回拉。

**V6.4f**：FG 冷色衣物加权 + centers 平衡（保青绿、压侧脸偏红）；FG 色数同步放开。人评：仍偏灰绿/侧脸红，加权不够。

**V6.5（色相分桶）**：HSV 分 warm/cool/neutral + vivid_warm；桶内 Lab k-means。人评：色准升，但硬分桶+冷色强留针致不规则斑块。

**V6.5b**：同上 + 标签 5×5 中值。人评：糊成油画，过平滑。

**V6.5c**：分桶出色心 → 全图 Lab 重绑；**不做标签中值**；cool G>R+开运算；无冷色强留针。贴图用原像素 Lab。

**V6.6（压平色板）**：过量化 → 小面积并入 → 面积≥5%/高色度保底 → 合并到 n 色；色度加权 Lab 贴色。不增依赖；无中值、无冷色强留针。人评：色准好。

**V6.6b–d（细节加压）**：多脸 dens_face、结构 dens、max_side 840。人评相对 v66 无感；**冻结仍用 v66**，代码保留多脸并集，`max_side` 回 720。

```powershell
python -m embroideryforge stitch-photo samples\test_figure_kobe_multi.jpg --out exports\photo_stitch_exp\figure_kobe_multi_v66 --mode hybrid --density-unified --bg keep --colors 12 --resolution high
```

---

## 附录 A — 对照协议与人评（V6.3d 起）

### A.1 固定命令

```powershell
python -m embroideryforge stitch-photo <样例.jpg> --out exports\photo_stitch_exp\<名>_vN --mode hybrid --density-unified --bg keep --colors 12 --resolution high
```

- 新实验只升 `vN` / `vNb`，**勿覆盖** [`docs/BASELINE.md`](BASELINE.md) 中的冻结目录  
- 必交件：`compare_preview.png`、`auto_stitch_thick.png`、`photo_stitch_meta.json`（查 `mode=density_split`、`dens_face_*`）

### A.2 难例对照表（2026-11-08 · 统一 V6.6）

| 样例 | 路径 | 脸检测 | 已知难点 | 冻结/最新导出 | 针数 |
|------|------|--------|----------|----------------|------|
| 牛仔+街景 | `samples/test_figure_cowboy.jpg` | mediapipe | 街景糊、帽檐、持枪手 | 基线 `v63d`；最新 `figure_cowboy_scene_v66` | 313241 |
| 合成脸武侠 | `samples/test_figure_kobe_wuxia.jpg` | mediapipe | 脸/颈色差、虚化树叶 | 基线 `v63d`；最新 `figure_kobe_wuxia_v66` | 372833 |
| Homelander 舱内 | `samples/test_figure_kobe_homelander.jpg` | mediapipe | 浅发高光脸、冷色舱 | 最新 `figure_kobe_homelander_v66` | 321028 |
| 皇庭指向 | `samples/test_figure_kobe_emperor.jpg` | mediapipe | 高帽、红墙金瓦、举手 | 最新 `figure_kobe_emperor_v66` | 346569 |
| 多人院落 | `samples/test_figure_kobe_multi.jpg` | mediapipe+Haar | 青绿衣；色准主评 | 最新 `figure_kobe_multi_v66` | 371141 |
| 半身制服 | `samples/test_figure_uniform.jpg` | 视构图 | 抠底半身；`--bg auto` | `figure_uniform_v6` | — |

### A.3 人评清单（各 1–5 分，≥3 可演示）

| 项 | 看什么 |
|----|--------|
| 五官可辨 | 眼/嘴是否成块糊死 |
| 头身同色 | 脸与衣领是否两套色阶（V6.3d 应明显好于 3c） |
| 背景可读 | 建筑/树叶是否「一团」；边缘是否可辨 |
| 缝隙/接缝 | 人物轮廓是否露底或硬边光晕 |
| 总针数合理 | meta 约 15–40 万（场景）；异常飙升需记原因 |

### A.4 V6.6 演示集人评（2026-11-08）

评分依据：用户对 multi「色准好」的确认 + 各样例 `auto_stitch_thick` / meta（未再跑 chibi rubric）。≥3 视为可演示。

| 样例 | 五官 | 头身同色 | 背景 | 缝隙 | 针数 | 均分 | 备注 |
|------|------|----------|------|------|------|------|------|
| cowboy_v66 | 3 | 4 | 3 | 4 | 5 | 3.8 | 街景仍偏糊；针数正常 |
| wuxia_v66 | 3 | 4 | 3 | 4 | 5 | 3.8 | 树叶虚化仍难读 |
| homelander_v66 | 3 | 4 | 4 | 4 | 5 | 4.0 | 冷舱色阶尚可；脸高光易平 |
| emperor_v66 | 3 | 4 | 4 | 4 | 5 | 4.0 | 红联/金瓦靠压平色板 |
| multi_v66 | 3 | 4 | 4 | 4 | 5 | 4.0 | **主演示**；色准人评通过 |

人评分记在实验笔记或本附录；**不**跑 chibi `rubric_pass_95`。
