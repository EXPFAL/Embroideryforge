# 路线：动物照片绣（复用 Photo-Stitch V6.6）

**状态**：B0 仍是动物演示。方案 C 叠色先测两张，**人评未过**（2026-11-15）：02 与 B0 区别不大且眼睛更偏；01 中间狗毛色不再发灰，但整张观感下降。不铺到其余图，不加第三层。人像默认仍是 V6.6。  
**上游**：人像轨保持 V6.6，见 [`路线-照片绣PhotoStitch.md`](路线-照片绣PhotoStitch.md)，**不改**。  
**成熟方案调研**：[`调研-动物照片绣成熟方案.md`](调研-动物照片绣成熟方案.md)。  
**前沿文献**：[`调研-前沿文献-照片绣.md`](调研-前沿文献-照片绣.md)（流线只作否决记录）。  
**样例**：`samples/test_animal_01.jpg` … `09`。

---

## 主张 / 非主张

| 主张 | 非主张 |
|------|--------|
| 复用 V6.6 的 A1 短针，先把构图收成一只动物 | 不追毛发丝理；流线已人评否决 |
| 人评只问：厚预览还是不是这只动物 | 不靠文献指标或 Vision Judge 分数过关 |
| `--domain animal` 跳过 dens_face | 不新开宠物检测模型 |
| A1：关 FG 暖肤压色；FG 结构 dens；BG 保绿 | 不改人像默认 `dampen_warm_skin` |
| 人评：物种可辨、毛色、绿叶、毛发轮廓 | 不套 chibi `rubric_pass_95` |
| `--bg keep` 保全帧（多宠不被 rembg 切碎） | 不提交 exports 二进制 |

---

## 优化方案 B（2026-11-15）

A1 是全帧多宠，针预算摊在屋子、垫子和绿叶上。A2 改去追边缘方向，thick 变成木纹。下一刀不换针法，只改「绣什么」。

**过关标准**：`auto_stitch_thick.png` 对着裁切后的 `subject.png`，第一眼仍是那一只动物。不过关就停，不开新引擎。

### B0 — 单动物构图（下一刀）

针法仍是 A1（`--domain animal`，不要 `--streamlines-patch`）。

1. 每张图只留一只动物。罩用矩形或手绘，禁止再用「前景最大连通域」。
2. 第一张：`test_animal_02.jpg` 前景那只趴着的橘猫。另一只猫、紫垫、墙都不绣。
3. 导出 `exports/photo_stitch_exp/animal_02_b0/`，与 `animal_02_a1` 的 thick 并排。
4. 人看三句：是不是这只猫、橘毛还在不在、眼睛还是不是眼睛。

**2026-11-15 已导出**（裁切从粗，不再修罩）：`exports/photo_stitch_exp/animal_02_b0/`，约 31.6 万针。并排 `compare_vs_a1_thick.png`（左 A1 全帧，右 B0）。针法仍是 A1，`--bg keep`。

### B1 — 仅当 B0 像猫但发灰

仍在同一罩、同一套 A1 里：

- 线色收到 8–10，沿用动物色板，不贴人像肤色表。
- 眼部高光留空，不在眼睛上再加密。

### 停

- B0 单猫 thick 仍不像照片：这条已经过关，不再加裁切。
- 不重跑全帧拧参。
- 不复活流线、行扫。人像默认命令仍是 V6.6。

---

## 方案 C — 商用叠色（2026-11-15）

B0 人评：01 狗舌发黑、左侧两狗毛色发灰；03、08 皮毛涂抹感重；06 毛色偏淡；09 皮毛细节不足。02 三项过关，观感是 A1 放大。这些都出在 `quantize_tones_flatten`：每个位置只留一种线色，暗红并进黑色，棕色并进灰色，大面短针连成色块。

商用 Color PhotoStitch / Brother Photo Stitch 1 的照片模式用两种线叠出中间调。下一版走这条，不上流线，不改短针走向。

1. 色板仍 8–12 色，动物域，不贴人像肤色表。
2. 每个像素保留最近两色和混合比。
3. 第一层绣主色，沿用现在的短针。
4. 第二层只绣混合比落在中间的像素，密度约主色的四成。对应 PS1 的 Detail，先固定这个比例。
5. 开关 `--photo-blend`。默认命令不变，人像 V6.6 不自动打开。

第一张用 `animal_02_b0_input/crop.png`。过关后再跑 01、03、06、08、09 的 B0 裁切。

**2026-11-15 先测两张**（`--photo-blend`）：`animal_02_c0`（46.5 万针）、`animal_01_c0`（91.6 万针）。

**人评**：02 与 B0 区别不大，眼睛颜色更偏。01 中间狗毛色不再发灰，整张观感下降。叠色层盖住了已经对的地方，眼睛也被次色改掉。停在 B0，不跑 03/06/08/09，不加第三层。`--photo-blend` 只留作这次记录。

过关看这五句：01 舌头不是纯黑、左侧狗毛能看出棕色；03 和 08 的毛能看出两色颗粒；06 橘毛不发白；09 斑点边比 B0 多一层深色；02 仍是那只猫。

叠完若仍是 B0 那张放大图，动物路线停在 B0，不再加第三层。

---

## A1 统一命令（当前演示）

```powershell
python -m embroideryforge stitch-photo samples\test_animal_04.jpg --out exports\photo_stitch_exp\animal_04_a1 --mode hybrid --density-unified --bg keep --colors 12 --resolution high --domain animal
```

- meta：`domain=animal`、`v6.v66_animal_a1=true`、`animal_no_skin_dampen`、`animal_fg_structure_dens`、layers **无** `dens_face_*`
- 必看：`compare_preview.png`、`auto_stitch_thick.png`、`photo_stitch_meta.json`

### 演示优先（3）

| 导出 | 样例 | 语义 | 看点 |
|------|------|------|------|
| `animal_04_a1` | `test_animal_04.jpg` | 室内狗群+鹿 | 主演示；毛色/轮廓 |
| `animal_02_a1` | `test_animal_02.jpg` | 室内双橘猫 | **毛色**（勿发灰） |
| `animal_01_a1` | `test_animal_01.jpg` | 户外三狗 | **绿叶** + 毛发轮廓 |

### A1 针数（2026-11-14）

| 导出 | stitches | 备注 |
|------|----------|------|
| animal_04_a1 | 358133 | 主演示；相对 A0 持平 |
| animal_02_a1 | 309873 | 橘猫色回归 |
| animal_01_a1 | 421205 | 绿叶大面积；略超 40 万 |

A0 全套 `animal_*_a0` 仍留本地作对照（针数见下节）。

---

## A0 对照（历史）

同命令，输出目录 `animal_*_a0`；meta 为 `v66_animal_a0`（仅跳 dens_face，仍压肤）。

| 导出 | stitches |
|------|----------|
| animal_01_a0 | 414844 |
| animal_02_a0 | 311089 |
| animal_03_a0 | 382095 |
| animal_04_a0 | 357378 |
| animal_05–09_a0 | 见本地 meta |

---

## 人评清单（各 1–5，≥3 可演示）

| 项 | 看什么 | A1 主看样例 |
|----|--------|-------------|
| 物种可辨 | 狗/猫/兔/鹿是否一眼能分 | 04 |
| 毛色大致对 | 橘/棕/鹿褐是否发灰、串肤粉 | **02**、04 |
| 毛发轮廓 | 眼鼻暗部与躯干明暗（**不**要求丝理） | 04、01 |
| 绿叶/背景 | 草木是否更贴原图 | **01** |
| 针数合理 | 约 15–45 万；>50 万记原因 | 01 |

---

## 后续（A2）— 人评否决，停

**2026-11-15**：`animal_02_a2_stream/auto_stitch_thick.png` 约 63 万针。结构张量把边缘、阴影、垫子纹理积成连续长曲线，黑缝读成皱纹/木纹；前景最大连通域把两只猫和紫垫缝成一块。优化的是「方向连续」，不是「像这张照片」。

演示命令仍是上面的 **A1**。`--streamlines-patch` 只留作失败记录，不作为下一默认。

见 [`调研-动物照片绣成熟方案.md`](调研-动物照片绣成熟方案.md) §5；吸收笔记 [`调研-前沿文献-照片绣.md`](调研-前沿文献-照片绣.md) §8。

```powershell
python -m embroideryforge stitch-photo samples\test_animal_02.jpg --out exports\photo_stitch_exp\animal_02_a2_stream --streamlines-patch exports\photo_stitch_exp\animal_02_a1\fg_mask.png --colors 8 --resolution high --bg keep --domain animal
# 省略 MASK 路径则用主体最大连通域（--streamlines-patch）
```

| 项 | 说明 |
|----|------|
| 模块 | [`embroideryforge/streamlines_patch.py`](../embroideryforge/streamlines_patch.py) |
| 导出 | `animal_02_a2_stream/`：`streamlines_preview.png`、`direction_field.png`、`auto.dst`、`compare_preview.png` |
| meta | `a2_streamlines=true`、`engine=streamlines_patch_a2` |
| 对照 | `animal_02_a1` thick / compare；人评加「毛发丝理是否可辨」 |
| 默认 dens | **仍走 A1**；流线开关保留，人评已否决 |

**验收记录（2026-11-15）**：thick 像木纹/恐怖片纹理，否决。不加深集成。

- 不做：继续 A1 拧参；整图重写 V6.6；把流线或密度预算当人像/动物的下一默认
