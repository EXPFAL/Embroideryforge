# 路线：DeepSeek 识图比对反馈（Vision Judge）

**状态**：已接通（2026-11-14）  
**密钥**：项目根 `.env` 的 `DEEPSEEK_API_KEY`（与 `--llm` 针法计划同钥）  
**模型**：`deepseek-flash`（识图 + 文本）

---

## 主张 / 非主张

| 主张 | 非主张 |
|------|--------|
| **subject.png**（成绣输入）+ 针迹预览 → 结构化 JSON 评分 | 不替代实绣；不替代 chibi `rubric_pass_95` |
| 补人评缺漏（维度 + evidence） | **不**根据分数自动改 dens / 色板 |
| 单目录 + A/B pair | 不上传未缩边的超大 PNG |

---

## 命令

```powershell
# 单导出
python -m embroideryforge judge-photo exports\photo_stitch_exp\figure_portrait_p0_smoke --image samples\test_figure_kobe_multi.jpg --rubric portrait

# A/B（例：v66 vs P0）
python -m embroideryforge judge-photo-pair exports\photo_stitch_exp\figure_kobe_multi_v66 exports\photo_stitch_exp\figure_portrait_p0_smoke --image samples\test_figure_kobe_multi.jpg --rubric portrait

# 动物
python -m embroideryforge judge-photo exports\photo_stitch_exp\animal_02_a1 --image samples\test_animal_02.jpg --rubric animal
```

- 产出：`vision_judge.json` / `vision_judge_pair.json`（写在导出目录）  
- **比对原图**：优先各导出目录的 `subject.png`（成绣实际输入）；缺省再用 `--image`  
- `--rubric auto`：读 `photo_stitch_meta.json` 的 `domain==animal` → animal，否则 portrait  
- 无 Key：JSON 里 `ok=false` + `error=DEEPSEEK_API_KEY not set`

---

## Rubric

- **portrait**：五官、头身同色、背景、缝隙、针数、色准（1–5）；`demo_ok` = 均分≥3 且无维=1  
- **animal**：物种/姿态、毛色、毛发轮廓、绿叶/背景、针数  

每维必有 `evidence` 短句；解析失败不填满分。

---

## 闭环用法

1. `stitch-photo` 导出  
2. `judge-photo` → 读低分维与 evidence  
3. **人工**选下一旋钮（`--stitches-per-mm2` / `--density-contrast` / `--portrait-p0` / `--streamlines-patch`…）  
4. 再导出 → `judge-photo-pair` 看是否提升  

### 验收门（文献吸收后写死）

| 门 | 角色 |
|----|------|
| 人眼 + `compare_preview` / thick | **最终**观感门 |
| Vision Judge（`subject.png` 成对） | 辅评、记 evidence |
| 实绣（有机时） | 布上色准 / 密度；屏幕裁判不能替代 |
| 文献分数 / GAN metrics | **不**作验收 |

动物 A2 对照例：

```powershell
python -m embroideryforge judge-photo-pair exports\photo_stitch_exp\animal_02_a1 exports\photo_stitch_exp\animal_02_a2_stream --image samples\test_animal_02.jpg --rubric animal
```

---

## 局限

- 送模前缩边（默认 max_side=1280）；细节可能被抹平。  
- 裁判看预览图，不是布面实绣。  
- DeepSeek 现也用于课题一文本 `stitch_plan`（`--llm`）；本裁判是**第二条用途**（识图辅评）。  
