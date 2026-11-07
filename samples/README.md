# 样例

| 文件 | 说明 |
|------|------|
| `golden-chibi.jpg` | **课题一主测**（扁平插画） |
| `golden-portrait.jpg` | 人像 / EFR·portrait 域 |
| `golden-landscape.jpg` | 风景（不单开照片绣主轨） |
| `golden-logo.jpg` | 初代企鹅，历史对照 |
| `test_figure_uniform.jpg` | 半身制服（V6 抠底） |
| `test_figure_cowboy.jpg` | **人物+场景**（宽檐帽难例，`--bg keep`） |
| `test_figure_kobe_wuxia.jpg` | **合成脸**难例（MediaPipe 常命中） |
| `test_chibi_complex_bg.jpg` | 复杂背景插画（rembg） |
| `test_portrait.jpg` | 侧脸压力测（早期） |

冻结路径与复现命令见 [`docs/BASELINE.md`](../docs/BASELINE.md)。  
照片绣路线：[`docs/路线-照片绣PhotoStitch.md`](../docs/路线-照片绣PhotoStitch.md)。

```powershell
# 课题一
python -m embroideryforge stitch-auto samples\golden-chibi.jpg --out exports\smoke --llm --quality dense --bg auto

# 照片绣场景分轨（统一参数）
python -m embroideryforge stitch-photo samples\test_figure_cowboy.jpg --out exports\photo_stitch_exp\figure_cowboy_scene_vN --mode hybrid --density-unified --bg keep --colors 12 --resolution high
```
