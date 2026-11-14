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
| `test_animal_01.jpg` … `09` | **动物 A0** 实拍（见下表） |

### 动物样例对照（Downloads/Animal → samples）

| samples | 原 hash 前缀 | 语义（粗标） |
|---------|--------------|--------------|
| `test_animal_01.jpg` | `051895776f30…` | 户外三狗 + 绿叶 |
| `test_animal_02.jpg` | `0717c7689f55…` | 室内双橘猫 |
| `test_animal_03.jpg` | `424d6c7ead7f…` | 户外兔+猫+狗 |
| `test_animal_04.jpg` | `529e3fe504ce…` | 室内狗群+鹿 |
| `test_animal_05.jpg` | `74318134465a…` | 室内鹿狗贴鼻 |
| `test_animal_06.jpg` | `8cbf23ba9c6a…` | 雪地橘猫+兔 |
| `test_animal_07.jpg` | `90393a4fe4f4…` | 室内狗+幼鹿互动 |
| `test_animal_08.jpg` | `cd115b2766bf…` | 室内成犬+鹿+幼犬群 |
| `test_animal_09.jpg` | `e67a2e0d6f5d…` | 户外斑点鹿+犬 |

冻结路径与复现命令见 [`docs/BASELINE.md`](../docs/BASELINE.md)。  
照片绣（人像）：[`docs/路线-照片绣PhotoStitch.md`](../docs/路线-照片绣PhotoStitch.md)。  
动物照片绣：[`docs/路线-动物PhotoStitch.md`](../docs/路线-动物PhotoStitch.md)。

```powershell
# 课题一
python -m embroideryforge stitch-auto samples\golden-chibi.jpg --out exports\smoke --llm --quality dense --bg auto

# 照片绣场景分轨（统一参数）
python -m embroideryforge stitch-photo samples\test_figure_cowboy.jpg --out exports\photo_stitch_exp\figure_cowboy_scene_vN --mode hybrid --density-unified --bg keep --colors 12 --resolution high
```
