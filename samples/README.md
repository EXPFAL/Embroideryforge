# 样例

| 文件 | 说明 |
|------|------|
| `golden-logo.jpg` | 初代（企鹅），历史对照 |
| `golden-chibi.jpg` | **插画主测图**（见 `docs/BASELINE.md`） |
| `golden-portrait.jpg` | **人像**（黄帽黄衣，复杂背景） |
| `golden-landscape.jpg` | **风景**（合成可复现样例，照片域实验） |
| `test_portrait.jpg` | 侧脸压力测（早期） |

照片域：EFR 对照见 `docs/创新点-照片EFR.md`；主路线 Photo-Stitch 见 `docs/路线-照片绣PhotoStitch.md`。

```powershell
python -m embroideryforge stitch-photo samples\golden-portrait.jpg --out exports\photo_stitch_exp\portrait_p1 --colors 9 --resolution high --bg auto
```
