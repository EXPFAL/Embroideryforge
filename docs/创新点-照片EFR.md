# 创新点：照片域绣友好中间表示（EFR）

> **角色更新（2026-11-05）**：EFR 为**扁平可绣对照轨**（已落地）。若对标 Wilcom 彩色照片绣的五官可辨，主路线见 `docs/路线-照片绣PhotoStitch.md`。

## 主张

naive 照片量化直填会系统性「惊悚」。引入 **EFR（少色 + 边缘硬化）** 后再走 `stitch_plan` + 工具链，相对 naive 可复现对照，失败可解释。

## 非主张

不宣称超过 Wilcom / Embird Sfumato / 手工打版师；不训练 MSEmbGAN；不集成方向场库。

## 主路径

```text
Photo → rembg(可选) → EFR → segment(portrait) → plan(+msemb启发式) → DST
```

插画 `golden-chibi`：**默认不开 EFR**，基线 DoD 不变。

**五官**：EFR 只负责大面扁平填充；眉眼口从 **去背原图** 抽暗部叠 `features` 走针（satin/running），避免「只剩模糊色块」。

## 验收命令

```powershell
cd "D:\WORK\TEST\Project 1"
.\.venv\Scripts\Activate.ps1

# naive
python -m embroideryforge stitch-auto samples\golden-portrait.jpg --out exports\photo_exp\portrait_naive --domain portrait --no-efr --llm --quality dense --bg auto

# EFR
python -m embroideryforge stitch-auto samples\golden-portrait.jpg --out exports\photo_exp\portrait_efr --domain portrait --efr --llm --quality dense --bg auto

python -m embroideryforge compare-photo exports\photo_exp\portrait_naive exports\photo_exp\portrait_efr --out exports\photo_exp\compare_portrait.json

# 风景
python -m embroideryforge stitch-auto samples\golden-landscape.jpg --out exports\photo_exp\landscape_efr --domain portrait --efr --llm --quality dense --bg auto

# 插画基线回归
python -m embroideryforge evaluate exports\accept_flat_v3 --image samples\golden-chibi.jpg --ref-dst exports\inkscape_smoke\chibi_multicolor.dst
```

## 人评模板（可选）

| 评分人 | 样例 | 可识别 1-5 | 不惊悚 1-5 | 备注 |
|--------|------|------------|------------|------|
| | naive 预览 | | | |
| | efr 预览 | | | |

## 相关工作

Ink/Stitch、embroidery-magic、embroidery-streamlines、MSEmb：借鉴「扁平稿可自动」边界，不抄写真一键绣。

## 三句复述

1. 工业/开源默认干净少色稿，照片是真难题。  
2. 创新在 EFR 中间表示 + 可检查 plan，不是更大生成模型。  
3. 插画保交付；照片讲相对 naive 的增益与仍存边界。
