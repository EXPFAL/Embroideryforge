# EmbroideryForge

AI 辅助机器刺绣：图片 → 分区 / 密度场 → 多色 **DST**，带 CLI、可选 DeepSeek 针迹规划与自动评估。

**两条主路线**

| 路线 | 命令 | 适合 |
|------|------|------|
| 插画 / 色块自动绣 | `stitch-auto` | Q 版、扁平插画、logo |
| 照片绣 Photo-Stitch | `stitch-photo` | 人像、半身、人物+场景（V6 密度场） |

仓库：https://github.com/EXPFAL/Embroideryforge

---

## 快速开始

```powershell
git clone https://github.com/EXPFAL/Embroideryforge.git
cd Embroideryforge
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
python -m pytest -q
```

可选：在 `.env` 中配置 `DEEPSEEK_API_KEY`（仅本地，勿提交）。

### 插画自动绣

```powershell
python -m embroideryforge stitch-auto samples\golden-chibi.jpg --out exports\smoke --llm --quality dense
```

### 照片绣（人物 / 场景）

```powershell
# 半身统一密度（抠底）
python -m embroideryforge stitch-photo samples\test_figure_uniform.jpg --out exports\photo_uniform --mode hybrid --density-unified --bg auto --colors 11 --resolution high

# 人物+场景分轨（保留背景，V6.3）
python -m embroideryforge stitch-photo samples\test_figure_cowboy.jpg --out exports\photo_cowboy --mode hybrid --density-unified --bg keep --colors 12 --resolution high
```

主预览看导出目录中的 `auto_stitch_thick.png`（黑底厚针）与 `compare_preview.png`。

### 可选 GUI

```powershell
python -m streamlit run app\streamlit_app.py
```

---

## 功能概览

- **DST 输出**：`pyembroidery` 写入 Tajima DST 与预览 PNG  
- **去背景**：`rembg` / GrabCut / 白底跳过；场景模式 `--bg keep` 保留全帧  
- **照片绣 V6**：Bayer 密度短针 + 蛇形扫描；人物/背景分轨再整合；脸密轨复用人物色阶  
- **MediaPipe**：脸 / 手几何（可用时）作局部加密，不强制结构叠层  
- **评估**：`evaluate` 五维打分（课题对齐）  
- **向量化**：`vectorize` 辅助路径（vtracer）

---

## 仓库结构

```
embroideryforge/   核心库与 CLI（segment / plan / density_field / bg …）
app/               Streamlit 可选界面
samples/           黄金样例与测试图
tests/             pytest
docs/              产品说明、照片绣路线、开荒教程
exports/           运行输出（默认 gitignore）
models/            MediaPipe 等模型说明（大文件不入库）
```

---

## 文档

| 文档 | 内容 |
|------|------|
| [开荒教程-笔记本线下实践](docs/开荒教程-笔记本线下实践.md) | 克隆、venv、样例、现场排障 |
| [路线-照片绣 PhotoStitch](docs/路线-照片绣PhotoStitch.md) | 照片绣版本演进（至 V6.3） |
| [PRODUCT](docs/PRODUCT.md) | 产品说明 |
| [课题一设计说明](docs/课题一设计说明-chibi基线.md) | chibi 基线与评估 |

评估示例：

```powershell
python -m embroideryforge evaluate exports\smoke --image samples\golden-chibi.jpg
```

---

## 环境要求

- Python 3.10+（推荐 3.11 / 3.12；MediaPipe 需可用的本地 wheels）  
- Windows 下建议 PowerShell + `py -3 -m venv`  
- 依赖见 [`requirements.txt`](requirements.txt)

---

## 许可与密钥

- 源码以本仓库为准；当前未附加开源许可证文件  
- `.env` / API Key **不入库**；模板见 [`.env.example`](.env.example)
