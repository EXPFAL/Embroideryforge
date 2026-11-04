# EmbroideryForge

AI 辅助机器刺绣：图片 → 去背景/分区 → `stitch_plan`（规则或 DeepSeek）→ 多色 DST → 色对齐预览 → 课题五维评估。

仓库：https://github.com/EXPFAL/Embroideryforge

## 笔记本 / 线下开荒（必读）

台式机开发、笔记本实践时，请按：

**[docs/开荒教程-笔记本线下实践.md](docs/开荒教程-笔记本线下实践.md)**

内含：必装/可选软件与下载链接、克隆、venv、`.env`、pytest、白底/复杂背景样例、迁移检查表与现场排障。

## 快速开始（本机已有 Python）

```powershell
git clone https://github.com/EXPFAL/Embroideryforge.git
cd Embroideryforge
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
python -m pytest -q
python -m embroideryforge stitch-auto samples\golden-chibi.jpg --out exports\smoke --llm --quality dense
```

## 结构

```
app/                 Streamlit（可选 GUI）
embroideryforge/     核心：segment / plan / llm / autofill / bg / evaluate
samples/             黄金与测试样例
tests/               单元与冒烟测试
docs/                产品说明、设计说明、开荒教程
exports/             运行输出（默认 gitignore）
```

## 课题对齐

见 `docs/课题一设计说明-chibi基线.md`、`docs/PRODUCT.md`。  
评估：`python -m embroideryforge evaluate <导出目录> --image <原图>` → `rubric_pass_95`。

## 许可与密钥

- 代码以本仓库为准；`.env` 不入库  
- DeepSeek Key 仅本地配置，参见 `.env.example`
