# Paper Read

这是一个 Markdown-first 的论文阅读库。每篇论文放在 `papers/<paper-id>/` 下，用 `meta.yaml` 描述元数据，并通过 `tools/build_index.py` 自动生成索引。

## 目录结构

```text
paper_read/
├── papers/                 # 每篇论文一个目录
├── index/                  # 自动生成的索引
├── templates/              # 新论文模板
├── tools/                  # 工具脚本
└── docs/superpowers/plans/ # 实施计划文档
```

## 新增论文流程

1. 在 `papers/` 下创建新目录，例如 `papers/2026-06-paper-slug/`。
2. 复制 `templates/meta.yaml` 到新目录并填写。
3. 复制 `templates/paper-README.md` 到新目录，填写论文标题和 URL。
4. 放入本地 `paper.pdf`，PDF 会被 `.gitignore` 忽略，不上传 Git。
5. 将抽取文本、翻译、摘要、阅读笔记分别放入 `source/` 和 `notes/`。
6. 运行：

```bash
python3 tools/build_index.py
```

## 状态约定

- `queued`：准备读
- `reading`：正在读
- `summarized`：已有摘要
- `deep-read`：有深入阅读笔记
- `implemented`：已复现实验或落地相关想法
- `archived`：归档

## 索引

- `index/all-papers.md`
- `index/by-date.md`
- `index/by-topic.md`
- `index/by-status.md`
