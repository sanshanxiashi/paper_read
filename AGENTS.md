# AGENTS.md

本仓库是一个 Markdown-first 的论文阅读库，用于组织论文 PDF、抽取文本、中文翻译、中文摘要、阅读笔记、外部解读和自动生成的索引。

## 项目结构

```text
paper_read/
├── README.md
├── AGENTS.md
├── papers/
│   └── <paper-id>/
│       ├── meta.yaml
│       ├── README.md
│       ├── paper.pdf
│       ├── source/
│       │   ├── paper.en.txt
│       │   └── paper.zh.md
│       ├── notes/
│       │   ├── reading-notes.md
│       │   ├── summary.zh.md
│       │   └── external-views.md
│       └── assets/
│           └── figures/
├── index/
├── templates/
├── tools/
├── src/
├── spec/
└── docs/superpowers/plans/
```

## 论文目录规则

- 每篇论文放在 `papers/<paper-id>/` 下。
- `paper-id` 使用 `YYYY-MM-short-slug` 格式，例如 `2026-06-mai-thinking-1`。
- 原始 PDF 统一命名为 `paper.pdf`。
- 抽取文本和翻译放在 `source/` 下。
- 所有可读笔记放在 `notes/` 下。
- 截图、论文图表、手动画图等放在 `assets/figures/` 下。
- 不要把某篇论文的正式文件留在仓库根目录。

## 元数据

每篇论文目录必须包含 `meta.yaml`。新增论文时，从 `templates/meta.yaml` 复制并填写。

必填字段：

```yaml
id: "YYYY-MM-paper-slug"
title: "Paper Title"
authors:
  - "Author Name"
year: YYYY
read_date: "YYYY-MM-DD"
status: "queued"
topics:
  - "topic-slug"
source_url: ""
local_pdf: "paper.pdf"
notes:
  reading: "notes/reading-notes.md"
  summary: "notes/summary.zh.md"
  external_views: "notes/external-views.md"
```

除非用户明确要求，否则 `status` 使用以下取值：

- `queued`：准备阅读
- `reading`：正在阅读
- `summarized`：已有摘要
- `deep-read`：已有深入阅读笔记
- `implemented`：已复现实验或落地相关想法
- `archived`：归档

`topics` 使用稳定的小写英文 slug，例如：

- `pretraining`
- `mid-training`
- `post-training`
- `reinforcement-learning`
- `reasoning-model`
- `data-mixture`
- `synthetic-data`
- `evaluation`
- `scaling-law`
- `agentic-coding`
- `safety`
- `model-infrastructure`

## 索引生成

`index/` 下的文件是自动生成索引。新增或修改任何 `meta.yaml` 后，必须运行：

```bash
python3 tools/build_index.py
```

该命令会重新生成：

- `index/all-papers.md`
- `index/by-date.md`
- `index/by-topic.md`
- `index/by-status.md`

在声称索引已更新前，必须确认命令成功退出。

## 阅读笔记写法

- `notes/reading-notes.md` 是主阅读笔记，用于存放问答、概念解释、公式推导、个人理解和研究想法。
- `notes/summary.zh.md` 是中文导读或章节摘要，要求更简洁。
- `notes/external-views.md` 用于记录外部博客、文章、视频或讨论串的解读。
- 新讨论优先作为新章节追加，并同步更新笔记索引。
- 外部文章必须保留来源链接。
- 解释论文观点时，要区分“论文明确说的内容”和“推断 / 个人理解”。

## 翻译与文本抽取

- 英文抽取文本放在 `source/paper.en.txt`。
- 中文全文翻译放在 `source/paper.zh.md`。
- 临时缓存、样例翻译、一次性脚本不要留在仓库根目录。
- 可复用工具放在 `tools/` 下。

## Spec-First 编码规范

### 核心原则

**任何 Python 代码新增或修改之前，必须先写对应的 spec。**

这里的 Python 代码包括：

- `src/**/*.py`
- `tools/**/*.py`
- 实验脚本
- 数据处理脚本
- 评测脚本
- 论文解析、翻译、索引、检索等辅助脚本

仅修改 Markdown 文档、`meta.yaml`、模板文件或纯数据文件时，不强制新增 spec。

### Python 代码位置

后续新增的正式 Python 代码默认放到：

```text
src/
```

`tools/` 只放仓库级命令行工具或维护脚本，例如索引生成、批量迁移、PDF 抽取等。

默认 Python 环境为：

```bash
/mnt/data/leizhu13/miniconda3/envs/gpt37/bin/python3.7
```

新增脚本、验证命令和 spec 中的示例命令，默认使用该解释器。除非用户明确要求，不要假设可以使用其他 Python 版本。

### Spec 文件位置

```text
spec/
  README.md                         # spec 索引 + 模块数据流总览
  <module_name>.spec.md             # Python 模块规格
```

如果 `spec/` 目录不存在，首次新增或修改 Python 代码前必须创建。

新模块优先建立同名 spec，例如：

```text
src/paper_index/search.py -> spec/paper_index.search.spec.md
tools/build_index.py      -> spec/tools.build_index.spec.md
```

### 工作流约束

1. **改代码前先改 spec**：先在 `spec/*.spec.md` 中写清楚本次变更的目标、输入输出、关键字段、指标口径、边界情况和兼容性要求，再修改 `.py` 文件。
2. **新增 Python 文件前先建 spec**：新增模块、脚本或 CLI 入口前，必须先创建或更新对应 spec，说明模块职责、命令行参数、数据格式和失败处理。
3. **代码必须符合 spec**：实现完成后，对照 spec 验证行为一致；如果实现过程中发现 spec 不合理，先更新 spec，再调整代码。
4. **测试与验证写入 spec**：spec 中必须列出最小验证命令或样例，尤其是索引、抽取、翻译、检索和评测脚本要说明输入文件、输出文件和预期关键输出。
5. **回滚时同步回滚 spec**：回滚 Python 代码行为时，必须同步回滚或更新对应 spec，避免文档与实现不一致。

### Spec 最小模板

```markdown
# <模块名> Spec

## 目标

本模块解决什么问题，非目标是什么。

## 输入

输入文件、字段、CLI 参数和默认值。

## 输出

输出文件、字段、打印指标和错误信息。

## 行为契约

核心函数/流程的业务规则、排序规则、筛选规则和兼容性要求。

## 边界情况

空输入、解析失败、重复 id、缺失字段、旧格式兼容等处理方式。

## 验证

可执行的最小验证命令和预期关键输出。
```

## 编辑约束

- 除非用户明确要求，不要删除论文 PDF、笔记、翻译、元数据或生成索引。
- 如果论文已经有索引或笔记，不要随意重命名 `paper-id`；如确需重命名，必须同步更新链接并重新生成索引。
- 保留用户已有笔记。优先局部追加或局部编辑，不要整文件重写。
- 搜索文件或文本时优先使用 `rg`。
- 手动编辑文件时使用 `apply_patch`。
- 发生结构性变更后，运行 `python3 tools/build_index.py` 验证。
- 新增或修改 Python 代码时，遵守上面的 Spec-First 编码规范。
