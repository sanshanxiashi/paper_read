# SkillOpt: Executive Strategy for Self-Evolving Agent Skills 阅读笔记

原文：`paper.pdf`

全文中文翻译：`source/paper.zh.md`

中文导读：`notes/summary.zh.md`

本文档用于存放阅读过程中的概念解释、公式推导、问答讨论、个人理解和后续研究想法。

## 笔记索引

- [核心问题](#核心问题)
- [方法拆解](#方法拆解)
- [实验结论](#实验结论)
- [六类实验任务的数据形态](#六类实验任务的数据形态)
- [每类 benchmark 是否共享一份 skill](#每类-benchmark-是否共享一份-skill)
- [SkillOpt-Sleep](#skillopt-sleep)
- [本地规则 judge 的来源与局限](#本地规则-judge-的来源与局限)
- [值得追问的问题](#值得追问的问题)
- [可复现想法](#可复现想法)

## 核心问题

论文要回答的问题是：如果 agent skill 是冻结模型的外部过程性知识，那么能不能像训练模型参数一样，稳定、可控、可验证地训练这份 skill 文档？

它反对的不是 prompt/skill 本身，而是三种不稳定做法：

- 人工写 skill：成本高，难以根据 rollout 反馈系统更新。
- 一次性 LLM 生成 skill：没有训练循环，不能保证比初始版本更好。
- 无约束自我改写：文本变化幅度大，可能擦掉有效规则，也难以解释哪次修改带来收益。

## 方法拆解

论文把 skill 记作自然语言策略 `s`，把目标模型 `M` 固定住，通过 harness `h(M, x, s)` 在任务 `x` 上得到轨迹 `tau` 和分数 `r`。训练集提供 rollout 证据，selection split 负责接受或拒绝候选 skill，test split 只用于最终报告。

关键状态包括：

- `s_cur`：当前 skill。
- `s_best`：通过 selection gate 的最好 skill，也就是最终导出的 `best_skill.md`。
- score cache：避免重复评估同一 skill hash。
- rejected-step buffer：保存被拒绝的编辑和相关失败模式。
- optimizer-side meta skill：只给优化器看，不随部署 skill 一起发给目标模型。

一次 step 的流程：

1. 用当前 skill 跑一批训练任务，收集成功/失败轨迹、工具调用、输出、评分器反馈等。
2. 将失败和成功轨迹分成 reflection minibatch。
3. 优化器模型从失败中提出补救规则，从成功中提出应保留规则。
4. 分层合并编辑建议，并优先处理失败修复。
5. 按文本学习率 `Lt` 只应用前 `Lt` 个编辑。
6. 在 selection split 上重新评估候选 skill。
7. 严格变好才接受；否则拒绝，并把失败编辑变成下一轮负反馈。

这里的“文本学习率”不是连续数值步长，而是每轮最多允许多少条 skill edits。它限制了 skill 在文本空间中的移动幅度，使后续 rejected edits 和 accepted edits 仍然有可解释的优化历史。

## 实验结论

论文报告的主结果很强：SkillOpt 在 52 个评估单元上 best or tied-best。最明显的收益来自过程性任务，例如 SpreadsheetBench、OfficeQA、LiveMathematicianBench 和 ALFWorld；这符合直觉，因为这些任务的错误通常是“没有稳定执行过程”，而不是纯知识缺失。

消融结论比主表更值得看：

- 更多训练证据通常帮助过程性任务，但 SearchQA 这类 headroom 较小的任务较早饱和。
- reflection minibatch 和 rollout batch 的具体大小不是特别脆弱。
- 有界文本学习率比无约束 rewrite 更稳。
- rejected-edit buffer 有稳定收益，但不是部署成本。
- slow/meta update 对 SpreadsheetBench 特别关键，说明复杂工具任务需要跨 batch 的长期经验沉淀。

迁移结果说明 `best_skill.md` 不是完全记忆训练样本。尤其是 SpreadsheetBench 在 Codex 与 Claude Code 之间迁移的结果，暗示 skill 里学到的是“先检查 workbook 结构和公式、再写入静态值、再验证目标范围”这类程序性纪律，而不是某个 harness 的固定命令。

## 六类实验任务的数据形态

论文正文没有逐条列出六个 benchmark 的原始样本，它主要说明任务形态和 evaluator。下面是按论文 protocol 和公开 benchmark 形态整理的代表性样例；除非另有说明，这些是“形态示例”，不是论文逐字原始样本。

| Benchmark | 一条 task 的数据长什么样 | 代表性例子 |
|---|---|---|
| SearchQA | Jeopardy 风格问题 + 搜索引擎 snippets/titles/urls + 标准短答案 | 输入问题：“He was the only U.S. president to also serve as Chief Justice.” 搜索片段包含 “William Howard Taft served as the 27th president and later as Chief Justice...” 输出：`William Howard Taft` |
| SpreadsheetBench / Sheet | 一个 `.xlsx` 文件 + 自然语言操作指令 + golden workbook/cell-level judge | 输入文件 `sales.xlsx` 有 `Orders` sheet：`Region, Product, Revenue`。指令：“在 Summary!B2:B5 中填入每个 Region 的总 Revenue，并保留其他单元格不变。” Agent 需要用 `openpyxl` / `pandas` 读取、计算、写回 workbook。评分看目标单元格是否和 golden file 一致。 |
| OfficeQA | 本地文档语料库，任务需要检索 PDF/parsed pages、读表、算数、格式化答案 | 输入：一组 Treasury Bulletin PDF/解析页 + 问题：“In the June 1968 bulletin, what was the reported total for marketable public debt securities held by commercial banks?” Agent 要定位文档、表格、单位并输出精确数值。 |
| DocVQA | 单页文档图片 + OCR/视觉内容 + 问题；答案通常是图片中的一个文本 span | 输入：一张扫描表格/表单/信件图片。问题：“What is the invoice number?” 图片右上角写着 `Invoice No. 78431`。输出：`78431`。 |
| LiveMathematicianBench / LiveMath | 来自近期 arXiv theorem 的数学多选题；输入给 question + choices | 输入：问题问某个新论文定理在给定条件下能推出哪种最强结论。选项 A 是弱 corollary，B 是错误加强，C 是正确最强 theorem statement。输出：`C`。 |
| ALFWorld | 文本化 household environment；给目标、当前 observation、可执行 action；agent 多步交互 | 目标：“put a clean mug in the cabinet”。动作序列可能是：`go to countertop` -> `take mug from countertop` -> `go to sink` -> `clean mug with sink` -> `go to cabinet` -> `put mug in cabinet`。成功由环境状态判定。 |

论文特别强调这些 task 不只是最终答案格式不同，rollout 轨迹也不同：SearchQA、DocVQA、LiveMath 更像单轮 QA；OfficeQA 最多约 24 次 tool calls；SpreadsheetBench 是多轮代码执行和真实 `openpyxl/pandas` runtime，最多约 30 turns；ALFWorld 是持续环境交互，最多约 50 steps。

## 每类 benchmark 是否共享一份 skill

这六类 benchmark 不是混在一起训练一份共同 skill。论文的基本设定是：给定一个目标 domain、一个初始 skill、一个固定 target model 和 harness，SkillOpt 在该 domain 的 `train / selection / test` split 上迭代优化，最后导出该 setting 的 `best_skill.md`。

所以实验里更接近以下形式：

- SearchQA skill。
- SpreadsheetBench skill。
- OfficeQA skill。
- DocVQA skill。
- LiveMath skill。
- ALFWorld skill。

每份 skill 学到的是该 benchmark 的过程性规则。例如 SpreadsheetBench 学 workbook 结构检查、公式处理、写入静态值和目标 range 验证；ALFWorld 学 visited/frontier ledger、避免重复搜索、拿到目标物后再去目标位置。

论文后面做了迁移实验，例如把 SpreadsheetBench 上训练出的 skill 从 Codex harness 迁移到 Claude Code harness，或把 OlympiadBench skill 迁到 Omni-MATH。但这是“先在一个源任务/环境训练一份 skill，再拿去另一个相关任务/环境测试”，不是六个任务联合训练一份通用 skill。

## SkillOpt-Sleep

`SkillOpt-Sleep` 是 Microsoft SkillOpt 仓库在论文主实验之外新增的部署期伴随工具。它不是训练论文六个 benchmark 的主代码，而是把同样的思想应用到本地 coding agent 的日常使用：在用户不用 agent 的时候，回放历史会话、挖掘重复失败、生成 bounded text edits，并用 held-out gate 验证后暂存给用户 review/adopt。

它的一晚可以概括为：

```text
harvest session transcripts
  -> mine recurring tasks
  -> replay offline
  -> reflect / consolidate into bounded edits
  -> gate on real held-out tasks
  -> stage proposal for user review
```

仓库里的典型例子是 `gbrain-evals/skillopt-v1` 的 `brief-writer`。初始 skill 只说“写一份短、清晰、可读的 research brief，先给答案”，但没有要求固定结构。训练任务类似：

```json
{
  "task_id": "bm-001",
  "task": "Write a brief on whether a seed-stage dev-tools startup should ship a free tier.",
  "judge": {
    "kind": "rule",
    "checks": [
      {"op": "section_present", "arg": "Key Risks"},
      {"op": "regex", "arg": "[Cc]onfidence\\s*[:=]"}
    ]
  }
}
```

原始 skill 可能写出内容合理的 brief，但缺少 `Key Risks` 小节和 `Confidence:` 行，因此本地规则 judge 判失败。Sleep 离线 replay 后学到的 edits 是：

```markdown
- Every brief must include a clearly labeled section exactly titled `Key Risks`.
- Every brief must include a line beginning `Confidence:` followed by a concise confidence level or rationale.
- Preserve required sections even when keeping the brief short; shorten the analysis before omitting `## Key Risks` or `Confidence:`.
```

这些规则会被追加到 learned block 中，例如：

```markdown
<!-- SKILLOPT-SLEEP:LEARNED START -->
...
<!-- SKILLOPT-SLEEP:LEARNED END -->
```

然后面对 held-out 任务“Write a brief on whether a solo founder should take on a technical cofounder.”，新 skill 会推动模型输出包含 `## Key Risks` 和 `Confidence: Medium` 之类的结构，因此通过规则 judge。

同一个公开 benchmark 里还有几个小型缺陷 skill：

| Seed | 初始缺陷 | Judge 检查 | Sleep 学到什么 |
|---|---|---|---|
| `advisor` / `seed-no-verdict` | 只讨论利弊，不给明确结论 | 必须有 `Recommendation:` 和 `Confidence:` | 每次 advice 都要给 verdict + confidence |
| `thorough-analyst` / `seed-verbose` | 过度展开，写太长 | `max_chars <= 1200` | 先规划结构和字数，不要自由长文后再裁剪 |
| `quick-answerer` / `seed-no-brain-first` | 明确说不要搜索、不要用工具 | 必须调用 `search` tool | 写入 override：回答前必须实际调用 search |

所以 `SkillOpt-Sleep` 的重点不是“模型凭空幻想变聪明”，而是把历史失败、工具日志、用户纠正和可评分任务变成夜间优化数据，再把通过验证的规则写入长期 skill/memory。

## 本地规则 judge 的来源与局限

在 `gbrain-evals/skillopt-v1` 这个例子里，本地规则 judge 是 benchmark 预先设置的，不是 Codex 自己挖出来的。例如 `brief-writer` 明确写了：

```json
"checks": [
  {"op": "section_present", "arg": "Key Risks"},
  {"op": "regex", "arg": "[Cc]onfidence\\s*[:=]"}
]
```

因此这个例子证明的是：在已知评测标准存在时，Sleep 能从失败 rollout 中自动归纳出 skill edits，并用 held-out gate 防止坏规则进入长期记忆。它不证明系统能完全无监督地发现“什么才是好”。

更准确地说，Sleep 的监督信号可以分为三类：

| 场景 | judge 从哪来 | 可信度 |
|---|---|---|
| gbrain benchmark | 人工写好的 rule judge | 高，但任务很小、很人工 |
| 真实工程任务 | 测试、lint、CI、命令退出码、文件 diff、工具调用日志 | 高，最适合 Sleep |
| 普通对话/开放任务 | LLM miner 从 transcript 里抽 rubric，或用模型 judge | 可用，但风险更大，需要人工 review |

如果用户已经明确知道规则，例如“所有 brief 必须有 Key Risks 和 Confidence”，那直接改 skill 更快，没必要等 sleep。Sleep 更有价值的场景是：规则没有被显式写出，但历史会话里反复出现类似失败。例如用户多次指出 SQL 没加 `LIMIT`、又扫全表、应该先 sample；Sleep 可以从 transcripts 里挖出模式，提出规则：

```markdown
- For exploratory SQL, add `LIMIT 100` unless the user explicitly requests a full scan.
```

然后用真实 held-out 任务或本地检查验证。关键结论是：judge 必须来自某种外部反馈，不能凭空产生。最可靠的是程序化反馈，例如测试、CI、exact answer、工具调用日志、格式检查。LLM 可以帮忙生成 rubric，但那只是弱监督，不能等同于真实验证。

## 值得追问的问题

1. Selection split 的规模和代表性有多关键？论文说 gate 与 test 趋势一致，但真实业务里验证集经常偏窄。
2. 如果评分器本身有漏洞，SkillOpt 是否会学到 reward hacking 风格的 skill？
3. 单一 skill 文档是否足够覆盖高异质任务？论文承认它没有扩展成大型 skill library。
4. 对没有自动评分器的开放式任务，应该用 LLM judge、人类评审，还是多指标 gate？
5. 训练 token 成本在 SearchQA、DocVQA 这类长上下文任务上很高，什么时候值得摊销？

## 可复现想法

可以先做一个仓库内小实验：选一组有标准答案的论文问答或表格处理任务，固定 `train/selection/test`，让优化器只修改一份 Markdown skill。每次修改必须生成 patch report，包含：

- 修改前后 skill hash。
- 本次 add/delete/replace edits。
- selection 分数变化。
- 是否接受。
- 拒绝原因和后续 negative feedback。

这样可以把论文的核心思想缩小到一个可审计的 agent skill training loop。
