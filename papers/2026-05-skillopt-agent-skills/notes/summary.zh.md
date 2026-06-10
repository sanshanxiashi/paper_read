# SkillOpt: Executive Strategy for Self-Evolving Agent Skills 中文导读

原文：https://arxiv.org/abs/2605.23904

## 论文信息

- 标题：SkillOpt: Executive Strategy for Self-Evolving Agent Skills
- 作者：Yifan Yang, Ziyang Gong, Weiquan Huang, Qihao Yang, Ziwei Zhou, Zisu Huang, Yan Li, Xuemei Gao, Qi Dai, Bei Liu, Kai Qiu, Yuqing Yang, Dongdong Chen, Xue Yang, Chong Luo
- 年份：2026
- 机构：Microsoft, Shanghai Jiao Tong University, Tongji University, Fudan University
- 主题：agent skills, text-space optimization, validation-gated update, agent evaluation

## 一句话概括

SkillOpt 把 agent skill 文档当成冻结模型外部的一份“可训练文本状态”：用 rollout 轨迹产生增删改建议，用文本学习率限制每次修改幅度，并且只有候选 skill 在 held-out selection split 上严格变好时才接受。

## 论文明确说的内容

论文认为，agent 的领域适配不只发生在模型权重或 prompt 上，也发生在 agent 如何搜集证据、调用工具、遵循任务约束和格式化输出的过程里。Skill 文档正好承载这类过程性知识，因此应该像参数一样被系统优化，而不是只靠人工写、一次性 LLM 生成，或无约束自我改写。

SkillOpt 的核心循环是：

1. 固定目标模型和执行 harness，用当前 skill 在训练任务上跑 rollout。
2. 优化器模型读取成功和失败轨迹，按 minibatch 反思，生成结构化的 add/delete/replace edits。
3. 合并和排序编辑建议，只保留不超过文本学习率预算 `Lt` 的少量修改。
4. 将候选 skill 放到 selection split 上评估；只有分数严格高于当前 skill 才接受，否则把失败编辑放入 rejected-edit buffer。
5. 跨 epoch 做 slow/meta update，把长期稳定的改进、退化和持续失败总结成训练侧指导；部署时只导出紧凑的 `best_skill.md`。

论文在六类 benchmark、七个目标模型、三种执行环境上评估，包括 SearchQA、SpreadsheetBench、OfficeQA、DocVQA、LiveMathematicianBench 和 ALFWorld；执行环境包含 direct chat、Codex harness 和 Claude Code harness。论文报告 SkillOpt 在 52 个 `(model, benchmark, harness)` 单元上都是最好或并列最好。

在 GPT-5.5 direct chat 设置下，论文报告从无 skill 到 SkillOpt 的平均提升是 +23.5 分；在 Codex agentic loop 内是 +24.8 分；在 Claude Code 内是 +19.1 分。代表性提升包括 SpreadsheetBench 41.8 -> 80.7、OfficeQA 33.1 -> 72.1、LiveMathematicianBench 37.6 -> 66.9。

消融实验支持四个关键设计：有界文本学习率、held-out validation gate、rejected-edit buffer、epoch-wise slow/meta update。特别是在 SpreadsheetBench 上，去掉 meta skill 和 slow update 会从 77.5 掉到 55.0，说明长期稳定过程知识对工具型任务很重要。

迁移实验显示，训练好的 skill 可以跨模型规模、跨 Codex/Claude Code harness、以及从 OlympiadBench 到 Omni-MATH 的相近数学任务迁移。论文特别强调：优化器模型只在离线训练时调用，部署阶段没有额外 optimizer 调用，只增加一份短文本 skill。

## 我的理解 / 推断

这篇论文最有价值的地方不是“自动写 prompt”，而是把 skill 写作变成了带验证集的训练问题。它把几个深度学习训练里的工程纪律搬到文本空间：batch evidence 对应梯度估计，编辑预算对应 learning rate，selection split 对应 validation gate，rejected edits 对应负反馈，slow/meta update 对应长期动量或经验总结。

这种范式最适合有自动评分器、可重复 rollout、错误模式稳定的任务，例如表格处理、文件工具链、结构化 QA、数学选择题、具身环境等。对开放式研究、主观写作、多目标偏好优化任务，论文自己的限制也成立：validation gate 不容易定义，可能需要人类或模型评审器。

从 agent 工程角度看，SkillOpt 的实用启发是：不要让 agent 无限制重写自己的全局提示；更稳的做法是保留小步 patch、显式记录 rejected edits，并且把每次变更绑定到可复现的验证集分数。

## 关键贡献

- 将 agent skill learning 表述为冻结模型外部自然语言状态的优化问题。
- 给出一套受控文本优化器：rollout evidence、minibatch reflection、add/delete/replace edits、文本学习率、held-out gate、rejected buffer、slow/meta update。
- 在 direct chat、Codex、Claude Code 三类 harness 下做系统实验，而不是只在单一 prompt 环境里验证。
- 展示训练出的 skill 仍然短小可读，论文报告最终 skill 约 300 到 2000 tokens，且通常只接受 1 到 4 次 bounded edits。
- 展示跨模型、跨 harness、跨相近 benchmark 的迁移潜力。

## 阅读建议

优先读第 3 节 Method，抓住 SkillOpt 的状态变量和训练循环；再读第 4.1 节主结果，确认收益来自哪些任务；接着读第 4.2 节消融，判断每个组件是否真的必要；最后读 Appendix B 的限制，避免把它误读成通用自我进化方法。

如果想落地复现，建议先选一个有自动判分的小型任务集合，固定 train/selection/test split，只优化一份 `best_skill.md`，并记录每次 patch、selection 分数和 rejected edits。
