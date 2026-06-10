# OneReason Technical Report 中文导读

原文：https://arxiv.org/abs/2606.06260

## 论文信息

- 标题：OneReason Technical Report
- 作者：OneRec Team 等
- 年份：2026
- 主题：生成式推荐、推理模型、itemic-token 对齐、推荐强化学习、工业部署

## 一句话概括

OneReason 试图让生成式推荐模型真正获得“先想再推荐”的能力：先把离散 itemic token 对齐到可理解的文本语义，再用推荐专用 CoT 和分域强化学习让 thinking mode 稳定转化为推荐收益。

## 摘要要点

- 论文指出，推荐场景中的 CoT 难点不只是缺少提示词，而是 itemic token 本身通常是不透明 ID，缺少语义 grounding。
- 作者把推荐推理能力拆成两类：Perception，即 itemic token 与底层语言语义的对齐；Cognition，即把用户行为序列重组为稳定、可解释的潜在兴趣点。
- OneReason 的训练流程包括三部分：强 itemic-token perception 的预训练、三层 cognition-enhanced CoT 的 SFT、以及 specialize-then-unify 的 RL 配方。
- 论文提出 OneReason-Bench，把推荐推理分为 R0 Perception、R1 Derivation、R2 Evolution、R3 Recommendation 四层。
- 在线部署采用 Fast-Slow Thinking 架构：慢模型 OneReason 生成高质量推理信号，快模型 OneRec 吸收这些信号用于低延迟线上服务。

## 关键贡献

- 明确诊断了推荐中 thinking mode 失败的两个根因：itemic-token 感知不足，以及推荐 CoT 结构质量不足。
- 给出推荐专用三段式 CoT：Persona Abstraction、Interest Expansion、Transition Inference。
- 在 RL 阶段提出先分域强化、再统一蒸馏或 RFT 的训练策略，以降低多业务域之间的优化干扰。
- 报告了 Kuaishou 真实业务场景中的线上 A/B 收益。

## 阅读建议

优先阅读第 2 节的推荐推理定义、第 5 节的三段式 CoT 设计、第 6 节的 specialize-then-unify RL，以及第 9 节的工业部署。阅读时需要区分论文明确报告的实验现象和作者对 CoT 机制的推断。
