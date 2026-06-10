# Why Larger Models Learn More 中文导读

原文：<https://arxiv.org/abs/2605.29548>

## 论文信息

- 标题：Why Larger Models Learn More: Effects of Capacity, Interference, and Rare-Task Retention
- 作者：Jing Huang, Daniel Wurgaft, Rachit Bansal, Laura Ruis, Naomi Saphra, David Alvarez-Melis, Andrew Kyle Lampinen, Christopher Potts, Ekdeep Singh Lubana
- 年份：2026
- arXiv：2605.29548v2，2026-06-01
- 主题：scaling law, data mixture, rare-task retention, gradient interference, OLMo pretraining

## 一句话概括

论文试图解释为什么大模型会学到小模型学不到的任务：核心不是小模型只缺训练数据，而是在混合数据训练中，小模型的有限神经元更容易被高频或低复杂度任务占用，导致稀有和复杂任务的特征被频繁任务梯度覆盖；更大的模型能降低这种干扰并保留稀有任务信号。

## 摘要要点

- 作者先用 scaling-law 的现象学论证区分两类收益：靠更多数据可弥补的收益，以及必须靠模型规模才能弥补的收益。
- 合成多任务回归实验显示，模型宽度增加会优先补上低频或复杂任务，因为这些任务的特征 utility 更低，小模型在容量受限时不会保留这些特征。
- 干扰机制解释为：稀有任务样本出现后会推动模型朝稀有任务特征更新，但在下一次稀有样本到来前，频繁任务更新可能把这部分信号覆盖掉。
- 大模型可以为频繁任务分配足够资源，使频繁任务残差和梯度变弱，从而减少对稀有任务特征的覆盖。
- OLMo 4M 到 4B 参数预训练实验复现了类似现象：更大的模型更能学到低频且复杂的注入任务，也表现出更多任务特征和更低任务间梯度干扰。

## 关键贡献

- 把“大模型学得更多”从涌现能力叙事转成数据混合中的资源竞争问题。
- 给出可分析的合成任务模型，把任务频率、任务复杂度、模型宽度和特征保留顺序联系起来。
- 用 rare-task injection 动态展示“学到一点又被覆盖”和“大模型逐步累积稀有任务信号”的差异。
- 在 OLMo 预训练管线中验证合成实验结论，使论点不只停留在 toy setting。

## 阅读建议

优先读第 2 节的定义，明确“learnable via data scaling”和“learnable via model scaling”的差别；再读第 3 节理解 utility 排序和梯度干扰机制；最后读第 4 节看 OLMo 实验如何把理论映射到真实预训练。
