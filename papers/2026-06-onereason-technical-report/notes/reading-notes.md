# OneReason Technical Report 阅读笔记

原文：`paper.pdf`

全文中文翻译：`source/paper.zh.md`

中文导读：`notes/summary.zh.md`

本文档用于存放阅读过程中的概念解释、公式推导、问答讨论、个人理解和后续研究想法。

## 笔记索引

- [核心问题](#core-question)
- [论文明确说的内容](#paper-claims)
- [推断与个人理解](#interpretation)
- [讨论：推荐 CoT、SFT、RFT 与 RL](#discussion-cot-sft-rft-rl)
- [迁移想法：数学逐行批改的归因 CoT](#math-grading-attribution-cot)
- [待深入问题](#open-questions)

<a id="core-question"></a>
## 核心问题

为什么在生成式推荐模型里，`thinking mode` 过去经常不能稳定超过 `non-thinking mode`？

论文给出的答案是：推荐推理需要同时具备 itemic-token 的语义感知能力和推荐专用的认知推理结构。只有让离散 itemic token 与真实语义对齐，并让 CoT 能够围绕用户兴趣压缩、兴趣扩展和兴趣转移展开，thinking mode 才可能稳定带来推荐收益。

<a id="paper-claims"></a>
## 论文明确说的内容

- OneRec 系列生成式推荐模型已经在短视频、直播、广告、电商等真实业务中部署，但纯 itemic sequential data 难以自然产生有意义的 CoT。
- 作者从多模态 CoT 鲁棒性研究中借鉴了两个条件：模态感知对齐和高质量 CoT。
- OneReason-Bench 将能力拆成 R0 Perception、R1 Derivation、R2 Evolution、R3 Recommendation。
- 预训练使用 token、item、relation、user 四种粒度数据来强化 itemic-token 与文本语义空间的对齐。
- SFT 的 R3 推荐 CoT 采用 Persona Abstraction、Interest Expansion、Transition Inference 三阶段。
- RL 阶段发现直接多域混合 RL 容易产生跨域干扰，因此采用先分域 RL，再通过 RFT 或 MOPD 统一。
- 论文报告线上部署采用 Fast-Slow Thinking 架构，并在快手本地生活广告场景中带来收入和曝光提升。

<a id="interpretation"></a>
## 推断与个人理解

- 这篇论文更像是对“推荐 CoT 为什么失败”的系统性调试报告，而不是单纯提出一个新模型结构。
- 推荐推理不是数学式 deduction，而是 abductive reasoning：从噪声行为日志中假设潜在兴趣，再选择最可能延续的方向。
- RFT 更偏向筛选成功推理轨迹，因此更容易让 thinking mode 明显超过 non-thinking mode；MOPD 更偏向整体分布校准，因此 non-thinking 能力也会被同步增强。
- 论文中“CoT 监督也能提升 non-thinking 推理”的现象很重要，但机制尚未被严格因果拆解，可能来自压缩、推理或二者交互。

<a id="discussion-cot-sft-rft-rl"></a>
## 讨论：推荐 CoT、SFT、RFT 与 RL

### 推荐 CoT 的 thinking mode 为什么不 work？

推荐 CoT 过去不稳定 work，核心原因不是“推荐不适合推理”，而是很多 CoT 在错误对象上推理、用错误结构推理，并且后续优化方式没有直接筛掉无效推理。

首先，推荐里的 itemic token 不是自然语言概念，而是离散 ID 序列，例如 `<|video_begin|><a_7300><b_7894><c_4541>`。如果模型没有先学会这些 token 对应的内容、风格、商品属性、受众和行为语义，那么 CoT 只是在不透明符号上编一段看似合理的文本。论文把这类问题类比为多模态模型中的 perception failure：感知对象没有对齐，后面的推理越长越容易空转。

其次，推荐推理不是数学推理。数学题更接近 deductive reasoning，有明确条件、明确中间步骤和单一答案；推荐更接近 abductive reasoning，用户真实意图不可观测，历史行为有噪声，一个用户也可能同时有多个合理兴趣方向。因此，推荐 CoT 不能只是“用户看过 A，所以喜欢 B，所以推荐 C”的表层相似解释，而要能从行为日志中压缩潜在兴趣、保留少量候选假设，再判断哪个兴趣方向最可能延续。

第三，自由文本 CoT 可能引入 textual inertia。也就是说，模型生成的自然语言解释可能让通用文本先验占主导，反而稀释历史 itemic token 中更具体的证据。一个 CoT 写出“用户喜欢游戏、二次元、年轻男性内容”对人类可读，但未必能提高具体 target itemic token 的概率，甚至可能把预测分布带到更宽泛的大类空间。

第四，CoT 质量差会造成 reasoning drift。典型问题包括过度解释、抓错证据、把偶然行为当成稳定偏好、把平台共现当成因果、把目标 item 的信息泄漏进 rationale。论文强调推荐 CoT 需要 evidence-grounded、decision-oriented、leakage-safe，而不是仅仅流畅可读。

最后，多域混合 RL 会放大干扰。广告、电商、直播、短视频的用户意图、item 语义和 reward landscape 不同。论文观察到 SFT 后直接做 mixed-domain RL 时，thinking mode 仍然可能弱于 non-thinking；但单域 RL 中 thinking mode 更容易超过 non-thinking。这说明“推理”本身有用，但多业务域混在一起优化时会把推理能力冲散。

### 本文如何重构推荐 CoT？

OneReason 不是直接写一个更长的 prompt，而是按 R0-R3 重新搭建推荐推理能力。

R0 是 Perception，目标是让 itemic token 可读。模型需要知道每一层 token 的语义贡献：粗粒度 token 判断内容大类，中粒度 token 缩小到人物、场景或属性，细粒度 token 支撑最终 caption。这个阶段的 CoT 不负责推荐决策，只负责把 itemic-token 到文本语义的 grounding 显式化。

R1 是 Derivation，目标是学会 item-to-item 的局部桥接。它不是简单学习“相似 item”，而是学习一个源 item 如何自然引出后续需求。例如用户看装修板材价格，可能自然转向软装、床品、空间搭配。这个阶段训练的是从源侧证据推出 follow-up need 的一跳推理能力。

R2 是 Evolution，目标是学会用户兴趣随时间演化。它不把历史行为当成无序集合，而是抽取触发、细化、纠正、闭环等时间结构。例如搜索触发需求，后续点击缩小参数，购买完成某个需求，跨域行为形成 echo。这个阶段让模型理解用户兴趣是过程，不是静态标签。

R3 是真正的 Recommendation CoT。R3 把 R0-R2 组合起来，构造最终推荐链路。核心协议是三段式：

- Persona Abstraction：把稀疏、噪声行为压缩成软画像或用户状态。这里不是给用户贴死标签，而是形成可被后续证据修正的 prior，例如家庭实用消费型用户、直播购物敏感用户、可能多人共用设备。
- Interest Expansion：从近期行为中展开少量候选兴趣假设。关键是少量且 evidence-grounded。论文实验显示 expansion width 取 1、3、5 通常比 10、20 好，因为候选太多会把强信号稀释掉。
- Transition Inference：对候选兴趣做比较并提交决策。比较维度包括证据强度、近期性、时间连续性、画像兼容性、目标域兼容性和泄漏风险。最后才输出 target itemic token。

抽象流程可以写成：

```text
用户历史 H + 用户画像 P
-> 压缩成用户状态 C
-> 展开候选兴趣集合 Z
-> 比较候选兴趣 z
-> 选择最可能的下一兴趣方向
-> 输出 itemic token
```

旧式 CoT 往往是：

```text
用户看过 A、B、C
所以用户喜欢某类内容
所以推荐 D
```

OneReason 的 CoT 更像：

```text
用户状态压缩：
这些行为显示用户可能是 X 类型，但这只是软先验。

兴趣展开：
候选 A：由搜索、点击、购买共同支持，近期强。
候选 B：有长期偏好，但近期弱。
候选 C：只是偶然浏览，降权。
候选 D：跨域行为有桥接关系，可以保留。

转移判断：
A 的证据强度、近期性、行为闭环最好；
B 作为辅助兴趣；
C 不作为主方向；
因此下一步应围绕 A 推荐。

输出：
<|video_begin|><a_x><b_y><c_z>
```

关键差异是：OneReason 把 CoT 从“解释文本”改成了“推荐决策中间变量”。Persona Abstraction 是降噪和压缩，Interest Expansion 是受控搜索，Transition Inference 是排序和提交。

### 为什么 SFT 阶段 CoT 可能降低 target likelihood？

论文用 `ΔLL = log p(y_GT | x, CoT) - log p(y_GT | x)` 衡量 CoT 是否真正帮助预测 target item。如果 `ΔLL < 0`，表示模型看了 CoT 之后，反而比不看 CoT 更不相信真实 target item。

这不等价于 SFT CoT 是错的。更准确地说，SFT 数据里的 CoT 可以是 teacher 视角或人工构造流程视角的 golden，但不一定是当前 student 模型视角的 golden。

原因包括：

- Teacher-golden 不等于 student-useful。Teacher 写出的 rationale 对人可读、逻辑上合理，但 student 未必能把这段自然语言推理转成 itemic-token 分布增益。
- SFT 优化的是模仿，不是命中。SFT loss 只要求模型复现 `<think>... reasoning ...</think> target_itemic_token`，没有直接优化“这段 CoT 是否提高 target item 概率”。
- 推荐 CoT 容易产生宽泛文本先验。比如“用户喜欢游戏、二次元、年轻男性内容”虽然合理，但对具体 `<|ad_begin|><a_123><b_456><c_789>` 未必有判别力。
- 推荐存在多个 plausible item。SFT trace 可能解释了一个大方向，但没有足够强的区分信号把 ground-truth item 从大量相似候选中拉出来。
- 训练时 CoT 是 teacher 写的，推理时 CoT 是 student 自己写的。student 生成的 CoT 可能发生 drift，早期一句判断偏了，后续 item 预测就会被带偏。

因此，“SFT 阶段 CoT 是 golden”最多说明它是数据构造流程认可的 rationale，不说明它对当前模型预测有正因果贡献。

### RFT 为什么能把 CoT 变成正贡献？

RFT 的关键差异是：它不是保留“看起来合理”的 CoT，而是保留实际命中 target 的 CoT trajectory。

流程可以概括为：

```text
SFT model / domain RL teacher
-> 对同一个用户上下文采样多条 CoT + itemic token
-> 用真实 target / reward 检查是否命中
-> 只保留 verified successful trajectories
-> 再用这些轨迹做 SFT-style 训练
```

这样，CoT 从 teacher-rationale 变成 outcome-verified rationale。RFT 训练的是“这种 CoT 风格和这种中间兴趣判断确实曾经把模型带到正确 itemic token”，而不是“这段 CoT 人看起来合理”。

这也解释了论文为什么说 RFT 更容易保证 `thinking > non-thinking`。它筛掉了大量解释合理但预测没用的 CoT，只蒸馏推理链和正确推荐对齐的轨迹。

### 本文里的 RFT 是接着 SFT 训练，还是重新训练？

论文里的 RFT 不是从头重新训练。更准确地说，它是在已有 checkpoint 上继续训练。

需要区分 RFT 数据来源和 RFT 初始化 checkpoint：

- RFT 数据来源：domain-specific RL teachers。广告、电商、直播、视频各自强化学习后的 specialist model 采样出 CoT + itemic token，再经过 outcome filtering，只保留命中轨迹。
- RFT 初始化：论文 Figure 10 和 RFT 描述中，最终 RFT 以 Mix-RL checkpoint 为基础继续训练，而不是从零训练，也不是简单从原始 SFT checkpoint 直接训练。

完整路径更接近：

```text
SFT model
-> domain-specific GRPO 得到各域 teacher
-> mixed-domain GRPO 得到 Mix-RL checkpoint
-> 用各域 teacher 采样并过滤 successful trajectories
-> 从 Mix-RL checkpoint 继续做 RFT
-> final RFT model
```

如果没有 Mix-RL，也可以做简化版：从 SFT checkpoint 继续 RFT。但论文的完整 recipe 是 `SFT -> Single-domain RL teachers + Mix-RL student -> RFT from Mix-RL checkpoint`。

### RFT 和 RL 的关系

RFT 不是 policy-gradient RL 算法本身，而是 reward-filtered supervised learning。它更像：

```text
sampling / exploration / reward filtering
-> verified trajectories
-> supervised fine-tuning on verified trajectories
```

所以 RFT 处在 SFT 和 RL 之间：

- 像 SFT：训练目标仍然是 next-token prediction，不直接做 policy gradient。
- 像 RL：数据不是静态人工标注，而是通过 rollout + reward / rule filtering 得到，只保留高 reward 轨迹。

因此可以把路线粗略分成：

```text
SFT -> RL
```

直接用 GRPO/PPO 等 policy optimization，让模型根据 reward 更新策略。

```text
SFT -> RFT
```

先从模型或 teacher 采样多条轨迹，用 reward 过滤，再用成功轨迹继续 SFT。

但本文实际采用更复杂的 specialize-then-unify：

```text
SFT
-> single-domain RL teachers
-> Mix-RL checkpoint
-> RFT 或 MOPD 做多域统一
```

一句话总结：RFT 是 reward-filtered supervised learning，不是 policy-gradient RL；但它依赖 rollout 和 reward selection，所以可以看成一种离线、低风险、监督化的 RL 后处理。

### 对实践的启发

如果要复用本文思路，比较合理的简化流程是：

```text
1. SFT：训练结构化 CoT 格式
2. 多采样：对每个样本采 N 条 CoT，每条 CoT 再采 K 个 item
3. 过滤：
   - target hit / Recall@K hit
   - CoT 不泄漏 target
   - CoT 引用的历史 item 合法且来自 user history
   - CoT 后 target likelihood 比 no-CoT 更高，最好 ΔLL > 0
4. RFT：只用通过过滤的 CoT + item 继续训练
5. 评估：
   - thinking vs non-thinking
   - ΔLL
   - Recall@K
   - head/tail 分桶
```

关键不在“再训一轮”，而在“用真实推荐命中结果筛掉无效 CoT”。如果只是把 SFT 数据重复训练，很可能只会强化格式模仿；如果用 outcome-verified trajectories 做 RFT，才更接近论文中 RFT 带来的正向 CoT 贡献。

<a id="math-grading-attribution-cot"></a>
## 迁移想法：数学逐行批改的归因 CoT

一个可迁移场景是数学解答题的学生逐行作答批改：需要判断每一行是正确、错误、笔误、遗漏、未完成等。这个任务与 OneReason 的推荐 CoT 有相似结构：二者都不是简单分类，而是在模糊边界下做证据化决策。

推荐任务中，模型需要从用户历史中压缩兴趣、展开候选意图、判断兴趣转移，再输出 item。数学逐行批改中，模型需要从题目、参考解、学生历史步骤和当前行中识别数学语义、展开可能错误类型、判断边界条件，再输出标签与归因。

可以建立如下对应：

```text
推荐任务：
用户历史 -> 压缩兴趣 -> 展开候选意图 -> 判断转移 -> 输出 item

数学批改：
题目 + 标准解/参考步骤 + 学生逐行作答
-> 对齐当前行的数学语义
-> 展开可能错误类型
-> 判断边界条件
-> 输出标签 + 归因
```

### 为什么适合归因式 CoT？

正确、错误、笔误、遗漏、未完成之间不是硬边界。例如同样写错一个负号，可能有多种判法：

- 如果前后推导都显示学生实际知道正确值，只是当前行抄错，可能判为笔误。
- 如果后续持续依赖这个错误值，通常应判为错误。
- 如果当前行本身没有错，但跳过了必要推理桥梁，可能判为遗漏。
- 如果解题链条没有达到可判定目标，可能判为未完成。

因此，这个任务需要模型学习 decision boundary，而不是只学习从文本到标签的表面映射。

### 可类比 OneReason 的四层结构

R0：数学语义感知。先让模型读懂当前行在数学上表达了什么，而不是直接判标签。需要识别本行做了什么操作、用了哪个公式或定理、变量和符号是否可解释、是否与上一行存在代数等价关系。

R1：局部步骤校验。判断上一行到当前行是否成立：是否代数等价，变形是否合法，是否有计算错误，是否有符号抄写错误，是否引入未说明的新假设。

R2：全局解题轨迹演化。很多行需要看后续才能判断。例如当前行有笔误但后面自动纠正，倾向笔误；当前行错误且后面持续使用，倾向错误；当前行跳过关键证明但答案对，可能是遗漏；当前行只是中间式且没有继续，可能是未完成。

R3：最终批改决策。把当前行的语义、局部校验和全局轨迹合并起来，输出标签和归因。它不应该复述完整解题过程，而应该解释为什么这一行落在某个标签边界内。

### 推荐的批改 CoT 结构

可以把推荐中的 `Persona Abstraction -> Interest Expansion -> Transition Inference` 改写为：

```text
Step Semantics -> Error Hypothesis Expansion -> Boundary Decision
```

含义如下：

- Step Semantics：这行在做什么数学动作。
- Error Hypothesis Expansion：展开少量候选归因，例如正确等价变形、计算错误、符号笔误、关键步骤遗漏、未完成。
- Boundary Decision：根据上下文证据选择标签，例如局部表记错误且后文恢复则判笔误，错误影响后续推导则判错误，当前行正确但缺少必要依据则判遗漏。

一个训练样例可以组织成：

```json
{
  "line_id": 4,
  "student_line": "x = -2",
  "label": "笔误",
  "reason": {
    "step_semantics": "本行是在由上一行的一次方程求解 x。",
    "local_check": "上一行正确求解应得到 x = 2，本行符号为负。",
    "context_check": "后续步骤继续使用 x = 2，说明学生实际意图与正确值一致。",
    "boundary_decision": "错误只出现在当前行符号，且未传播到后续推导，因此判为笔误而非错误。"
  }
}
```

### SFT + RFT 的训练启发

这个任务也适合使用 `SFT -> 多采样归因 -> outcome/judge filtering -> RFT`。

SFT 阶段可以先教模型输出结构化批改归因，但 SFT 标注里的 reason 即使看起来合理，也不一定真正帮助模型稳定区分边界。可以让模型对同一行生成多种批改归因，然后过滤：

- label 是否与人工标注一致。
- reason 是否引用了正确上下文。
- 是否正确区分“笔误 vs 错误”。
- 是否没有替学生脑补没有写出的意图。
- 是否通过 rule checker、symbolic verifier 或 stronger LLM judge。

再只保留高质量轨迹做 RFT。关键不是简单重复 SFT 数据，而是用 outcome-verified 或 judge-verified 的归因链筛掉无效 CoT。

### 需要特别定义的边界

“笔误”是最危险的标签，因为模型容易把小错误都宽容成笔误。需要在数据和 CoT 中明确边界：

```text
笔误 = 局部表记错误 + 上下文显示正确意图 + 不影响或后续自我修正
错误 = 数学关系不成立 + 后续继续依赖该错误
遗漏 = 结论可能对，但必要推理桥梁缺失
未完成 = 解题链条没有达到可判定目标
```

一句话总结：推荐里的 CoT 是用户意图归因，数学逐行批改里的 CoT 是错误类型归因。二者本质上都是模糊边界下的证据化决策，因此 OneReason 的“先对齐语义、再构造归因 CoT、最后用验证轨迹筛选”的思路可以迁移到这个批改任务。

<a id="open-questions"></a>
## 待深入问题

- itemic-token perception 的收益能否迁移到非推荐场景，例如工具调用轨迹、代码仓库对象、数据合成实体 ID？
- 三段式 CoT 中哪些部分真正贡献预测收益，哪些只是可解释性包装？
- RFT 与 MOPD 在稀疏奖励场景中的差异，能否用于一般 RLVR 或 agent trajectory distillation？
- 在线 Fast-Slow Thinking 的工程边界是什么：慢模型刷新频率、快模型蒸馏延迟、ROI 与召回覆盖之间如何取舍？
