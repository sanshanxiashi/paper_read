# MAI-Thinking-1 论文阅读笔记

原文：`mai-thinking-1.pdf`

全文中文翻译：`mai-thinking-1.zh.md`

中文导读：`mai-thinking-1.zh-summary.md`

本文档用于存放阅读 MAI-Thinking-1: Building a Hill-Climbing Machine 时遇到的概念、公式、指标和个人理解。它不是逐字翻译，也不是完整摘要，而是面向后续复习和查找的阅读笔记。

## 笔记索引

- [2.7 节：bits-per-byte (BPB) 是什么，怎么算](#27-节bits-per-byte-bpb-是什么怎么算)
- [2.8.4 节与图 11：MFU 和 EG 是什么，图应该怎么看](#284-节与图-11mfu-和-eg-是什么图应该怎么看)
- [EG 里的 cost 到底是 FLOPs 还是训练时间](#eg-里的-cost-到底是-flops-还是训练时间)
- [FLOPs cost 可以怎样拆成架构单位成本和训练量](#flops-cost-可以怎样拆成架构单位成本和训练量)
- [图 11 的实验主要是在调架构还是调数据](#图-11-的实验主要是在调架构还是调数据)
- [从小模型实验推广到大模型：scaling ladder 的作用](#从小模型实验推广到大模型scaling-ladder-的作用)
- [数据实验外推：从 20B tokens 推到 1T-token midtrain](#数据实验外推从-20b-tokens-推到-1t-token-midtrain)
- [外部解读 1：训练大模型不是造火箭，是攀岩](#外部解读-1训练大模型不是造火箭是攀岩)
- [外部解读 2：让模型思考不难，让它持续思考才难](#外部解读-2让模型思考不难让它持续思考才难)
- [Pre-train / mid-train 的 NLL eval 数据要求](#pre-train--mid-train-的-nll-eval-数据要求)
- [数学 NLL 低是否代表最终能答对题](#数学-nll-低是否代表最终能答对题)
- [MAI-Thinking-1 到底有没有做 SFT](#mai-thinking-1-到底有没有做-sft)
- [从任务反馈中长出来的 CoT 是否会很冗长](#从任务反馈中长出来的-cot-是否会很冗长)
- [为什么 MAI 在 pre-train / mid-train 尽量去除合成数据](#为什么-mai-在-pre-train--mid-train-尽量去除合成数据)
- [合成数据放进 mid-train 是否可能造成 teacher 污染](#合成数据放进-mid-train-是否可能造成-teacher-污染)
- [RL-first 的 reason trace 和专家 CoT SFT 的关系](#rl-first-的-reason-trace-和专家-cot-sft-的关系)
- [如果领域任务不做 RL，mid-train 放合成 reason trace 是否可行](#如果领域任务不做-rlmid-train-放合成-reason-trace-是否可行)

## 2.7 节：bits-per-byte (BPB) 是什么，怎么算

论文位置：第 2.7 节 Evaluation and Comparison with Contemporaneous Models，图 10。

原文核心句：

> We focus on pre-training base models in this comparison and report bits-per-byte (BPB) values which are invariant across tokenizers.

### 这个指标在比较什么

BPB 可以理解为：

> 模型为了预测或压缩一段文本，平均每个原始字节需要多少 bit 的不确定性成本。

所以它是一个越低越好的指标。BPB 越低，说明模型对这份 held-out 数据的预测越准确，也就是对该数据分布建模得越好。

在 2.7 节里，作者用 BPB 比较不同 base pre-trained models 在四类内部 held-out 评测集上的预训练质量：

- Held-Out Code
- Held-Out QA
- Held-Out STEM
- Held-Out Math

这些评测不是在看后训练后的聊天能力，而是在看 base model 的 next-token prediction 能力。

### 计算公式

假设评测文本是字符串 `x`，用 UTF-8 编码后有 `B` 个 byte。

某个模型自己的 tokenizer 会把文本切成 token：

```text
t1, t2, ..., tN
```

模型对每个 token 给出条件概率：

```text
p(t_i | t_<i)
```

总负对数似然为：

```text
NLL = - sum_i log p(t_i | t_<i)
```

如果 `log` 是自然对数，也就是 PyTorch / HuggingFace loss 常见的 nats 单位，那么：

```text
BPB = NLL / (B * ln 2)
```

如果已经用 `log2` 计算 NLL，那么：

```text
BPB = - sum_i log2 p(t_i | t_<i) / B
```

如果 HuggingFace 返回的是平均 token loss：

```text
loss = NLL / N
```

那么：

```text
BPB = loss * N / (B * ln 2)
```

也就是：

```text
BPB = 每 token 平均 NLL * token 数 / 原文字节数 / ln(2)
```

### 为什么不用 token loss 或 perplexity

不同模型使用的 tokenizer 不同。同一句文本，在不同 tokenizer 下可能被切成不同数量的 token。

例如：

```text
MAI-Thinking-1 is strong.
```

一个 tokenizer 可能切成 6 个 token，另一个 tokenizer 可能切成 9 个 token。直接比较每 token loss 或 perplexity 会受到 token 粒度影响，因此跨模型不公平。

BPB 的做法是把总 NLL 统一除以原始文本的 byte 数。对于同一份评测文本，byte 数是固定的，所以它比 per-token loss 更适合跨 tokenizer 比较。

这就是论文说 BPB values are invariant across tokenizers 的原因。

### 和压缩的关系

语言模型的 NLL 可以看作对文本进行熵编码时的理论编码成本。模型越能准确预测下一个 token，给真实 token 的概率越高，NLL 越低，压缩这段文本所需的 bit 数也越少。

因此：

```text
BPB 越低 = 平均每个 byte 需要的编码 bit 越少 = 模型预测越准
BPB 越高 = 平均每个 byte 需要的编码 bit 越多 = 模型越不确定
```

### 和 perplexity 的关系

Perplexity 通常定义为：

```text
PPL = exp(NLL / token数)
```

它是按 token 归一化的，因此跨 tokenizer 比较不稳定。

BPB 可以写成：

```text
BPB = log2(PPL) * token数 / byte数
```

所以 BPB 更像是文本压缩成本，而不是每 token 困惑度。

### 如何理解图 10 的数值

图 10 中的柱状图都标注了 lower is better。

例如 Code held-out 上某个模型的 BPB 为 `0.2441`，意思是：

> 该模型在这份代码评测集上，平均每个原始 byte 产生约 0.2441 bit 的负对数概率成本。

这个数值越低，表示模型对这类代码文本越熟悉，预测能力越强。

因此，2.7 节中的比较可以理解为：

- 不是比较聊天能力；
- 不是比较 RL 后推理能力；
- 而是在比较不同 base model 对相同 held-out 文本的概率建模能力；
- 用 BPB 是为了尽量消除 tokenizer 差异带来的不公平。

### 个人理解

BPB 是评估 base model 预训练质量时很实用的统一标尺。它把不同 tokenizer 下的 token-level NLL 转成 byte-level 成本，让不同模型可以在同一份 held-out 文本上更公平地比较。

在这篇论文里，作者用 BPB 证明 MAI-Base-1 在若干内部保留任务上，相比相近 active parameter 规模的模型有更低的预测成本。这个结果支撑的是“预训练阶段的 base model 已经具备较强分布建模能力”，而不是最终 MAI-Thinking-1 的完整推理能力。

## 2.8.4 节与图 11：MFU 和 EG 是什么，图应该怎么看

论文位置：第 2.8.4 节 Co-optimizing Performance with Model Architecture，图 11。

图 11 里有两个核心指标：

- `MFU`：Model FLOP Utilization
- `EG`：Efficiency Gain

### MFU 是什么

MFU 衡量训练系统实际把硬件理论算力用起来了多少。

论文定义为：

```text
MFU = FLOP / (t_step * FLOP_spec)
```

其中：

- `FLOP`：一个训练 step 中主要计算核产生的 FLOPs，例如 GEMM 和 attention。
- `t_step`：一个完整训练 step 的端到端耗时，包括 data loading、forward、backward、通信和 optimizer update。
- `FLOP_spec`：硬件理论峰值算力。论文里用 GB200 的 FP16/BF16 tensor core 峰值做归一化。

所以 MFU 可以理解为：

> 理论上硬件有 100% 算力，实际端到端训练中被模型有效计算利用了多少比例。

图 11 中 MFU 大约在 `16%` 到 `22%`。这不是说 GPU 只忙了这么点，而是相对于理论峰值 FLOPs 的端到端有效利用率。大模型训练中的通信、同步、memory-bound kernel、CPU launch overhead、MoE routing 等都会拉低 MFU。

### EG 是什么

EG 是 Efficiency Gain，论文第 2.2.2 节定义过。

它衡量的是：

> 候选模型达到某个 eval loss 时，相比 baseline 节省了多少训练成本。

公式是：

```text
EG = baseline 达到同样 loss 需要的 cost / 当前模型实际 cost
```

如果：

```text
EG = 1.40
```

意思是：

> baseline 模型要多花 40% 训练成本，才能达到当前模型同样的 loss。

EG 越高，说明模型架构或训练配方在“质量-计算效率”上越好。

### 图 11 怎么看

图 11 分上下两部分。

上半部分是 `MFU (%)`：

- 看每个模型版本在真实 GB200 训练系统上的硬件利用效率。
- 浅色柱表示新架构刚换上去、还没做该版本专门系统优化时的 MFU。
- 深色柱表示针对该版本做完 kernel、通信、memory、runtime 优化后的 MFU。

下半部分是 `EG over v2`：

- 以 v2 作为 baseline。
- v2 的 EG 是 `1.00x`。
- v3 约 `1.40x`。
- v4 约 `1.69x`。
- v5 仍约 `1.69x`，但模型规模增加到 35B active / 1T total。

图中版本变化大致是：

```text
v2: First GB200 baseline
v3: Dropless MoE
v4: Chips, sparsity, granularity 增加
v5: Model size 增加
```

这张图想表达的是：

> 每次模型架构变强，EG 往往会上升；但新架构会引入新的系统瓶颈，初始 MFU 可能下降。团队需要通过系统优化把 MFU 拉回 20% 左右或以上。

例如：

- v2：初始 MFU 约 `18%`，优化后到 `22%`。
- v3：改成 dropless MoE，EG 提升到约 `1.40x`，MFU 仍能维持约 `22%`。
- v4：专家数、稀疏度、粒度增加，EG 到约 `1.69x`，但初始 MFU 掉到约 `16%`，优化后回到约 `20%`。
- v5：模型规模变大到 35B active / 1T total，初始 MFU 约 `18%`，优化后约 `20%`。

个人理解：

> 图 11 展示的是 MAI-Base-1 的架构和系统协同优化过程：模型设计让 EG 变高，系统优化负责把被新架构拖低的 MFU 拉回来。

## EG 里的 cost 到底是 FLOPs 还是训练时间

论文里的 `cost C` 可以有两种定义，但默认更常用的是 training FLOPs。

第 2.2.2 节说，最常用的是：

```text
cost C = training FLOPs
```

也就是训练计算量，而不是训练时间。

所以默认情况下：

```text
EG = baseline 达到同样 eval loss 需要的训练 FLOPs
     / 当前模型达到这个 eval loss 实际花掉的训练 FLOPs
```

这不直接包括 wall-clock 时间、通信等待、kernel 是否高效、GPU 利用率等系统因素。

如果作者想把训练时间作为 cost，他们会明确叫：

```text
EGTime
```

论文第 2.2.2 节也说明了这一点：默认用 FLOPs；如果关心现有训练栈下的真实硬件效率，则用 time 作为 cost，并称为 `EGTime`。

所以图 11 里的 `EG over v2` 更合理的理解是：

> 相对 v2 baseline，新版本模型在达到同等 eval loss 时，需要的训练 FLOPs 更少了多少。

它不是直接表示：

> 新版本真实 wall-clock 训练时间缩短了多少。

例如：

```text
v3 EG = 1.40x
```

意思是：

```text
v2 baseline 要花 1.40 倍训练 FLOPs，才能达到 v3 当前达到的 loss。
```

但如果 v3 的 MFU 很低，它真实 wall-clock 训练时间未必更短。因此图 11 才同时画 MFU：作者想表达模型计算效率和系统运行效率需要共同优化。

## FLOPs cost 可以怎样拆成架构单位成本和训练量

可以把训练 FLOPs cost 近似理解为：

```text
当前模型实际 FLOPs cost
≈ 每个 token 的训练 FLOPs * 已训练 token 数
```

或者按 step 写：

```text
当前模型实际 FLOPs cost
≈ 每个 step 的训练 FLOPs * 已训练 step 数
```

这两种写法等价，因为：

```text
每个 step 的训练 FLOPs
= 每个 token 的训练 FLOPs * 每个 step 的 token 数
```

因此可以粗略理解成：

```text
FLOPs cost = 架构单位计算成本 * 训练 token 数
```

其中，乘号前面主要由模型和训练配置决定，例如：

- 参数规模
- 层数
- hidden size
- attention 结构
- FFN / MoE 结构
- MoE top-k 激活专家数
- sequence length
- 是否 activation checkpointing / recomputation
- forward、backward、optimizer 的计算定义

乘号后面主要是训练量：

```text
训练 token 数 = step 数 * global batch size * sequence length
```

对于 MoE 要特别注意：total parameters 很大，但每个 token 只激活部分专家，所以每 token FLOPs 更接近 active parameters，而不是 total parameters。

EG 比较的是：

```text
baseline 达到同样效果的 FLOPs cost / 当前模型 FLOPs cost
```

所以一个候选架构可能每 token 更贵，但如果它 loss 下降更快，总 FLOPs cost 仍然可能更省。EG 看的是综合结果，而不是单独看每 token FLOPs。

## 图 11 的实验主要是在调架构还是调数据

限定在图 11，本图主要是在看不同模型版本的架构和系统协同演进，而不是主要在看训练数据量或训练分布变化。

图 11 的版本变化是：

```text
v2: GB200 baseline
v3: Dropless MoE
v4: experts / sparsity / granularity 增加
v5: model size 增加
```

这些主要是模型架构或训练系统相关变化。图 11 下半部分的 `EG over v2` 表示：

```text
相对 v2，这些新架构达到同等 eval loss 所需的 FLOPs cost 更少
```

因此这里主要是在说：

> 改架构以后，模型的“每单位 FLOPs 带来的 loss 下降效率”变好了。

但如果看整篇论文，不能说他们只调架构。论文还有大量数据实验，尤其是：

- 2.4 数据处理
- 2.5 data mixture selection
- 2.5.2 不同 data mix 在小 scale 和大 scale 下排名可能不一致
- 2.5.3 最终数据混合选择
- 2.5.4 mid-training data mixture

这些实验调整的是：

```text
训练数据分布 / 数据质量 / 数据采样权重 / 数据处理策略
```

它们也会影响同样 FLOPs 下的 eval loss，也就是也会影响达到同样质量需要多少 FLOPs。只是图 11 不是在讲这个。

总结：

```text
图 11：主要是架构 + 系统优化带来的 EG / MFU 变化。
2.2 架构消融：主要是不同架构单位计算成本和建模效率。
2.5 数据混合实验：主要是训练数据分布带来的质量和 scaling 变化。
```

## 从小模型实验推广到大模型：scaling ladder 的作用

如果大模型是 1T 级别，小模型是 10B 级别，论文不是简单地认为：

```text
10B 上赢了，所以 1T 上也会赢
```

它的做法是：

```text
小模型实验
-> 多尺度 scaling ladder
-> 拟合 scaling law / 看 EG 是否随 scale 保持
-> 中等或较大规模验证
-> 才进入大模型 baseline
```

### 架构实验如何推广

论文使用 scaling ladder：训练一串规模递增的模型，而不是只训练一个 10B。

关键控制变量是：

```text
TPP = tokens per active parameter
```

也就是每个 active parameter 对应多少训练 token。

架构消融通常在接近 Chinchilla-optimal 的 `100-200 TPP` 做；主训练则是更高的 `500-1000 TPP`，因为他们希望得到更 compact、适合 heavy inference 的模型。

然后对 baseline ladder 拟合 scaling law：

```text
L = f(C) = A * C^(-alpha) + E
```

其中：

- `L` 是 eval loss
- `C` 是训练 cost，通常是 FLOPs
- `A, alpha, E` 是拟合参数

候选架构在某个 cost `C'` 下得到 loss `L'`，再反问：

```text
baseline 要花多少 FLOPs 才能达到同样 L'？
```

然后计算：

```text
EG = baseline 达到同样 loss 的 cost / candidate 实际 cost
```

如果候选架构在多个 scale 上 EG 都稳定大于 1，才说明这个改动更可能推广到更大模型。

因此，对于 `10B -> 1T` 的场景，不能只看 10B，而应该看类似：

```text
1B / 4B / 10B / 20B / 35B active ...
```

这些点上的趋势。如果 10B 有收益，但到 20B 收益消失，就不应该相信这个改动能稳定推广到 1T。

### 数据实验更危险

数据分布实验比架构实验更容易出现 scale-dependent effect。

论文第 2.5.2 节有一个关键例子：他们比较 `stem-heavy-mix` 和 `code-heavy-mix`。

小规模时：

```text
stem-heavy-mix 更好
```

但到 23B active、训练约 20T tokens 时：

```text
code-heavy-mix 反而更好
```

这就是 data mix 的排名反转。论文称这挑战了 `rank invariance hypothesis`，也就是“小模型上的数据混合排序能保持到大模型”的假设。

他们的解释是：某些 STEM 数据源对小模型很有帮助，但重复度更高、内容多样性不足；模型变大、训练更久后，这类数据的边际收益下降，甚至不如更丰富的数据混合。

因此数据实验不能简单说：

```text
10B 上 A 数据 mix > B 数据 mix
所以 1T 上 A 也 > B
```

论文实际做法是：

```text
大量小模型快速搜索 data mix
-> 用 NLL eval suite 打分
-> 找候选 mixture
-> 在更大 scale 做验证
-> 观察 scaling behavior
-> 最后才选 final mix
```

一句话总结：

> 论文不是把小模型结果直接推广到大模型，而是用 scaling ladder 和 scaling law 检查收益是否随规模持续存在；架构实验相对更适合这样外推，数据实验必须额外防止 rank reversal。

## 数据实验外推：从 20B tokens 推到 1T-token midtrain

场景：

```text
最终模型：10B 参数
目标训练量：1T tokens midtrain
当前预算：想先在 20B tokens 上比较不同数据分布、数据质量和数据处理策略
```

这里的目标不是从 `20B tokens` 精确预测 `1T tokens` 的最终分数，而是判断：

```text
不同数据策略在 1T-token midtrain horizon 下，谁更可能最好
```

这类问题最危险的是：短训练时好的 data mix，长训练后不一定好。MAI-Thinking-1 论文第 2.5.2 节的 rank reversal 就是这个问题。

### 不要只跑一个 20B 点

如果最终目标是 `1T tokens`，只在 `20B tokens` 看结果很容易误判。

更稳的是做 token horizon ladder：

```text
20B -> 50B -> 100B -> 200B
```

如果预算允许，再加：

```text
400B
```

不一定每个候选数据策略都跑到 200B 或 400B。可以先用 20B / 50B 筛掉明显差的，再把 top candidates 拉长。

重点不是某个点的绝对值，而是曲线：

```text
eval loss vs training tokens
```

尤其要看不同数据策略的曲线是否有交叉趋势。

### 对每个数据策略拟合 learning curve

可以对每个候选 data mix / data processing 策略拟合：

```text
L(T) = A * T^(-alpha) + E
```

其中：

- `T` 是 midtrain token 数
- `L(T)` 是 held-out eval loss / BPB / NLL
- `alpha` 表示随 token 增加的改善速度
- `E` 表示不可约或近似平台 loss

然后外推到：

```text
T = 1T
```

但这个外推主要用于比较候选策略的相对排序，不要过度相信绝对数值。前提是同一模型规模、同一训练 recipe、同一 eval suite。

### 重点看 slope，不只看 early loss

20B tokens 时最好的 mix，可能只是 early learning 快。

更应该看：

- 谁在 20B 到 100B 之间 loss 下降更快
- 谁的曲线更早变平
- 谁出现过拟合或 held-out loss 反弹
- 谁在目标 domain 上好，但 general eval 掉得很厉害

常见风险是：

```text
高质量小数据 / 高重复数据
```

在 20B 上很好，因为信号密度高；但到 1T 时因为重复太多、多样性不足，边际收益下降。这正是论文里 `stem-heavy-mix` 失败的模式。

### 必须记录 effective epochs

对每个数据 bucket 记录：

```text
unique tokens
sampled tokens
effective epochs = sampled tokens / unique tokens
```

例如某个高质量数学数据只有 `50B unique tokens`，你在 1T midtrain 里计划采样 `300B`，那就是：

```text
6 epochs
```

这个风险在 20B 实验里可能完全看不出来，因为 20B 时还没重复多少。

因此短跑实验也应该使用最终 1T 计划中的采样权重和数据顺序策略，并额外计算如果跑到 1T，每个 bucket 会被重复多少次。

### 评测集要分层

midtrain 数据实验至少要有这些 eval：

- 目标领域 eval：你想提升的能力
- 保留通用 eval：防止灾难性偏移
- 长上下文 eval：如果 midtrain 改 context
- 代码 / 数学 / STEM / 知识分开看
- 训练源近邻 eval：检查是否只是记住某类数据
- 重复敏感 eval：检查高重复数据是否导致泛化变差

总分可以有，但不能只看总分。不同 data mix 的 tradeoff 很强，一个 mix 可能数学好但代码掉，一个 mix 可能目标任务好但 general knowledge 掉。

### 推荐实验流程

一个实用流程：

```text
阶段 A：20B tokens
跑很多候选，筛掉明显差的。
看 early loss、目标 eval、general eval、安全边界。

阶段 B：50B-100B tokens
只跑 top 20%-30% 候选。
开始看 slope、曲线交叉、重复数据收益衰减。

阶段 C：200B-400B tokens
只跑 top 2-4 个候选。
验证排序是否稳定，检查是否有 rank reversal。

阶段 D：选择 final mix 跑 1T
```

### 对 20B -> 1T 要非常保守

`20B -> 1T` 是 `50x` 外推，跨度很大，单点预测风险非常高。

如果只能做 20B 实验，最好只把它当作：

```text
早期筛选信号
```

不要把它当作 1T 最终排序依据。

最低限度建议至少加一个：

```text
100B
```

因为 `20B -> 1T` 是 50x，而 `100B -> 1T` 是 10x，外推可信度会高很多。

### 判断候选策略是否值得上 1T

比较好的候选通常满足：

- 20B 时不一定第一，但在 50B / 100B slope 更好
- 目标 eval 提升明显
- 通用 eval 不明显下降
- 高质量小数据没有过高 effective epochs
- held-out loss 没有变平或反弹
- 不同 eval category 的收益不是靠单一污染或近邻来源撑起来
- 在不同 random seed / data order 下排序稳定

危险信号：

- 20B 很强，但 50B / 100B slope 变差
- 某个小数据 bucket 被重复很多轮
- 目标 eval 好，但 general eval 明显掉
- training loss 继续降，held-out loss 不降
- 不同 scale 下候选排序反复变化

一句话方法论：

> 对 10B 模型的 1T-token midtrain 数据实验，不能用 20B 单点直接外推；应该做 token horizon ladder，拟合每个数据策略的 loss-vs-token 曲线，并用中等长度训练验证排序稳定性，重点防止高质量小数据在长训练下收益衰减和 data mix rank reversal。

## 外部解读 1：训练大模型不是造火箭，是攀岩

来源：<https://yage.ai/share/mai-thinking-1-hill-climbing-20260603.html>

文章标题：Microsoft AI 发布 MAI-Thinking-1 技术报告：训练大模型不是造火箭，是攀岩

这篇文章主要不是在复述 MAI-Thinking-1 的模型指标，而是在提炼论文背后的研发方法论。

核心观点：

> MAI-Thinking-1 报告的真正价值，不是微软做了一个强模型，而是它公开展示了一套顶尖 AI lab 如何做大模型研发决策的“攀爬机器”。

### 主要观点

1. pre-training 太贵，不能靠拍脑袋试错。

MAI-Base-1 使用 8192 块 GB200、30T tokens 级别训练。在这种成本下，每个架构、数据配比、训练 recipe 的选择都必须有实验依据。大模型太贵，不能直接在最终规模上试，所以行业常用小模型实验来筛方案。

2. 小模型实验不能直接外推到大模型。

文章重点讲了论文里的 `stem-heavy mix` vs `code-heavy mix` 例子。

小规模 5B 模型上，STEM 数据占比更高的 `stem-heavy` 在 STEM eval 上更好；但放大到 23B、训练约 20T tokens 后，`code-heavy` 反而超过了它。

这说明：

```text
rank invariance hypothesis 不可靠
```

也就是小模型上 A 胜过 B，不代表大模型上 A 仍然胜过 B。

文章的解释是：高质量但多样性不足的数据，对小模型像“浓缩营养剂”；但大模型学得快，重复几轮后收益耗尽，长训练时反而不如更丰富的数据分布。

3. 要看趋势，不是看单点。

作者认为 MAI 的关键方法论是：不要只在一个规模上看哪个方案好，而要做多尺度实验，看候选方案随 scale 变化的趋势。

也就是论文里的 scaling ladder：

```text
小模型 -> 中模型 -> 更大模型
```

真正值得追的不是“某个 scale 上有用”的 idea，而是：

> 在更大的尺度上依然有用，甚至越来越有用的 idea。

4. EG 把“趋势好不好”变成可计算指标。

文章把 Efficiency Gain (EG) 解释成 MAI 方法论的核心工具。

EG 的意义是：

```text
baseline 要多花多少训练成本，才能达到 candidate 当前达到的效果
```

例如：

```text
EG = 1.3
```

表示 baseline 要多花 30% cost 才能追上 candidate。

作者认为 EG 解决了几个研发问题：

- 不同改动可以统一比较，比如架构、数据、recipe。
- scaling law 的边际收益递减被纳入计算。
- 可以画 EG-vs-cost 曲线，看方案有没有 scaling 后劲。
- 可以区分算法效率和工程效率：`EG_FLOPs` 看理论计算效率，`EG_Time` 看真实训练时间效率。

5. 图 11 展示“创新先让系统变差，再靠工程爬回来”。

文章特别解读了 Figure 11。它认为图 11 的重点不只是 EG 一直上升，而是 MFU 每次架构变化后都会先暴跌，然后通过工程优化爬回来。

例如 v4：

```text
专家数 192 -> 512
routing top-4 -> top-8
引入 Latent MoE
MFU 从 22% 掉到 16%
后续通过 FlashAttention、CPU launch、batching 等优化回到 20%
```

这说明“攀岩”的含义：

> 每一步架构创新都会先制造新的系统瓶颈，让训练效率变差；但如果 scaling ladder 和 EG 告诉你方向是对的，就值得投入工程把 MFU 拉回来。

### 个人理解

这篇文章把 MAI-Thinking-1 论文从“模型报告”重新解释成“研发系统报告”。它关注的不是某个 trick，而是如何构造一个能保护研发团队不被小规模实验误导的系统。

一句话总结：

> 顶尖 AI lab 的竞争，不只是哪个架构 idea 更好，而是谁有一套系统，能快速验证 idea、可靠放大 idea，并在放大过程中持续修复工程瓶颈。

## 外部解读 2：让模型思考不难，让它持续思考才难

来源：<https://yage.ai/share/mai-thinking-1-reasoning-philosophies-20260603.html>

文章标题：Microsoft AI 的 MAI-Thinking-1：让模型思考不难，让它持续思考才难

这篇文章主要解读 MAI-Thinking-1 的推理强化学习训练哲学，并把 MAI、DeepSeek、GLM 的推理路线做了对比。

核心观点：

> 让模型开始推理并不是最难的，真正难的是让推理 RL 连续训练几千步还不崩。

### 推理 RL 为什么容易崩

文章用 GRPO 作为背景：给模型一道题，让它生成一组答案，对答案打分，再根据相对好坏调整模型生成概率。

GRPO 的问题是，训练久了模型容易走向两个极端：

- 熵坍塌：模型过度自信，反复输出相似答案，失去探索能力。
- 策略发散：模型突然输出混乱内容，训练崩溃。

文章认为 MAI 的关键贡献不是“让模型会想”，而是为长时间推理 RL 建立稳定性纪律。

### 三个机制

1. 自适应熵控制：像恒温器一样控制模型的“自信度”。

如果模型 entropy 下降、变得太死板，就放宽更新上限，让它更敢探索；如果 entropy 上升、开始乱猜，就收紧上限，让它稳定下来。

文章把它比作恒温器：

> 模型太死板就放开一点，太发散就收紧一点。

2. outer ratio clip：给 GRPO 没管住的角落加硬上限。

GRPO 原本有些更新方向不裁剪，因为这些方向看起来像“模型在修正错误”。但 MAI 发现，这些不受约束的角落偶尔会造成极端梯度 spike。

MAI 的做法是在已有裁剪之外加一个绝对上限：

```text
正常情况下碰不到，但极端情况下能截断灾难性更新
```

文章把它比作断路器：

> 平时什么都不做，一旦出现极端情况，先切断电路保护训练。

3. 自蒸馏：训练崩了之后抢救进度。

前两个机制能减少崩溃，但不能彻底消灭崩溃。MAI 接受“偶尔会崩”这个事实，设计自蒸馏流程：

```text
定期保存当前模型成功推理 traces
如果 RL run 崩溃
就用这些 traces 训练一个干净 checkpoint
再从新 checkpoint 继续 RL
```

文章强调，MAI 发现大约 100 万条成功推理记录就足以让新模型接近旧模型水平，更多数据收益递减，甚至可能压窄探索空间。

### 与 DeepSeek 和 GLM 的路线对比

文章认为 2026 年几家模型都在做推理 RL，但解决的问题不同。

MAI 的瓶颈是训练稳定性：

```text
目标：让 RL climb 几千步不滑下来
方法：entropy control + outer ratio clip + self-distillation
```

DeepSeek 的瓶颈是计算效率：

```text
目标：让百万 token 上下文推理训练变得可承受
方法：压缩注意力，把单 token FLOPs 和 KV cache 成本大幅降下来
```

GLM 的瓶颈是跨轮次持久性：

```text
目标：multi-turn agent 不要每轮都重新推导过去上下文
方法：Preserved Thinking / Interleaved Thinking
```

文章的比喻是：

```text
MAI：保证引擎不熄火
DeepSeek：让引擎跑得更快
GLM：让引擎记住上一次的路线
```

### 个人理解

这篇文章把 MAI 的 RL 部分解释成一套“训练稳定性工程”。它关注的不是单次推理能力有多强，而是推理能力能否在长时间 RL 中稳定累积。

一句话总结：

> 推理模型竞争的重点正在从“能不能思考”转向“能持续思考多久、能在多大上下文里思考、能不能跨多轮延续思考”。

## Pre-train / mid-train 的 NLL eval 数据要求

问题：MAI 这种 pre-train / mid-train 的 NLL eval，对 eval 数据要求是不是很高？论文有没有说怎么构造或挑选？

答案：要求很高。NLL eval 的形式比 Q&A / generation benchmark 简单，但 eval 数据本身仍然必须干净、私有、去污染、能代表目标能力。

### 论文里的 eval 来源

论文第 2.3 节说，他们的 NLL eval suite 来自三类来源：

- vendor 专门创建、完全 held-out from training 的数据
- Microsoft 内部、不在公网的数据
- public/web 来源，但会小心地从训练数据中移除

论文还强调：

```text
All examples are carefully deduplicated from any training data.
```

也就是每个 eval 样本都要和训练数据做去重 / 去污染。

### 不依赖公开 benchmark 做日常 pre-train eval

论文第 2.3.1 节说，公开 benchmark 很容易泄漏进训练集，尤其 GitHub 上有大量 benchmark 原题、答案、模型生成结果和镜像副本。

他们做了几类去污染：

```text
移除 huggingface.co 和镜像域名数据
全训练源做 universal 20-gram fuzzy dedup
相似度阈值 80%
```

但他们也承认这些方法不完美，所以构建了 private benchmarks，尽量确保：

```text
not found elsewhere on the web
```

### 覆盖面要宽，不能只看单个总集

他们有接近 40 个内部 NLL benchmarks，分成 5 类：

- Code
- STEM
- Math
- General Knowledge
- Multilingual

表 3 给的例子包括：

- Code：Microsoft code / pull requests、Human-AI coding sessions
- STEM：vendor commissioned 的 graduate-level STEM worked solutions
- Math：vendor commissioned 的 advanced math worked solutions
- General Knowledge：online community discussions、human-AI interactions、dedup 后的 public trivia、vendor hard trivia
- Multilingual：multilingual Human-AI interactions

这说明 eval 数据不是随便拿一批网页文本，而是按目标能力维度构造，并且每类有多个 benchmark。

### NLL eval 比 Q&A benchmark 容易在哪里

论文第 2.3.2 节说，高质量 Q&A eval 需要问题设计、难度校准、去重、质检和专家审核，成本很高。

而 NLL eval 的门槛低一些：

```text
Any topic-relevant content can serve as an initial corpus.
```

也就是说，只要是目标领域相关、质量高、held-out 的文本，就可以作为 NLL eval 的初始语料。后续可以通过增加结构、删除不代表性样本、加入更高级材料、按类别分层来提高质量。

但这不代表 NLL eval 可以随便构造。它只是：

```text
不需要把每条样本做成问答题，也不需要 judge；
但必须保证数据干净、代表目标能力、没有训练污染。
```

### Long-context mid-train eval 的构造

附录 B.2 对 long-context eval 说得更具体。

Code NLL：

- 把内部代码仓库串成线性 token stream
- 抽取 256K-token chunks
- 最后 16K tokens 作为 suffix 算 NLL
- 前面的 prefix 从 16K、32K、64K、128K、256K 逐步增加
- 看更长 prefix 是否能降低 suffix NLL

Retrieval NLL：

- 选一个 32K-token 内部文档
- 切成两个 16K blocks：B 是 related prefix，A 是 suffix
- 在 B 和 A 中间插入越来越多 irrelevant chunks
- 把总 context 拉到 256K
- 看模型是否还能用远处相关信息降低 A 的 NLL

Generative QA：

- 用内部 repository documents 构造 QA
- 把 evidence 放在不同上下文位置
- 测 answer accuracy 随 context length / evidence position 的变化

所以 mid-train 的 long-context eval 不只是拿长文本算 loss，而是刻意测试“远距离相关信息是否可用”。

### 对 eval 数据的实际要求

如果要复现类似方法，eval 数据至少要满足：

- held-out：不能进训练集
- decontaminated：和训练集做 exact / fuzzy / n-gram 去重
- private 或难以在公网找到：避免 benchmark leakage
- domain-representative：覆盖关心的能力领域
- high quality：文本正确、干净、有足够信息密度
- category-balanced：不能只靠一个总集判断
- stable：每次实验使用同一套 eval，保证可比性
- sensitive：小改动能在 NLL 上反映出来，但不能被噪声主导

个人理解：

> MAI 选择 NLL eval 是为了让评测更便宜、更稳定、更适合高频实验；但 eval 数据本身仍然需要私有、干净、去污染、分领域、高质量，否则 NLL 再便宜也会把研发方向带偏。

## 数学 NLL 低是否代表最终能答对题

问题：`sensitive` 这个要求很难。尤其数学答题类数据，NLL 低不代表最后能答对题。MAI 的 NLL eval 是不是要结合后续 RL 才成立？

答案：是的。NLL eval 是 pre-train / mid-train 阶段的代理指标，不是最终 reasoning ability 的充分指标。

更准确地说：

> NLL eval 用来衡量 base / mid-trained model 是否吸收了有利于后续 RL 的知识和解题文本分布；最终能不能答题，还要靠 RL / post-training 和生成式 benchmark 验证。

### 数学为什么也能用 NLL eval

论文里的数学 / STEM NLL eval 不是只放：

```text
题目 -> 最终答案
```

而是更像：

```text
worked solutions to advanced math problems
worked solutions to graduate-level STEM problems
```

也就是包含完整解题过程的文本。

模型在这些 held-out worked solutions 上 NLL 低，说明它更能预测：

- 数学语言
- 解题步骤
- 公式变换
- 证明 / 推导结构
- 中间 reasoning pattern
- 最终答案格式

这比只预测一个选项或最终答案更贴近 pre-train 的 next-token objective。

### 但 NLL 不等于会独立解题

NLL eval 是 teacher forcing：

```text
真实前缀 -> 预测下一个 token
```

真实答题是 free generation：

```text
自己生成前缀 -> 继续生成
```

teacher forcing 下，每一步都条件在真实前缀上；自由生成时，中间一步错了，后面会连锁崩。因此数学 NLL eval 测的是：

```text
模型是否具备建模数学解题文本分布的能力
```

不是直接测：

```text
模型是否能独立解决数学题
```

它更像测“原材料吸收”和“潜在能力基础”，不是最终产品能力。

### 为什么 pre-train / mid-train 仍然用 NLL

原因包括：

- 生成式数学 benchmark 太贵：要采样、长 CoT、判分，可能还要 judge。
- base model 还不是 instruction-following model：可能知道数学，但不会按 benchmark 格式回答。
- 生成式 eval 噪声大：prompt、格式、temperature、答案抽取都会影响结果。
- NLL eval 和训练目标一致：pre-train 本来就是 next-token prediction，所以 NLL 对训练改动更敏感。

所以 NLL eval 的角色是：

```text
低成本、高频、低噪声的开发信号
```

而不是最终能力证明。

### 和后续 RL 的关系

可以把 MAI 的逻辑理解为：

```text
pre-train / mid-train NLL
-> 选择更适合后续 RL 爬坡的 base foundation
-> RL 把潜在能力转成可生成、可验证、可得分的推理行为
```

论文里也说，pre-train 和 mid-train 提供 broad predictive competence and knowledge，但不决定模型如何行为、如何解决长程任务、如何分配推理时计算。后面 RL 才教模型：

- 生成 chain of thought
- 根据任务反馈调整推理
- 使用工具
- 遵守偏好和安全信号
- 在长 horizon 任务里持续行动

### 代理指标失真的风险

NLL 很低的数据策略也可能伤害 RL，例如：

- 过多重复解题模板导致探索变窄
- 数据过于接近 benchmark 导致污染
- 只有 polished solutions，缺少错误修正 / 探索轨迹
- 泛化差，但 held-out NLL 看起来好
- short solution NLL 好，但长链推理能力不一定好

因此要做 correlation validation：

```text
pre-train / mid-train NLL 改善
是否能预测后续 RL 后 benchmark 改善？
```

可以维护几层指标：

```text
Level 1: NLL eval
高频、便宜，用于日常筛选。

Level 2: 小规模 generative eval
低频一点，用于确认 NLL 方向是否合理。

Level 3: post-RL eval
更贵，用于验证这个 pre-train / mid-train 改动是否真的帮助最终模型。

Level 4: full downstream benchmark / human eval
最贵，用于最终确认。
```

如果某个 NLL eval 长期和 Level 2 / 3 / 4 不相关，甚至反相关，就应该降权或移除。

### 数学 NLL eval 的构造建议

如果要做数学 NLL eval，不建议只放最终答案。更好的形式是：

```text
题目 + 完整 worked solution
```

并且分层：

- 短解题
- 长证明
- 竞赛数学
- 大学数学
- 符号推导
- word problem
- 多步计算
- 容易走错的 distractor 类型
- 不同解法风格

同时保留生成式数学 eval 作为校验，例如：

```text
AIME / MATH / internal Olympiad / proof QA
```

一句话总结：

> MAI 的数学 / STEM NLL eval 不是在声称“NLL 低就会答题”，而是在用低成本指标衡量 base / mid model 是否吸收了有利于后续 RL 的数学解题分布；它必须通过后续生成式 benchmark 和 RL climb 表现来校准，否则代理指标很容易失真。

## MAI-Thinking-1 到底有没有做 SFT

问题：MAI-Thinking-1 没有做 SFT 吗？只用 RL 就可以做 instruction following 和生成 CoT 吗？

答案：不是。MAI-Thinking-1 做了 SFT，而且不止一次。只是它强调：

> 初始 reasoning RL climb 不是从一个已经看过 reasoning traces 的 SFT reasoner 开始，而是从 mid-trained checkpoint 开始，通过 RL 让模型发展推理能力。

更准确的流程是：

```text
Pre-train
-> Mid-train
-> 三条 specialist RL climbs
   - STEM / competitive coding
   - agentic coding / tool use
   - helpfulness / safety
-> self-distillation SFT / trace distillation SFT
-> consolidation SFT
-> final lightweight RL
-> MAI-Thinking-1
```

### 初始 reasoning climb 没有 prior reasoning traces

论文第 3 章说：

```text
RL climb starts from a checkpoint with no prior exposure to reasoning traces
```

意思是：最初 STEM / reasoning RL climb 的起点不是一个已经用 CoT SFT 过的模型。模型需要在 RL 里从任务反馈中发展 reasoning traces。

但这不等于模型没有格式引导。Figure 14 给了初始 prompt template：

```text
Assistant first thinks...
<think> reasoning process here </think>
<answer> answer here </answer>
```

也就是说，早期用 prompt 诱导模型按照 `<think>` / `<answer>` 格式输出，然后通过 RL reward 推动它生成更有效的 reasoning。

### Self-distillation 本质就是 SFT

论文第 3.1.4 节明确说：

```text
collect rollouts generated during RL
perform SFT on a midtrained checkpoint using these rollouts
```

也就是：

```text
RL 产生 reasoning traces
-> 收集成功 / 高质量 traces
-> 对 mid-trained checkpoint 做 SFT
-> 得到更稳定的新起点
-> 继续 RL
```

这叫 self-distillation，但训练形式就是 SFT。

用途包括：

- 从 raw prompt template 迁移到 native chat format
- RL run 崩溃后恢复进度
- 把旧 checkpoint 的能力迁移到新 pre / mid-trained checkpoint
- 过滤 reward hacking 样本
- 保留之前 RL climb 发现的能力

所以 CoT 最初可以由 RL 发展出来，但后面会通过 self-distillation SFT 固化、迁移和稳定化。

### 最后三个 specialist 还做 consolidation SFT

论文第 3.5 节说，他们训练了三个 teacher：

```text
STEM / coding teacher
agentic teacher
helpfulness & safety teacher
```

然后：

```text
The SFT stage distills the three teachers into a single model.
```

也就是用 SFT 把三个 specialist teacher 的能力合并到一个 consolidated model。

Table 10 给了 consolidation SFT 的数据比例：

```text
STEM and Coding: 56% sample weight, 89% token weight
Agentic Capability: 11% sample weight, 9% token weight
General Helpfulness and Safety: 33% sample weight, 2% token weight
```

然后再做 final lightweight RL，得到 MAI-Thinking-1。

### Instruction following 不是只靠 STEM RL 自然出现

helpfulness and safety climb 是单独一条 RL climb，目标包括：

```text
human preference
instruction following
steerability
safety
honesty
style
```

它用了多种 reward：

- reward model
- AI judge
- rule-based / verifiable rewards

其中 instruction following 数据来自：

```text
expert-written contexts
synthetic data
```

还包括 system / developer / user 指令层级冲突、多语言、多轮场景、40+ domains 等。

因此指令遵循不是“只靠数学 RL 顺便学出来”。它有专门的数据、reward 和训练阶段。

一句话总结：

> MAI 的路线不是“无 SFT 纯 RL”，而是“先让 reasoning 在 RL 中从零爬起来，再用 self-distillation SFT 和 consolidation SFT 稳定、迁移、合并能力，最后再轻量 RL 对齐”。

## 从任务反馈中长出来的 CoT 是否会很冗长

问题：从任务反馈中发展 reasoning traces，CoT 会不会很长、有很多 reflection、不够精炼，而且可读性和专业性不稳定？

答案：是的，按论文描述，早期从 RL 里长出来的 reasoning traces 很可能会出现这些问题：

```text
长
反复 reflection
绕圈
hedging
模板化自检
可读性不稳定
专业表达不一定好
```

论文没有展示很多早期 CoT，但它的训练设计本身说明他们遇到了这些问题。

### Length penalty 说明 RL 会倾向变长

论文第 3.1.2 节有 length penalty，目的就是防止模型为了拿 reward 生成冗余长推理。

策略大致是：

- 简单题：更强 length penalty，鼓励简洁。
- 难题：允许更长 reasoning。
- 到 128K 阶段：移除 length penalty，让模型在复杂问题上充分探索。

这说明如果不控制，RL 确实容易学出很长、很啰嗦的 CoT。

### 论文明确提到 redundant loops 和 hedging behavior

论文说 length penalty 鼓励 concise and cost-efficient reasoning by removing：

```text
redundant loops
hedging behavior
```

这就说明 RL 产生的 CoT 可能会反复检查、绕圈、犹豫，甚至用“也许、可能、让我再确认”这类形式消耗 token。

### Self-distillation 会过滤 degenerate CoTs

论文第 3.5 节 consolidation SFT 里说，对 STEM 和 agentic teachers 的 rollouts：

```text
apply light filtering to remove degenerate CoTs
```

这说明不是所有 RL traces 都能直接拿来蒸馏。会有退化的、格式差的、循环的、不适合保留的 CoT。

### RL reward 不一定直接优化 CoT 可读性

STEM / coding RL 的 reward 多数是结果导向：

```text
答案对不对
代码能不能通过
judge 给不给高分
```

这类 reward 会优化“能解题”，但不一定优化：

```text
推理过程是否优雅
是否像人类专家
是否结构清晰
是否教学友好
是否最短
```

所以 CoT 的专业性和可读性如果没有额外 reward / filtering / SFT，很可能不稳定。

### 最终用户看到的未必是原始 CoT

MAI-Thinking-1 后面还有：

```text
self-distillation SFT
consolidation SFT
helpfulness & safety climb
style reward
final lightweight RL
```

这些阶段会对格式、风格、帮助性、安全性做调整。

尤其 helpfulness / safety climb 里有 style guide、reward model、LLM judge 等，目标包括：

```text
style
instruction following
steerability
helpfulness
honesty
```

所以最终可见回答的质量，不完全取决于早期 RL 里原生态 CoT 的样子。

### 个人理解

这条路线更像：

```text
RL 先找“能解题的轨迹”
SFT / filtering 再筛选和固化“相对稳定的轨迹”
helpfulness / style / final RL 再把外部回答打磨成人能用的形式
```

它不是一开始就追求漂亮 CoT，而是先追求：

```text
reward 能推上去
问题能解出来
能力能持续爬坡
```

然后再处理：

```text
冗余
格式
风格
可读性
安全性
```

关键风险包括：

- 结果正确但 reasoning 过程很丑
- CoT 里有无效 reflection
- 模型学会“看起来在思考”的模板
- 长 CoT 增加推理成本
- 蒸馏后可能把低质量 reasoning style 固化进去
- 过度过滤又可能损失探索多样性

一句话总结：

> 纯靠任务反馈长出来的 CoT 通常不天然精炼、可读或专业；MAI 的做法是先用 RL 发展可得分的 reasoning，再通过 length penalty、过滤、self-distillation、consolidation SFT、helpfulness/style RL 去控制冗余和风格。

## 为什么 MAI 在 pre-train / mid-train 尽量去除合成数据

问题：为什么 MAI-Thinking-1 的 pre-train 和 mid-train 要把合成数据都尽量去除？例如 mid-train 阶段加入 instruction following 合成数据、答题合成数据，可能可以直接提升 post-train 下游任务效果。

答案：MAI 不是全流程排斥合成数据，而是把合成数据主要放在 post-training / RL 阶段，而不是放进 pre-train / mid-train。

论文里的区分是：

```text
pre-train / mid-train:
尽量不用 LLM 生成的 synthetic data，也尽量去除 AI-generated content

post-train / RL / IF / safety / agentic:
会使用 synthetic data、synthetic environments、AI judge 等
```

### 1. 他们想让 base model 从人类知识中学能力，而不是继承其他模型行为

论文的核心原则之一是：

```text
capabilities should be learned, not inherited
```

他们认为从第三方模型蒸馏或吃大量 AI 生成内容，虽然能更快获得能力，但可能损害长期 RL climb 所需的：

```text
steerability
robustness
探索空间
```

如果 base / mid 阶段已经大量学习某个 teacher model 的表达习惯、CoT 模板、拒答风格和错误模式，后面 RL 不是在“从人类知识中学习”，而是在修正另一个模型留下的分布偏置。

### 2. Pre-train / mid-train 的目标不是 instruction following

论文说，pre / mid-training 给 base model 提供：

```text
broad predictive competence and knowledge
```

但不指定模型：

```text
how to behave
how to solve long-horizon tasks
how to allocate inference-time computation
```

这些行为层面的东西，他们放到 RL climb / helpfulness safety climb 里教。

所以如果 mid-train 加大量 instruction-following 合成数据，短期可能提升下游 benchmark，但会混淆 mid-train 的定位：

```text
mid-train: 强化知识、代码、数学、长上下文 foundation
post-train: 教行为、格式、指令遵循、偏好、安全
```

### 3. 合成答题数据可能压窄后续 RL 探索空间

如果 mid-train 加很多合成 CoT / 答题轨迹，模型可能更早学会某种固定推理风格。短期看 downstream task 会涨，但后续 RL 可能遇到问题：

- 过早固化 teacher 的解题模板
- 输出分布变窄，探索能力下降
- 学到 teacher 的错误模式和 hallucination
- CoT 看起来像推理，但其实只是 imitation pattern
- 后续 RL 更难发现新的策略

论文 self-distillation 部分也有类似观察：self-distillation 数据太多会 over-constrain policy，narrowing its output distribution，导致 RL resume 后探索空间变小。

### 4. 合成数据会污染 scaling 实验信号

MAI 很依赖 NLL eval、data mix ablation、scaling ladder 来判断数据和架构改动。

如果 pre / mid 数据里混了大量模型生成内容，问题会变复杂：

```text
收益来自真实知识？
还是来自 teacher style？
还是来自 benchmark-like pattern？
还是来自合成数据重复模板？
```

这会让 scaling law / EG / data mix attribution 更难解释。

他们的 hill-climbing machine 需要清晰的因果信号。干净、可追踪的人类数据更适合做 pre-train / mid-train 的系统级优化。

### 5. 合成数据容易引入重复和低多样性问题

合成数据常见问题：

- 模板重复
- 风格单一
- 过度解释
- 常见 reasoning pattern 过密
- tail knowledge 缺失
- teacher model 偏见和错误复制

mid-train 是 trillion-token 量级，如果合成数据比例大，重复和低多样性风险会被放大。

### 6. 数据治理更简单

论文强调 clean、enterprise-grade、licensed / public human-generated data。对 Microsoft 这种机构，pre-train 数据治理很重要。

合成数据还要追踪：

- teacher model 来源
- 生成数据许可
- 是否包含第三方模型输出限制
- 是否隐含训练集泄漏
- 是否含 benchmark contamination
- 是否复制了模型安全 / 偏见问题

把 synthetic data 留到 post-training 的特定任务数据里，更容易审计和控制。

### 为什么后训练又用 synthetic data

因为 post-training 的目标不同。

比如 instruction following 需要覆盖大量约束组合：

```text
system / developer / user 冲突
多轮对话
多语言
格式约束
40+ domains
边界场景
```

人工写很贵，synthetic data 很适合做覆盖扩展。

论文第 3.4.2 节说，instruction following 用了：

```text
expert-written contexts
synthetic data
```

并且他们认为：

```text
expert-written prompts help bootstrap capabilities
synthetic data enables maximum coverage
```

也就是说：

- 人工专家数据：打底，保证质量和复杂约束。
- 合成数据：扩覆盖，制造大量场景。
- RL reward / judge / verifier：负责筛和优化行为。

一句话总结：

> MAI 不是认为合成数据无用，而是把它从 pre-train / mid-train 中移出去：base / mid 阶段尽量只学干净的人类知识和长上下文基础能力，避免继承 teacher 模型的分布偏置；指令遵循、答题格式、CoT、偏好和安全则放到 RL、self-distillation SFT、consolidation SFT 和 helpfulness/safety climb 中处理。

## 合成数据放进 mid-train 是否可能造成 teacher 污染

问题：所以合成数据放到 mid-train 阶段，可能“投毒”，例如某个 teacher 模型的输出里有毒，是这个意思吗？

答案：可以这么理解，但这里的“投毒”不一定是恶意投毒，也包括更广义的分布污染、行为污染和偏置继承。

合成数据放进 mid-train 的风险是：

> 它会被当成 foundation 的一部分学进去，而不是像 post-train 那样在较小、更可控的行为数据阶段里调整。

### 可能的问题

1. teacher 的错误被固化。

如果 teacher 在某些领域会 hallucinate、推理跳步、公式错误、代码习惯差，mid-train 会把这些模式学进 base / mid model。

尤其是 worked solution / CoT 数据，表面很流畅，但可能有隐蔽错误。NLL 训练不会知道它错，只会学它的分布。

2. teacher 的风格污染。

例如 teacher 特别喜欢：

```text
过度解释
反复 reflection
模板化 “let's verify”
长篇套话
过度 hedging
特定拒答风格
```

这些会进入模型基础分布。后面 RL 可以修，但等于先把坏习惯种进去，再花算力纠正。

3. teacher 的安全 / 拒答偏置被继承。

如果合成数据来自对齐过的模型，它可能有过度拒答、规避回答、特定安全策略、过度谨慎等行为。放在 mid-train 会让 base model 更早带上这些行为倾向。

MAI 的思路是：pre / mid 不定义“模型应该怎么行为”，行为放到 helpfulness / safety climb 里明确优化。

4. benchmark-like pattern 污染。

teacher 生成的数学 / 代码 / QA 数据可能很像公开 benchmark 的风格，甚至 teacher 记住过某些 benchmark。这样 eval 提升看起来很好，但实际可能是污染或风格贴近，不一定是真泛化。

5. 重复模板和低多样性污染。

合成数据经常高重复：

```text
题目结构重复
解法模板重复
语气重复
推理步骤重复
答案格式重复
```

短期训练很有效，但长 horizon 可能导致收益衰减、探索变窄，甚至影响后续 RL。

6. 后续 RL 探索空间变窄。

如果 mid-train 已经大量模仿某个 teacher 的 CoT，模型的输出分布会更像 teacher。后续 RL 不是从宽分布探索，而是在 teacher 风格附近微调。

论文 self-distillation 部分也提到类似风险：蒸馏数据太多会 over-constrain policy，narrow output distribution，影响 RL resume 后的探索。

### 不是说合成数据一定有毒

高质量、可验证、去重、来源可控的合成数据也可能很有用。只是 MAI 的取舍是：

```text
pre / mid 阶段宁愿保持 foundation 干净
post / RL 阶段再引入更可控的 synthetic data
```

因为 post-training 阶段：

- 数据量相对小，更容易审计
- reward / judge / verifier 可以过滤
- 失败影响更局部
- 可以针对 behavior 做修正
- 不会像 mid-train 那样深度改变 base distribution

一句话总结：

> 合成数据进 mid-train 的一个核心风险是 teacher 输出里的错误、风格、安全偏置、模板重复和 benchmark contamination 会被基础模型吸收并固化；MAI 选择把合成数据主要放到后训练，是为了让这种风险更可控。

## RL-first 的 reason trace 和专家 CoT SFT 的关系

问题：MAI-Thinking 的 reason trace 是不是纯靠 RL reward 从 foundation model 里激发出来的？那某个领域任务的专家 CoT SFT 还有用吗？

答案：MAI 初始 reasoning traces 主要是靠 RL reward 从 mid-trained foundation model 里激发 / 搜索出来的，但不是完全无引导。

更准确是：

```text
mid-trained foundation model
+ reasoning prompt template
+ task-specific reward
+ GRPO
=> 逐渐产生可得分的 reasoning traces
```

论文强调的是：

```text
no prior exposure to reasoning traces
```

意思是：初始 reasoning climb 不是先喂一批第三方专家 CoT SFT 让模型学会推理格式和策略，而是从 RL 中发展出来。

### 专家 CoT SFT 仍然有用

专家 CoT SFT 的价值主要在：

1. 冷启动。

如果 foundation model 太弱，RL 很难采到正样本。比如模型对某领域几乎不会做题，reward 全是 0，GRPO 没信号。

专家 CoT SFT 可以把模型带到一个能探索的区域：

```text
完全不会 -> 偶尔能做对 -> RL 有正负样本可学
```

2. 格式和任务协议。

例如医学、法律、代码审查、定理证明、工具调用，需要特定输出结构和专业流程。专家 CoT 可以教：

```text
先列条件
再引用依据
再做推导
最后给结论
```

这类 protocol 靠 RL 自己摸索会很慢。

3. 稀缺高难领域。

有些任务 reward 很贵或难以自动验证，例如开放式科研分析、复杂诊断推理、长文法律论证。专家 CoT 可以提供高密度先验。

4. 可读性和专业表达。

RL reward 常常只看结果对不对，不看推理过程是否专业、简洁、可审计。专家 CoT 可以教“好看的推理过程”。

### 专家 CoT SFT 的风险

这也是 MAI 不从第三方 CoT 蒸馏启动的原因。

风险包括：

```text
teacher 错误被继承
推理风格被固化
探索空间变窄
模型学会 imitation 而不是 task-grounded reasoning
CoT 很漂亮但不一定真实因果有效
后续 RL 被 teacher policy 限制
```

如果 SFT 数据太多、太单一，模型会更像 teacher，而不是自己通过 reward 找策略。

论文 self-distillation 部分也说过类似现象：太多 traces 会 over-constrain policy，narrow output distribution，影响后续探索。

### 合理位置

专家 CoT 更稳的用法可能是：

```text
1. 少量专家 CoT 做冷启动
2. 只教格式和高层策略，不灌太多完整模板
3. 用 verifier / reward 过滤 CoT 正确性
4. 后续必须接 RL，让模型用任务反馈修正专家偏差
5. 保持 prompt diversity，避免输出分布过窄
6. 不把第三方 teacher CoT 无节制放进 foundation mid-train
```

MAI 的路线更偏：

```text
先不用第三方专家 CoT
让 RL 自己爬出 reasoning traces
再用 self-distillation SFT 固化“自己 RL 找到的轨迹”
```

两种路线的取舍：

```text
专家 CoT SFT:
+ 冷启动快
+ 格式和专业表达好
+ 小模型 / 弱模型更有帮助
- 容易继承 teacher 偏差
- 探索空间变窄
- 可能过拟合漂亮但无效的推理模板

MAI 式 RL-first:
+ 能力更 task-grounded
+ 少继承第三方 teacher 偏差
+ 更符合长期 climb 和可控性
- 冷启动更难
- 需要强 foundation
- 需要大量 RL infra 和 reward
- 早期 CoT 可能冗长、丑、不稳定
```

一句话总结：

> MAI 的 reasoning trace 初始主要靠 RL reward 从 foundation model 中激发出来；专家 CoT SFT 仍然有用，但更适合作为受控冷启动、格式教学或领域先验，而不宜无节制放进 mid-train 或大量蒸馏，否则可能限制后续 RL 的探索和泛化。

## 如果领域任务不做 RL，mid-train 放合成 reason trace 是否可行

问题：如果领域任务不做 RL，例如 math 解答题学生批改任务，希望用 mid-train 合成 reason trace，然后做 SFT，这样在 mid-train 阶段放大量合成数据可行吗？

答案：可行，而且对这个场景可能是合理路线。MAI 避免 synthetic mid-train，是因为它的目标是训练通用 foundation + 后续大规模 RL climb。领域任务如果不做 RL，取舍不同。

场景：

```text
领域任务：math 解答题学生批改
训练路线：不做 RL
目标：让模型学会批改、解释、给反馈
```

这时合成 reason trace / 批改轨迹放到 mid-train 或 continued pretraining 里，可能很有价值。

### 推荐路线

```text
Base model
-> Domain continued pretraining / mid-train
-> Task SFT
-> Preference / rejection sampling SFT optional
```

### Domain mid-train 放什么

mid-train 更适合放领域语料和解题 / 批改相关文本分布，例如：

- 教材
- 习题解析
- 标准答案
- 多种解法
- 学生常见错误分析
- 批改说明
- rubric
- grading guideline
- 数学符号、LaTeX、中文数学表达
- 高质量 worked solutions
- 少量到中等量合成 reason traces

目标是让模型熟悉：

```text
数学知识
题型结构
学生答案风格
评分点
错误类型
批改语言
```

### Task SFT 放什么

SFT 更适合放真正任务格式：

```text
输入：题目 + 标准答案/评分标准 + 学生作答
输出：分数 + 错误定位 + 扣分理由 + 修改建议
```

例如：

```text
题目：
...

标准解：
...

学生答案：
...

请按 10 分制批改，指出每一步是否正确。
```

输出：

```text
得分：7/10

正确部分：
...

错误部分：
...

扣分说明：
...

建议：
...
```

这类 instruction-following 和格式约束，放 SFT 比放 mid-train 更直接。

### 合成数据可以大量放，但要控质量

核心风险不是“不能用合成数据”，而是：

```text
合成批改 trace 质量是否可靠
```

数学批改任务尤其危险，因为模型生成的批改理由可能看起来专业，但其实：

- 算错
- 误判学生步骤
- 给分不一致
- 漏掉等价解法
- 对非常规解法过度扣分
- 编造学生没有写的错误
- 评分标准漂移

如果把这些大量放进 mid-train / SFT，模型会稳定学坏。

### 比较稳的合成数据策略

1. 用强 verifier 或规则校验生成数据。

对能程序化验证的题，尽量用 sympy、数值代入、单元测试、答案等价判断。

2. 合成“学生错误”比合成“正确专家解”更有价值。

批改模型需要见很多错误类型：

```text
符号错误
漏步骤
计算错误
概念错误
单位错误
等价变形错误
证明逻辑断裂
答案对但过程错
过程对但最终算错
```

3. 合成数据要多样化。

不要只让一个 teacher 用同一种模板生成。要控制：

```text
题型
年级
难度
解法风格
学生答案长度
错误类型
批改语气
评分尺度
```

4. 合成数据要分层混入。

不建议一上来 90% synthetic。可以做 ablation：

```text
0%
10%
30%
50%
70%
```

看 held-out 人工批改集上的效果。

5. 保留人工 gold eval。

最关键的是要有一套真实人工标注 eval：

```text
题目 + 学生答案 + 人类老师评分 + 批注
```

否则 synthetic train + synthetic eval 容易自嗨。

### mid-train vs SFT 的比例建议

如果 base model 本身数学和中文能力还可以，更倾向：

```text
mid-train：少量到中等量 domain corpus，保持通用能力
SFT：高质量批改任务数据，强调格式和评分一致性
```

而不是把大量 instruction-style 合成 trace 都塞进 mid-train。

一个起始配比可以是：

```text
Domain mid-train:
70% 高质量真实数学 / 教材 / 解析 / 评分规范
20% 合成 worked solutions / 错误分析
10% 合成批改风格文本

Task SFT:
50% 人工或半人工批改样本
30% verifier 过滤过的合成批改样本
20% edge cases / adversarial student answers
```

最终要靠 eval ablation 调。

### 不做 RL 时，专家 CoT / SFT 更重要

MAI 能少用专家 CoT，是因为后面有强 RL reward 去发现和筛选推理路径。不做 RL，则需要靠：

```text
高质量 SFT 数据
严格 filtering
人工 eval
rejection sampling
多模型交叉审查
```

来替代 RL 的选择压力。

可以做一个轻量替代：

```text
生成 N 个批改输出
用规则 / verifier / 强 judge / 人工抽检打分
只保留高质量样本做 SFT
```

这不是 RL，但能提供类似“筛选压力”。

一句话总结：

> 对数学解答题批改任务，如果不做 RL，在 mid-train 阶段加入经过严格过滤的合成 reason trace 是可行的；但真正的批改格式、评分一致性和反馈风格最好主要靠 SFT 学，且必须用真实人工批改 eval 来防止合成数据把模型带偏。
