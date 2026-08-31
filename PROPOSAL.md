# Symbolic Search RSI: Compact Executable Search-Program Policies

> **状态：研究提案，无实验结果。** 本文中的架构、训练流程、reward 权重与 rollout 数值均为待验证设计；“RSI”在这里是分阶段的 verifiable trajectory flywheel 工作名，不表示已经实现通用 recursive self-improvement。

## 一句话摘要与核心假设

**一句话摘要：** 用一个 question/state-conditioned tiny Transformer 生成 typed、pointer-grounded、可执行的 search program，在固定 retriever、reader 与工具预算下，争取获得比自由文本 LLM Agent 更好的 accuracy–cost–latency Pareto，并把验证成功的短程序沉淀为后续训练数据。

**核心假设：** 搜索规划需要的动作熵远低于通用语言生成；若把动作空间收缩为有类型的操作符、当前状态中的实体指针和观察文本 span 指针，小模型可以把容量集中到“下一步查什么、打开什么、绑定什么、何时停止”，而不是重复承担自然语言格式、工具协议和参数拼写。

该假设不等于“tiny 必然优于 3B”。成立条件是：在相同后端和预算下，tiny typed planner 至少在部分 operating points 上以显著更低的 planning latency/cost 保持有竞争力的答案与证据质量。

## 1. 背景与问题定义

普通 LLM Agent 往往在自由文本轨迹中交替输出 `Thought / Action / Observation`。这种接口灵活，但把多个问题混在同一生成空间：

1. 推理内容、工具选择、参数构造和协议格式都由语言模型生成；
2. query、文档 rank、实体与证据可能被自由生成，而不是绑定到可见状态；
3. 非法动作、参数拼写漂移和不可重放的隐式决策增加训练噪声；
4. 长轨迹的 token、latency 与推理成本随自然语言解释增长。

本提案把搜索策略写成 **typed symbolic program**。策略只预测下一条合法指令；executor 负责调用工具、维护状态、比较结构化值并记录证据。自然语言问题和检索观察仍然存在，但动作本身不再是任意文本。

形式化地，给定问题 \(q\)、状态 \(s_t\)、剩余预算 \(b_t\)，planner 学习：

\[
\pi_\theta(a_t \mid q, s_t, b_t), \quad a_t \in \mathcal{A}_{\mathrm{legal}}(s_t)
\]

其中 \(\mathcal{A}_{\mathrm{legal}}(s_t)\) 由 grammar、类型与当前可见指针共同确定。

## 2. 第一阶段的精确目标

### 2.1 要做

- 学习一个 compact search-program policy；
- 让程序在固定 search environment 中执行、重放和验证；
- 测量答案质量、证据质量、工具成本、planner latency 与端到端 latency；
- 比较 SFT、best-of-N、fresh-rollout RL 和不同 planner 规模；
- 保存成功且成本较低的程序，作为明确标注来源的 replay/SFT 数据。

### 2.2 暂时不做

- 不训练 question proposer 或自动出题器；
- 不让 planner 承担自由文本 answer generation；
- 不实现通用 browser agent；
- 不训练 reward model；
- 不加入 MoE、复杂 attention 或额外 critic；
- 不声称仅凭成功轨迹 archive 已经形成 RSI。

answer 由固定 reader 从已打开文档与 evidence slots 中抽取或选择。这样第一阶段只检验 search planning，而不是把答案生成能力混入结果。

## 3. 最小系统设计

### 3.1 组件

| 组件 | 最小职责 | 第一阶段是否训练 |
|---|---|---|
| Tiny planner | 根据 question、typed state、预算预测下一 opcode 与参数指针 | 是 |
| Action mask | 屏蔽语法、类型或当前状态下不合法的动作 | 否，确定性 |
| Executor | 解析程序、调用 search/open、更新 slots、执行比较、记录 evidence | 否 |
| Retriever | 给 query 返回固定 top-k 结果 | 否，baseline 间冻结 |
| Reader | 从已打开文档抽取 answer/evidence/结构化属性候选 | 否，baseline 间冻结 |
| Archive | 保存每题 top-k 成功且最短/最低成本程序供监督 replay | 否 |

### 3.2 Planner 输入与输出

输入采用紧凑序列化：

```text
[QUESTION] question tokens + addressable spans
[ENTITIES] typed entity table
[RESULTS] current result ranks and snippets
[FACTS] reader-extracted typed facts with source pointers
[EVIDENCE] accepted evidence pointers
[BUDGET] remaining search/open/step counts
```

最小模型可用 decoder-only tiny Transformer。输出拆为：

1. opcode head：预测 `SEARCH`、`OPEN`、`BIND_ENTITY` 等有限操作；
2. pointer heads：指向 question span、当前 entity、当前 result rank、已打开文档 span 或 answer candidate；
3. 少量 finite enum：例如 `birth_date`，不允许任意生成字段名。

### 3.3 Typed pointer-grounded DSL

核心指令：

| 指令 | 类型约束 | 作用 |
|---|---|---|
| `SEARCH(QSPAN)` | `QSPAN` 必须指向问题中的连续 span | 用问题原文片段发起检索 |
| `SEARCH_JOIN(QSPAN, ENTITY)` | 问题 span + 已绑定实体 | 组合关系意图与实体表面形式 |
| `OPEN(rank)` | `rank` 必须存在于当前结果列表 | 打开文档并运行固定 reader |
| `BIND_ENTITY(doc, span)` | `span` 必须来自已打开 `doc` | 把观察中的实体写入 symbol table |
| `SELECT_MIN(entity_a, entity_b, field)` | 两实体都必须有同类型可比较字段 | 由 executor 选择字段值较小者 |
| `STOP(answer, evidence)` | answer/evidence 都必须指向当前候选或已见 span | 返回答案与证据集合 |

`SELECT_MIN` 是为比较型多跳问题加入的最小确定性原语；它不生成新知识，只比较 reader 已提取、带来源指针的同类型值。

完整最小语法见 [`dsl/search_program.ebnf`](dsl/search_program.ebnf)。

### 3.4 Action mask

action mask 在每一步由状态生成：

- 没有当前 result list 时，所有 `OPEN(rank)` 被屏蔽；
- 未绑定的 entity 不能作为 `SEARCH_JOIN` 参数；
- `BIND_ENTITY` 只能指向已打开文档中真实存在的 span；
- 缺少两个同类型字段时，`SELECT_MIN` 被屏蔽；
- 没有 answer candidate 或 evidence pointer 时，`STOP` 被屏蔽；
- 达到工具预算后，继续 `SEARCH/OPEN` 被屏蔽，只保留当时可合法执行的收束动作。

因此语法合法性由 mask 保证，不需要再为“格式正确”设置 reward。executor 仍记录检索无结果、reader 未抽取到目标字段等任务失败，但这些不是语法错误。

### 3.5 Pointer grounding

参数必须来自可见来源：

- `QSPAN`：问题 token 区间；
- `ENTITY`：此前由 `BIND_ENTITY` 建立的 symbol；
- `doc/span`：已打开观察中的 token 区间；
- `rank`：当前结果列表中的位置；
- `answer/evidence`：reader candidate 或观察 span 的指针。

例如观察中出现 `Pomona College` 时，模型预测的是对应 span 的起止位置，executor 才建立 `$E3 = "Pomona College"`；模型不能直接自由生成这个学校名。

## 4. 数据与 pseudo-program compilation

### 4.1 数据角色

| 数据 | 计划用途 | 边界 |
|---|---|---|
| [SearchGym](https://arxiv.org/abs/2601.14615) / [repo](https://github.com/JIA-Lab-research/SearchGym) | 对齐语料与可验证知识图上的低成本训练/调试环境 | 不把 simulator 结果自动等同真实 Web 表现 |
| HotpotQA | 两跳、多文档答案与 supporting facts | supporting facts 不是 search trajectory |
| [2WikiMultiHopQA](https://github.com/Alab-NII/2wikimultihop) | 跨文档关系与比较型问题 | 数据注释只用于编译/评估，不进入 rollout 隐状态 |
| [MuSiQue](https://github.com/StonyBrookNLP/musique) | 组合性更强的多跳 QA 与 OOD 评估 | 需要单独报告不可检索与 reader failure |

HotpotQA、2WikiMultiHopQA 和 MuSiQue 的 supporting facts 表示答案证据，不是人类或 agent 实际执行过的 search trajectory。本文只把它们当作 **pseudo-trace compiler 的约束或目标**。

### 4.2 Oracle/pseudo-program compiler

compiler 尝试把问题与 supporting facts 转成可执行程序：

1. 从 question span 构造第一条合法 query；
2. 在固定 retriever 上执行，只有真实返回的 rank 才能 `OPEN`；
3. 只有打开后的 observation span 才能 `BIND_ENTITY`；
4. 用 supporting facts 判断是否到达目标 evidence，但不得把隐藏 gold title、doc id、answer string 直接写进 planner 可见状态；
5. 在同一 retriever/reader/executor 上完整 replay；不能仅靠可见指针重放成功的程序被丢弃，不进入 SFT。

这个 leakage gate 有明确消费者：**训练集构建器**。它改变的决策是“该 pseudo-program 能否进入 SFT/replay 数据”。最小机制就是受限编译加同环境 replay，不额外引入 provenance 系统。

### 4.3 防止 gold leakage 的具体规则

- 数据划分先于程序编译与 archive 构建；
- runtime state 不暴露 supporting-fact 标记、gold paragraph id 或 gold answer；
- query 字符串只能由 question span 与已观察 entity 拼接；
- answer/evidence 只能绑定到已打开文本或固定 reader 候选；
- 编译器若需要未观察的实体、标题或答案才能继续，该样本记为 compile failure，而不是补一个 oracle 动作；
- 报告 compiler coverage，使“只保留容易编译样本”不会被误写成全数据结果。

## 5. Reward

对一条完整 rollout \(\tau\)，候选 reward 为：

\[
R(\tau)=
w_a R_{\mathrm{answer}}
+w_e R_{\mathrm{evidence}}
+w_\Delta \sum_t \Delta \mathrm{new\_evidence}_t
-\lambda_s N_{\mathrm{search}}
-\lambda_o N_{\mathrm{open}}
\]

其中：

- `R_answer`：answer EM 与 token-level F1 的预先注册组合；
- `R_evidence`：最终 supporting-fact/evidence recall（可同时报告 F1）；
- `Δnew_evidence`：某条目标 evidence 首次被可验证地发现时的增量，重复打开不重复计分；
- `N_search`、`N_open`：轻微成本，防止无意义工具循环；
- grammar 与类型合法性由 action mask 保证，不另发 syntax reward。

**不把 novelty/diversity 直接塞进 reward。** 第一阶段的目标是正确、可验证且低成本的搜索。若需要多样程序，可在分析中报告不同成功序列，但不让不可验证的“新颖性”覆盖答案与证据目标。

## 6. 训练流程

### Stage A — Oracle/pseudo-program compilation

- 在训练 split 上编译并 replay 可执行程序；
- 记录 compile success、失败原因、长度、search/open 次数；
- 只让可重放程序进入下一阶段。

### Stage B — SFT

- 对下一 opcode 与各 pointer head 做 teacher forcing；
- 验证集同时看 action accuracy、完整 program success、answer/evidence 与工具成本；
- tiny SFT 是必须独立报告的 baseline，而不是只作为 RL 初始化细节。

### Stage C — Fresh-rollout RL

- 用**当前 policy**在固定 executor 中生成 fresh rollouts；
- 优先采用 RLOO 或直接 REINFORCE，避免在 MVP 中引入额外 critic/reward model；
- 同题多样本用于 RLOO baseline 时，所有样本都来自当前 policy；
- 训练与评估严格按工具预算截断。

### Stage D — Successful program archive/replay

- 每题保存 top-k 成功程序，先按任务 reward，再按程序长度与 search/open 成本选择最短者；
- archive 作为后续 SFT/replay 数据来源，明确标注生成 policy 版本和可重放环境配置；
- **不能把 archive 中旧 rollout 的 logprob 当成当前 policy 的 on-policy 样本。**
- MVP 不对 archive 做伪装成 on-policy 的 policy-gradient 更新；若未来采用 off-policy 算法，必须另立方法与实验。

## 7. 公平 baseline

| Baseline | Planner 表示 | 训练 |
|---|---|---|
| Rule/heuristic | 手写 query 与 open 策略 | 无 |
| Tiny SFT | 同一 typed DSL | SFT |
| Tiny SFT + best-of-N | 同一 typed DSL，多次采样后用可验证 reward 选优 | SFT |
| Tiny SFT + RL | 同一 typed DSL | SFT + fresh-rollout RLOO/REINFORCE |
| 3B same-DSL planner | 同一 typed DSL 与 mask | 与 tiny 对齐的 SFT/RL 设置 |
| 3B free-text planner | `Thought/Action` 文本工具协议 | 同数据与工具接口 |

公平性约束：

- 所有模型使用同一 retriever、corpus、reader、top-k 与文档截断；
- 每题总 `SEARCH`、`OPEN` 和 step budget 相同；
- best-of-N 的所有候选工具调用计入总预算与成本；
- 同时报 planner-only 与 end-to-end latency，避免后端延迟掩盖 planner 差异；
- answer/evidence evaluator 完全相同；
- 3B free-text 的协议解析失败单独报告，不通过重试获得额外预算。

## 8. 三个决定生死的实验

### 实验 1：表示与规模的 accuracy–cost–latency Pareto

**问题：** tiny typed planner 是否提供真实 Pareto 点，而不仅是降低了参数量？

**比较：** rule、tiny SFT、tiny SFT+best-of-N、tiny SFT+RL、3B same-DSL、3B free-text。

**Metrics：**

- answer EM/F1；
- evidence recall/F1；
- complete program success；
- syntactically invalid action rate 与 executor failure rate；
- 每题 search/open/step 数、planner tokens；
- planner latency p50/p95、端到端 latency p50/p95、可归因的推理成本。

**Kill decision：** 若 tiny typed 在所有预算点都同时被 3B 或 tiny free-text 的质量、成本和延迟严格支配，则核心 compact-policy 主张不成立。

### 实验 2：RL 与 archive 是否产生可归因增益

**问题：** fresh-rollout RL 是否优于 SFT 和等预算 best-of-N？successful archive 是否只提供监督 replay 价值，而没有制造伪 on-policy 增益？

**最小消融：**

1. tiny SFT；
2. tiny SFT + best-of-N；
3. tiny SFT + fresh-rollout RL；
4. tiny SFT + fresh-rollout RL + shortest-success replay。

**Metrics：** 与实验 1 相同，另加 fresh-rollout success、archive replay success、每个成功程序成本和训练期间 reward/长度曲线。

**Kill decision：** 若 RL 在相同 rollout/tool compute 下不能稳定优于 SFT 与 best-of-N，则去掉“RL 必要”主张；若 replay 只复制训练题而损害 held-out，则停止 archive flywheel 扩展。

### 实验 3：OOD 泛化与反事实 leakage tests

**OOD：**

- 在 SearchGym/HotpotQA 训练，转到 2WikiMultiHopQA 与 MuSiQue；
- 按 hop 数、比较型/桥接型关系、未见实体与更紧工具预算分桶；
- 交换固定 retriever 的索引快照或 top-k 设置时单独报告，不与 in-domain 混合。

**反事实 leakage tests：**

1. rollout 输入中移除全部 supporting-fact annotation，行为与正常 runtime 应完全一致；
2. 对从未在 observation 中出现的隐藏 gold doc id/title 做置换，编译后程序与执行结果不应因此改变；
3. 在 compiler 输入中交换错误 supporting facts，程序应在受限 replay 中失败或被拒绝，而不能自动注入被交换的 gold title；
4. 对 answer string 做仅元数据层的替换，若答案从未作为观察 span 出现，`STOP` 必须不可绑定。

**Metrics：** OOD answer/evidence、compiler coverage、counterfactual invariance、被 leakage gate 拒绝的比例与原因。

**Kill decision：** 任一结果依赖 runtime 不可见的 gold metadata，则该数据管线结果无效；先修 compiler 并重跑同一协议，不把它包装成模型能力。

## 9. 7–14 天 MVP

| 时间 | 交付 |
|---|---|
| Day 1–2 | 冻结 DSL、typed state、action mask、executor 接口；接一个固定本地 retriever/reader |
| Day 3–5 | 为一个数据集实现受限 pseudo-program compiler；统计 coverage 并人工核查少量 replay |
| Day 5–7 | 训练 tiny SFT；跑 rule、tiny SFT、3B same-DSL 的最小对照 |
| Day 7–10 | 加入 current-policy fresh rollouts 与 RLOO/REINFORCE；记录 answer/evidence/cost |
| Day 10–12 | 加入 shortest-success archive 的监督 replay；完成 best-of-N 对照 |
| Day 12–14 | 跑三项生死实验的最小版本、OOD/leakage tests，并写诚实结论 |

MVP 的成功条件不是“完成所有自改进设想”，而是回答三个问题：typed tiny 是否有 Pareto 价值、RL 是否有可归因增益、pseudo-program 是否无 gold leakage 且可重放。

## 10. 后续 proposer–solver self-improvement 路线

只有 MVP 通过后才进入后续阶段：

1. **Phase 1 — Solver flywheel：** 用真实任务的 fresh successful programs 扩充监督集，继续保持固定问题分布；
2. **Phase 2 — Constrained proposer：** proposer 从封闭 corpus 与可验证 evidence chain 生成候选问题，executor 验证问题可解、答案与证据一致；
3. **Phase 3 — Curriculum：** 按 solver 的失败类型选择可验证候选，而不是直接奖励“新颖”；
4. **Phase 4 — Cross-model distillation：** 用 3B/7B solver 产生或复核成功 typed programs，再蒸馏给 tiny planner。

proposer 与 solver 必须分阶段评估。第一阶段不把自出题、answer generation 或 reward model 混入 search-policy 结果。

## 11. Novelty boundary

不能声称本文发明了：

- search RL；
- symbolic DSL 或 program synthesis；
- query reformulation；
- 自动出题、自博弈或 curriculum；
- trajectory replay。

可检验的候选贡献只有：

1. **小型条件式 typed planner 的 accuracy–cost–latency Pareto**：同后端、同 reader、同工具预算下，系统地比较 tiny 与 3B、typed 与 free-text；
2. **verifiable trajectory flywheel**：用 pointer-grounded、可重放、成功且较短的 search programs 形成透明的数据循环，并明确区分 on-policy RL 与 archive replay。

是否构成论文贡献取决于实验，不在提案阶段预先宣称。

## 12. 主要近邻

| 近邻 | 与本提案的关系 | 本提案边界 |
|---|---|---|
| [AlphaGPT 原仓库](https://github.com/imbue-bit/AlphaGPT) | 用户指定的项目语境近邻 | 仅作为原项目入口；不从仓库页面推断未经验证的学术结论 |
| [Task-Oriented Query Reformulation with Reinforcement Learning (2017)](https://arxiv.org/abs/1704.04572) | 用 RL 选择词项重写 query，以 document recall 为 reward | 本提案学习多步 typed search program，并显式追踪 evidence/cost |
| [DreamCoder](https://arxiv.org/abs/2006.08381) | 神经引导的程序搜索、可解释程序与 wake-sleep 学习 | 本提案不学习新 DSL 抽象，先固定极小 search DSL |
| [PAQ](https://arxiv.org/abs/2102.07033) | 大规模自动生成 QA pairs 与快速显式检索 | 可启发后续数据规模化，但 MVP 不自动出题 |
| [WebRL](https://arxiv.org/abs/2411.02337) | self-evolving online curriculum 与 Web-agent RL | 本提案第一阶段没有 Web UI、ORM 或自进化 curriculum |
| [Search-R1](https://arxiv.org/abs/2503.09516) | 多轮搜索交互中的 RL 与 outcome reward | 本提案关注更小模型、typed/pointer-grounded 动作和显式成本 Pareto |
| [ASearcher / Beyond Ten Turns](https://arxiv.org/abs/2508.07976) | 大规模异步 RL、长时程 agentic search 与自动 QA 合成 | 本提案刻意缩小到短、可执行程序和 7–14 天 MVP |
| [SearchGym](https://arxiv.org/abs/2601.14615) / [repo](https://github.com/JIA-Lab-research/SearchGym) | 对齐知识图与文档语料的高保真搜索模拟、SearchGym-RL | 可作为训练环境；仍需真实数据/OOD 评估，不能把 sim-to-real 当作既成结果 |
| [Absolute Zero](https://arxiv.org/abs/2505.03335) | proposer–solver 式、可验证执行器支持的自生成任务与推理改进 | 只启发后续阶段；MVP 不声称 zero-data 或 self-play |

## 13. 风险与 kill criteria

| 风险 | 可观察信号 | 决策 |
|---|---|---|
| DSL 过窄，表达不了真实搜索策略 | 大量问题因缺少必要操作而 compile/rollout 失败 | 先按失败簇增加一个有消费者的最小 typed primitive；若仍无覆盖则停止该 DSL |
| Tiny 容量不足 | 在同 DSL 下被 3B 全面支配 | 放弃“tiny Pareto”主张，保留 typed interface 作为独立工程结论 |
| Pointer grounding 限制召回 | gold entity 经常不在可见 question/observation span | 归因到 retriever/reader 或 DSL 可达性，不用自由生成 fallback 掩盖 |
| Pseudo-trace gold leakage | 反事实 metadata 改动影响 rollout | 结果作废，修 compiler 后重跑 |
| RL 无增益或不稳定 | 等预算下不优于 SFT/best-of-N | 删除 RL 必要性主张，停止扩大训练 |
| Archive 过拟合 | replay 提高训练题但降低 held-out/OOD | 停止 archive 扩展，仅保留 fresh-rollout 结果 |
| Reader 成为瓶颈 | evidence 已检索到但固定 reader 经常抽取失败 | 单独报告 reader ceiling；不把它算作 planner 搜索失败 |

## 14. 预期价值

若核心假设成立，预期价值是：

- 更低的 planning latency 与推理成本；
- 在完整 action mask 下，语法非法动作接近零；
- 每个参数都能追溯到 question、rank、entity 或 observation span；
- 程序可重放、可比较工具成本、可验证 evidence；
- 成功程序可为 3B/7B 或 tiny 模型生成明确来源的 SFT 数据；
- 为后续 proposer–solver self-improvement 提供窄而可测的基础。

这些是待验证价值，不是结果承诺。尤其不承诺 tiny planner 必然击败 3B；合理成功也可能是 tiny 在低延迟/低成本区域提供一个有用 Pareto 点。

