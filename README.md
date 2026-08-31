# Symbolic Search RSI

Symbolic Search RSI 是一个研究提案：让小型、question/state-conditioned Transformer 不再生成自由文本式搜索动作，而是生成可执行、可类型检查、参数由当前问题或观察结果指针绑定的 search program。

> **当前状态：research proposal。** 本仓库没有实现、训练产物或实验结果，也不声称已经实现 recursive self-improvement（RSI）。

第一阶段只研究 **search-program policy**：固定 retriever、reader 与 tool budget，比较 typed symbolic planner 和自由文本 planner 的 accuracy–cost–latency Pareto。后续才考虑用可验证的成功轨迹形成 proposer–solver 数据飞轮。

## 文档

- [主提案：`PROPOSAL.md`](PROPOSAL.md)
- [完整具体例子：2020 年诺贝尔化学奖](examples/nobel_chemistry_2020.md)
- [最小 DSL 语法：`dsl/search_program.ebnf`](dsl/search_program.ebnf)
- [当前进度：`progress.md`](progress.md)

## 范围边界

- 不同时实现出题、answer generation、browser agent 或 reward model。
- 不把 supporting facts 当作原生 search trajectory。
- 不把 novelty/diversity 直接加入 reward。
- 不承诺 tiny planner 必然击败 3B planner；核心问题是它能否提供更好的可验证 Pareto 点。

