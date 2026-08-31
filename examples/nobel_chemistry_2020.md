# Concrete Example: Nobel Chemistry 2020

> **用途边界：** 这是说明程序表示、状态变化、mask 和 reward 的手工示例，不是 benchmark result，也不声称以下轨迹真实联网执行。

## 问题与事实终点

**问题：**“2020 年诺贝尔化学奖两位得主中，出生更早的那位，本科就读的大学位于哪个国家？”

用于核对示例终点的一手页面：

- Nobel Prize 的 [Jennifer A. Doudna facts](https://www.nobelprize.org/prizes/chemistry/2020/doudna/facts/)：出生于 1964-02-19；
- Nobel Prize 的 [Emmanuelle Charpentier facts](https://www.nobelprize.org/prizes/chemistry/2020/charpentier/facts/)：出生于 1968-12-11；
- Doudna 的 Nobel Prize [biographical text](https://www.nobelprize.org/prizes/chemistry/2020/doudna/biographical/)：1981 年进入 Pomona College，1985 年毕业；
- Pomona College [About](https://www.pomona.edu/about)：学校位于 Southern California。

因此较早出生者是 Jennifer A. Doudna，本科院校是 Pomona College，所在国家为 **United States**。

## 1. 自由文本 LLM Agent 轨迹

以下是简化的 `Thought / Action / Observation` 表示：

| Step | Thought | Action | Observation |
|---|---|---|---|
| 1 | 先找 2020 年化学奖的两位得主。 | `Search["2020 Nobel Prize in Chemistry laureates"]` | 结果摘要提到 Jennifer A. Doudna 与 Emmanuelle Charpentier。 |
| 2 | 比较两人的出生日期。 | `Search["Jennifer Doudna date of birth"]` | 摘要给出 19 February 1964。 |
| 3 | 还需要另一位的日期。 | `Search["Emmanuelle Charpentier date of birth"]` | 摘要给出 11 December 1968；Doudna 更早。 |
| 4 | 查 Doudna 的本科院校。 | `Search["Jennifer Doudna undergraduate college"]` | 观察文本提到她进入并毕业于 Pomona College。 |
| 5 | 查学校所在国家。 | `Search["Pomona College country"]` | 观察文本定位到 Claremont/Southern California, United States。 |
| 6 | 证据链完整。 | `Finish["United States"]` | 结束。 |

该形式可读，但 action 参数是自由文本；模型可能拼错学校名、引用未观察实体或输出不符合工具协议的字符串。

## 2. 对应的 symbolic DSL program

为便于阅读，下面显示指针及其解析后的文本；runtime 实际保存 token/span/rank 索引，而不是让 planner 重新生成引号内字符串。

```text
01 SEARCH(<q:0:5> = "2020 年诺贝尔化学奖")
02 OPEN(1)
03 BIND_ENTITY($D1, <s:18:21> = "Jennifer A. Doudna") -> $E1
04 BIND_ENTITY($D1, <s:27:28> = "Emmanuelle Charpentier") -> $E2

05 SEARCH_JOIN(<q:8:9> = "出生", $E1)
06 OPEN(1)  # reader: birth_date($E1) = 1964-02-19 @ $D2:<s:44:46>
07 SEARCH_JOIN(<q:8:9> = "出生", $E2)
08 OPEN(1)  # reader: birth_date($E2) = 1968-12-11 @ $D3:<s:39:41>
09 SELECT_MIN($E1, $E2, birth_date) -> $E_earlier = $E1

10 SEARCH_JOIN(<q:15:18> = "本科就读的大学", $E_earlier)
11 OPEN(1)
12 BIND_ENTITY($D4, <s:52:53> = "Pomona College") -> $E3

13 SEARCH_JOIN(<q:22:24> = "位于哪个国家", $E3)
14 OPEN(1)  # reader: country($E3) = "United States" @ $D5:<s:31:32>
15 STOP(
     answer=$D5:<s:31:32>,
     evidence={
       $D1:<s:18:28>,
       $D2:<s:44:46>,
       $D3:<s:39:41>,
       $D4:<s:45:55>,
       $D5:<s:25:32>
     }
   )
```

关键点：

- 两位得主必须先从 `$D1` 的 observation span 绑定为 `$E1/$E2`；
- 两个出生日期由固定 reader 从已打开文档抽取，且保留来源指针；
- `SELECT_MIN` 只比较两个已存在的 `birth_date`，不访问隐藏 gold；
- `Pomona College` 必须从 `$D4` 绑定，不能由 planner 自由生成；
- 最终答案也是 `$D5` 中 `United States` span 的指针。

## 3. 每一步 state slots 的变化

初始状态：

```text
entities={}
results=[]
opened_docs={}
facts={}
selected={}
evidence={}
answer_candidates={}
budget={search:5, open:5, steps:15}
```

| Step | Action | State delta |
|---:|---|---|
| 1 | `SEARCH(qspan)` | `results=[R1...]`; `search_budget=4` |
| 2 | `OPEN(1)` | `opened_docs += {$D1}`; `results` 保留来源；`open_budget=4` |
| 3 | `BIND_ENTITY($D1, span)` | `entities[$E1]=Jennifer A. Doudna` |
| 4 | `BIND_ENTITY($D1, span)` | `entities[$E2]=Emmanuelle Charpentier` |
| 5 | `SEARCH_JOIN(qspan,$E1)` | `results=[R_Doudna_birth...]`; `search_budget=3` |
| 6 | `OPEN(1)` | `opened_docs += {$D2}`; `facts.birth_date[$E1]=1964-02-19@$D2` |
| 7 | `SEARCH_JOIN(qspan,$E2)` | `results=[R_Charpentier_birth...]`; `search_budget=2` |
| 8 | `OPEN(1)` | `opened_docs += {$D3}`; `facts.birth_date[$E2]=1968-12-11@$D3` |
| 9 | `SELECT_MIN(...)` | `selected[$E_earlier]=$E1` |
| 10 | `SEARCH_JOIN(qspan,$E_earlier)` | `results=[R_Doudna_education...]`; `search_budget=1` |
| 11 | `OPEN(1)` | `opened_docs += {$D4}`; education evidence candidate 出现 |
| 12 | `BIND_ENTITY($D4, span)` | `entities[$E3]=Pomona College` |
| 13 | `SEARCH_JOIN(qspan,$E3)` | `results=[R_Pomona_location...]`; `search_budget=0` |
| 14 | `OPEN(1)` | `opened_docs += {$D5}`; `facts.country[$E3]=United States@$D5`; `answer_candidates += {A1}` |
| 15 | `STOP(A1,evidence)` | 固化 answer/evidence，episode 结束 |

state 只增加 executor 能从已执行动作和当前 observation 验证的内容。planner 看不到 supporting-fact 标签、gold document id 或未打开文档。

## 4. Grammar/action mask 示例

考虑 **Step 12 完成后** 的状态：`$E1/$E2/$E3` 已绑定，当前打开文档是 `$D4`，但国家答案尚未观察。

### Allowed

| Candidate | 原因 |
|---|---|
| `SEARCH_JOIN(<q:22:24>, $E3)` | qspan 来自问题，`$E3` 已绑定 |
| `SEARCH_JOIN(<q:22:24>, $E1)` | 类型合法，虽然策略上不如 `$E3` |
| `BIND_ENTITY($D4, <valid observed span>)` | 指针落在已打开文档的实体 span |

### Masked

| Candidate | 屏蔽原因 |
|---|---|
| `OPEN(7)` | 当前没有与 rank 7 对应的 active result |
| `SEARCH_JOIN(<q:22:24>, $E9)` | `$E9` 未绑定 |
| `BIND_ENTITY($D4, "Harvard University")` | 参数不是 `$D4` 中 observation span 的指针 |
| `SELECT_MIN($E1,$E3,birth_date)` | `$E3` 没有 `birth_date` |
| `STOP("United States", ...)` | 国家 answer span 尚未出现，且自由字符串不是合法 answer pointer |

因此 mask 不判断“哪个合法动作最聪明”，只把不可执行或无来源的动作概率设为零；planner 仍需在 allowed set 中学习正确决策。

## 5. Pointer-grounded 参数示例

假设 `$D4` 的 observation token 为：

```text
... Doudna enrolled at Pomona College in 1981 and graduated in 1985 ...
```

planner 的 pointer head 选择 token 区间 `<s:52:53>`；executor 验证该区间属于 `$D4` 且 reader 标为实体，然后建立：

```text
$E3 = entity(
  surface="Pomona College",
  source=$D4:<s:52:53>
)
```

不合法的替代方式是让模型直接生成 `Pomona College`。即使文本恰好正确，自由生成也失去来源约束，并可能在其他样本中生成未观察学校。

## 6. 失败与成功 rollout 的 illustrative reward

> 以下系数和分数只用于说明 reward 分解，**不是实验结果**。

示意系数：

```text
answer EM/F1 contribution: up to +1.00
final evidence recall:     up to +0.50
first new evidence:        +0.10 each, capped at 4 milestones
SEARCH cost:               -0.02 each
OPEN cost:                 -0.01 each
syntax reward:              0.00 (legality comes from mask)
```

### 失败 rollout

轨迹只找到两位得主和 Doudna 的出生日期，随后错误地停止；没有定位另一位出生日期、学校和国家。

| 项 | 值 | Reward |
|---|---:|---:|
| Answer EM/F1 | 0 | `+0.00` |
| Evidence recall | 0.25 | `+0.125` |
| First-new evidence | 2 milestones | `+0.20` |
| Tool cost | 2 `SEARCH` + 2 `OPEN` | `-0.06` |
| **Total** | | **`+0.265`** |

### 成功 rollout

上面的 15-step program 找到两位得主、比较日期、绑定 Pomona College、定位国家，并返回完整证据。

| 项 | 值 | Reward |
|---|---:|---:|
| Answer EM/F1 | 1 | `+1.00` |
| Evidence recall | 1 | `+0.50` |
| First-new evidence | 4 capped milestones | `+0.40` |
| Tool cost | 5 `SEARCH` + 5 `OPEN` | `-0.15` |
| **Total** | | **`+1.75`** |

若另一个成功程序使用更少 search/open，在答案与证据相同的情况下会获得略高 reward，并更可能进入 shortest-success archive。archive 仍只用于监督 replay，旧 logprob 不会被当作当前 policy 的 on-policy 数据。

## 最终答案

**United States**

本例只展示如何把多跳搜索表示成 typed、pointer-grounded、可重放程序；它不构成该方法在任何 benchmark 上的结果。

