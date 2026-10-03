# RQ4：Qwen3-VL-8B 完整动作生成时延实验

日期：2026-10-03  
最终有效结果：`final_5reps_v3.json`。早期计时边界错误的版本不纳入本报告。

## 1. 研究问题与结论

本实验测量在一次完整 structured-action generation 中加入 Policy2Sample 的系统级时延影响，比较：

1. **Schema-only baseline：**XGrammar structural mask 与预先编译的完整有限动作格点 canonical trie 相交；不按安全策略过滤候选。
2. **Policy2Sample：**使用相同 XGrammar grammar，并在每个请求中依据当前 Context 重新计算安全动作域，再加入 per-request safe-domain prefix mask。

结果显示，Policy2Sample 相对 baseline 的 mean latency 在 Robot 增加 **33.44 ms（4.72%）**，在 IoT 增加 **83.99 ms（19.86%）**。该结果是本次 8B 配置与有限离散动作域上的测量；不是连续数值 token enforcement、任务成功率或真实物理执行结论。

## 2. Workload 与公平比较设置

- Robot：冻结的 Robot Core-30 latency workload，30 cases；其中含 3 个 STOP 控制 case，其余为 MOVE。
- IoT：冻结的 Hard-IoT Core-30 latency workload，30 cases。
- 每个 case、每种方法重复 5 次，共 `60 × 2 × 5 = 600` 次计时生成；另有 4 次 warm-up（每个 domain/method 一次）。
- 方法采用按 case 与 repetition 交替的配对顺序，减少运行顺序造成的系统漂移。
- 每次 request 使用相同 case prompt、动作 schema 与 context；Policy2Sample 的安全域不暴露给 baseline。
- greedy argmax（temperature 0），减少随机解码对运行时间的扰动。
- Robot MOVE 的完整有限动作格点为 6,000 个候选；IoT 为 3,888 个候选；STOP 控制域只有 STOP。

需要准确描述 baseline：为避免 XGrammar 对任意空白/格式变体进行无界继续，并让它和 P2S 在同一有限规范序列集合上比较，baseline 在 XGrammar 结构约束之外还使用一次编译的**完整动作格点 canonical trie**。该 trie 不筛除不安全候选；P2S 在相同结构约束之上使用当前 request 的安全候选 trie。因此论文不应把该 baseline 不加限定地称为“raw XGrammar-only”。

## 3. 计时边界

计时从当前请求的 setup 开始，终止于完整动作完成 JSON parse。对于 Policy2Sample，计时包含：

`Context/action domain projection → safe candidate serialization → per-request prefix trie specialization → prompt prefill → constrained decode → JSON parse`

对于 baseline，计时包含 request setup、prompt prefill、XGrammar/full-lattice trie constrained decode 与 JSON parse。两者都不包含模型加载、词表抽取/缓存、grammar compilation、完整动作格点的一次性 tokenization。这些一次性成本另行记录，以免与每请求成本混淆。

计时边界审计发现旧版曾把 P2S 的安全域构造放在计时器外，因此旧版无效。本报告只使用修正后、将 projection、候选序列化和每请求 prefix trie 构建纳入计时的 v3 结果。

## 4. 结果

### 4.1 Robot

| 方法 | n | Mean (ms) | P50 (ms) | P95 (ms) | Mean output tokens | P50 / P95 tokens |
|---|---:|---:|---:|---:|---:|---:|
| Schema-only baseline | 150 | 707.849 | 758.271 | 780.580 | 41.9 | 46 / 46 |
| Policy2Sample | 150 | 741.285 | 780.071 | 867.320 | 41.9 | 46 / 46 |
| Paired delta (P2S − baseline) | 150 pairs | +33.436 mean | +13.854 P50 | +110.170 P95 | — | — |

相对 baseline mean 的 overhead 为 **4.724%**。两方法平均输出 token 数相同；本 workload 下测得的差异包含 P2S 的每请求 specialization 以及解码/计时噪声。

### 4.2 IoT

| 方法 | n | Mean (ms) | P50 (ms) | P95 (ms) | Mean output tokens | P50 / P95 tokens |
|---|---:|---:|---:|---:|---:|---:|
| Schema-only baseline | 150 | 422.888 | 422.776 | 440.956 | 28.0 | 28 / 28 |
| Policy2Sample | 150 | 506.881 | 508.575 | 533.921 | 33.5 | 33.5 / 34 |
| Paired delta (P2S − baseline) | 150 pairs | +83.994 mean | +84.034 P50 | +107.569 P95 | — | — |

相对 baseline mean 的 overhead 为 **19.862%**。P2S 平均比 baseline 多生成 **5.5 tokens**；逐配对审计表明差异来自选择的动作值不同，而非序列化格式不同（见 4.3）。因此端到端时延增加不能全部归因于 semantic mask 或 solver；报告该差异时必须同时给出 token 数。

百分位数由 NumPy 默认 linear percentile 算法计算。相对 overhead 定义为：

`(P2S mean latency − baseline mean latency) / baseline mean latency × 100%`。

paired delta 的 P50/P95 是逐 case/repetition 配对时延差的分位数，不是两组独立分位数相减。

### 4.3 IoT 输出格式与动作选择审计

对 150 个正式配对逐条检查：

- 两种方法输出均与 `json.dumps(..., sort_keys=True, separators=(",", ":"))` 的 canonical JSON 完全相同：baseline **150/150**，P2S **150/150**。
- 两者均无 whitespace；字段集合与顺序相同：`action, d, e, h, mode, w`。IoT 数值字段均为整数，无一方输出小数；`mode` 均为 JSON string。两边使用相同 XGrammar grammar 与相同终止检查。
- 序列化/token path 相同：完整动作格点与安全子集均由同一 `compact()`、`iot_action_obj()` 及预计算的 `ids_by_candidate` 构造；两类 trie 共用同一个 candidate-string → token-ID 映射。
- 150 个配对中没有一对最终 action 相同。Schema-only **150/150** 次输出同一全零配置（28 tokens）；P2S 输出 12 种不同配置，token 数为 32–35（均值 33.5）。

因此观察到的 **+5.5 tokens** 来自动作选择导致的数值内容长度差异，不是空格、key 顺序、整数/小数格式或 serializer 分叉。IoT end-to-end latency 比较包含输出内容长度变化的影响，不能解释为“固定相同 action 时 semantic mask 单独增加多少时延”。此 audit 没有重跑推理或计时。

## 5. Policy2Sample 每请求 specialization 成本

下表为 P2S 逐请求记录的三个阶段的均值；三者都已包含在上述总 latency 中。

| Domain | Projection (ms) | Safe candidate serialization (ms) | Per-request prefix trie (ms) | 合计 (ms/action) |
|---|---:|---:|---:|---:|
| Robot | 2.139 | 4.904 | 24.583 | 31.626 |
| IoT | 1.991 | 4.354 | 17.632 | 23.976 |

因此当前开销主要落在安全候选表示与 per-request prefix trie 构建，而不是单独的 constraint projection。该结论只针对本次有限格点实现和 workload。

## 6. 一次性成本（不计入每请求 latency）

- Vocabulary preparation：9.771 ms，缓存命中。
- XGrammar grammar compilation：Robot MOVE 2.252 ms；Robot STOP 0.672 ms；IoT 1.029 ms。
- Global action candidate tokenization：841.426 ms。

这些成本在本报告的每请求时延中排除。若论文要报告冷启动/摊销总成本，应另外说明调用次数和 amortization 方法，不能直接把上表当作 cold-start latency。

## 7. 实验环境与可复现信息

- 模型：Qwen3-VL-8B-Instruct-Q8_0 GGUF。
- 模型 SHA-256：`0d264b3941185d00a74f75c4245521dae088ff1efc90ab8d1754e83f5844adb0`。
- 远程实验环境：Linux 6.8.0-136-generic x86_64，Python 3.11.15；主机标识已从匿名 artifact 中省略。
- GPU：2 × NVIDIA GeForce RTX 3090，24,576 MiB each。
- llama-cpp-python：0.3.33。
- XGrammar：结果 JSON 的运行时版本字段记录为 `unknown`；投稿稿件不要猜填版本号，若需要精确版本应从远程环境的包元数据补查。
- 运行脚本：[remote_rq4_e2e_latency_8b.py](../code/remote_rq4_e2e_latency_8b.py)。
- 原始逐次结果：[final_5reps_v3.json](../data/raw/final_5reps_v3.json)，包含每条 run 的方法、case、解析动作、候选匹配、token 数、各阶段时延和 warmup 标记。
- 固定 benchmark 输入快照：[Robot Core-30 v2](../data/raw/robot_core30_v2_frozen_input.json)、[Hard-IoT Core-30](../data/raw/hard_iot_core30_selection_input.json)；来源、SHA-256、运行配置和版本有效性说明见[RQ4 README](../README.md)。

## 8. 完整性与限制

- 最终文件共 604 rows：600 条正式测量与 4 条 warm-up；正式测量状态均为 COMPLETE。
- P2S 正式测量没有候选匹配失败（`policy_candidate_match=false` 为 0）。这只说明生成结果属于本次计算得到的 safe candidate set，不替代对源自然语言 policy 正确性的验证。
- 这里测的是有限离散动作候选与 tokenizer prefix mask，不是连续数值区间上的数字 token 级精确约束。
- 没有进行 AI2-THOR、SimuHome 或真实硬件 replay；也不应由此推断任务成功率或物理安全率。
- IoT 输出 token 数分布不同，case-level audit 已确认源于 action choice，而非格式差异；Robot token 数分布相同。IoT 的 E2E latency 差异应作为系统总开销报告，不应被描述成纯 mask 计算 overhead。
- Baseline 实际采用 XGrammar 加 full-lattice canonical trie；此实现细节需在论文中明确，避免给读者造成与无约束 JSON 结构解码比较的误解。
