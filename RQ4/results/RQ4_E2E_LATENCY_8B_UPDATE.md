# RQ4：8B 完整动作生成时延——更新摘要

**结论：**在冻结的 Robot 30 与 IoT 30 latency workload 上，Qwen3-VL-8B、greedy decoding、每个 case 每种方法 5 次配对测量。Policy2Sample 的端到端动作生成平均时延在 Robot 增加 **33.44 ms（4.72%）**，在 IoT 增加 **83.99 ms（19.86%）**。这是有限离散动作域下的系统级测量，不代表连续数值 tokenization 或真实硬件 replay。

| Workload | 方法 | n | Mean (ms) | P50 (ms) | P95 (ms) | 平均输出 tokens |
|---|---|---:|---:|---:|---:|---:|
| Robot | XGrammar + full action-lattice trie（无安全过滤） | 150 | 707.85 | 758.27 | 780.58 | 41.9 |
| Robot | XGrammar + Policy2Sample | 150 | 741.28 | 780.07 | 867.32 | 41.9 |
| IoT | XGrammar + full action-lattice trie（无安全过滤） | 150 | 422.89 | 422.78 | 440.96 | 28.0 |
| IoT | XGrammar + Policy2Sample | 150 | 506.88 | 508.57 | 533.92 | 33.5 |

Robot 的平均输出 token 数一致。IoT 的逐配对格式审计显示：两种方法均输出无空白、同 key 顺序的 canonical JSON，序列化路径一致；150 个配对没有一对生成相同动作。Baseline 150 次都选择全零配置，P2S 则选择了 12 种安全域内配置。因此多出的 5.5 tokens 来自动作值/token 长度不同，不是 serializer 不一致；IoT 的端到端时延差仍不能全归因于约束开销。P2S 每次请求均重新计算安全域，并将投影、候选序列化和安全前缀索引构建计入计时；这些步骤平均合计 Robot **31.63 ms/action**、IoT **23.98 ms/action**。

**计时口径：**从 request setup / per-request specialization 开始，包含 prompt prefill、受约束解码及 JSON parse；不含模型加载、词表抽取、grammar 编译和全动作格点的一次性 tokenization。Baseline 使用一次编译的完整动作格点 canonical trie 与 XGrammar 结构约束相交，以限制有限候选和规范序列化；它不应用安全过滤。故应称为“XGrammar + full action-lattice trie baseline”，不要简写成未经限定的 raw XGrammar。

**实验边界：**60 cases × 2 methods × 5 repetitions = 600 measured generations，另有 4 个 warmups；全部 600 条测量完成，Policy2Sample 生成值与当次安全域候选相符。动作参数来自有限离散格点，不是连续域测试；未在 AI2-THOR/SimuHome 中 replay，因此这里仅回答 runtime latency。

完整设置、计时组成、硬件和限制见[完整实验报告](RQ4_E2E_LATENCY_8B_FULL_REPORT.md)。归档与复现说明见[RQ4 README](../README.md)。原始逐次数据：[JSON](../data/raw/final_5reps_v3.json)；运行脚本：[runner](../code/remote_rq4_e2e_latency_8b.py)。
