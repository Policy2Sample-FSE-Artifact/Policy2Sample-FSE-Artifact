# RQ2 Report: Context-Moving Narrow Safety Domain

## Summary

We evaluated a fixed, coupled `MOVE(distance_cm, speed_cm_s)` safety policy over 33 changing Context snapshots and a finite catalog of 40 parameter pairs. At every snapshot, the oracle safe-speed projection contains only one or two values; the intersection of admissible speeds across the full sweep is empty. The study therefore removes a single fixed safe-speed fallback as an explanation for sweep-wide success.

| Model | Method | Safe parameter pair | Unsafe final output | No-action | Mean calls | Mean latency |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-VL-2B | Prompt-only | 2/33 (6.1%) | 31/33 | 0/33 | 1.00 | 327.6 ms |
| Qwen3-VL-2B | AgentSpec | 14/33 (42.4%) | 0/33 | 19/33 | 2.67 | 865.8 ms |
| Qwen3-VL-2B | Policy2Sample | 33/33 (100%) | 0/33 | 0/33 | 1.00 | 318.1 ms |
| Qwen3-VL-4B | Prompt-only | 2/33 (6.1%) | 31/33 | 0/33 | 1.00 | 452.5 ms |
| Qwen3-VL-4B | AgentSpec | 23/33 (69.7%) | 0/33 | 10/33 | 2.55 | 1155.7 ms |
| Qwen3-VL-4B | Policy2Sample | 33/33 (100%) | 0/33 | 0/33 | 1.00 | 442.6 ms |
| Qwen3-VL-8B | Prompt-only | 2/33 (6.1%) | 31/33 | 0/33 | 1.00 | 619.4 ms |
| Qwen3-VL-8B | AgentSpec | 26/33 (78.8%) | 0/33 | 7/33 | 2.48 | 1557.0 ms |
| Qwen3-VL-8B | Policy2Sample | 33/33 (100%) | 0/33 | 0/33 | 1.00 | 607.5 ms |

The Policy2Sample arm uses an oracle-enumerated safe set as the sampling-time constraint. It is an idealized enforcement control and does not measure policy compilation. AgentSpec uses monitor feedback and rejected-candidate exclusion, with a three-generation total budget for the reported cross-model table. “Safe parameter pair” is a safety-compliance metric, not an independent task-goal success measure.

### AgentSpec budget sensitivity on 2B

With a maximum of 20 total generations, 2B AgentSpec produced safe parameter pairs in **33/33** cases, with **0** unsafe outputs and **0** no-actions. Mean calls were 4.55, the maximum actually used was 12, and mean latency was 1493.7 ms. The highest permitted speed was selected in 20/33 cases; the remaining outputs were safe but did not maximize that preference. See [`results_agentspec_budget20_2b.json`](../data/raw/results_agentspec_budget20_2b.json).

## Interpretation limits

This is a controlled finite-catalog experiment, not a benchmark drawn from a physical robot dataset. The auto-close timer and numeric envelope were study-authored synthetic Context, not hardware-calibrated. Each point was run once using greedy decoding; no multi-seed uncertainty is available. The paper reports these results as mechanism-level evidence, not as real-device safety or general task-success rates.

## Reproducibility and data

Configuration, environment, and commands are documented in [`RQ2/README.md`](../README.md) and [`code/run_config.json`](../code/run_config.json). Per-case output files are in [`data/raw/`](../data/raw/); the machine-readable summary table is [`rq2_moving_narrow_window_summary.csv`](rq2_moving_narrow_window_summary.csv).
