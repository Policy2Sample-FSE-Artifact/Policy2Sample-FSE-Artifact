# RQ3 — Safety and task utility

This artifact presents the controlled Hard-IoT-96 workload and its results.

Hard-IoT-96 uses a study-authored finite action lattice and abstract energy
coefficients (`6h + 4d + 5w + 8e ≤ B`) to compare the four methods under a
controlled workload.

## Hard-IoT-96 structural-complexity workload

- Qwen3-VL-32B-Instruct-Q8_0, greedy decoding, temperature 0, fixed seed
  20260929; run 2026-09-29; 96 main cases × four methods = 384 method/case rows.
- 24 cases each in C1–C4, crossed with loose/medium/tight admissible-domain
  retention bins (8 cases per cell). The complete discrete action space has
  3,888 candidates.
- Methods: Prompt-only, Schema-only, AgentSpec (up to two repairs), and
  Policy2Sample. One additional prefix-dead-end diagnostic is embedded in the
  raw result but excluded from the 96-case main table. The earlier pilot log is
  retained for audit and is explicitly invalid/excluded (stale logits path).

| Method | First admissible | Final safe | Task/nearest-safe-target success | Mean calls | Mean generation latency |
|---|---:|---:|---:|---:|---:|
| Prompt-only | 69/96 (71.9%) | 69/96 (71.9%) | 63/96 (65.6%) | 1.00 | 1302.8 ms |
| Schema-only | 70/96 (72.9%) | 70/96 (72.9%) | 63/96 (65.6%) | 1.00 | 1370.5 ms |
| AgentSpec | 70/96 (72.9%) | 90/96 (93.8%) | 77/96 (80.2%) | 1.40 | 1941.2 ms |
| Policy2Sample | 96/96 (100%) | 96/96 (100%) | 88/96 (91.7%) | 1.00 | 1362.4 ms |

The energy model uses study-authored abstract load units.

Files:

- Raw 96-case result, progress log, preflight/sanity and exploratory diagnostic
  logs: [`data/raw/iot_hard96/`](data/raw/iot_hard96/)
- Runner and typed-domain dependency:
  [`code/hard_iot_complexity_experiment.py`](code/hard_iot_complexity_experiment.py),
  [`code/rq3_8b_z3_domains.py`](code/rq3_8b_z3_domains.py)
- Full source report and design record:
  [`results/Hard-IoT_完整实验报告.md`](results/Hard-IoT_完整实验报告.md),
  [`results/IOT_EXPERIMENT_FINAL_DESIGN.md`](results/IOT_EXPERIMENT_FINAL_DESIGN.md)
- Derived tables and per-case/method export:
  [`results/iot_hard96_summary.csv`](results/iot_hard96_summary.csv),
  [`results/iot_hard96_by_complexity.csv`](results/iot_hard96_by_complexity.csv),
  [`results/iot_hard96_by_retention.csv`](results/iot_hard96_by_retention.csv),
  [`results/iot_hard96_case_method_results.csv`](results/iot_hard96_case_method_results.csv),
  [`results/iot_prefix_dead_end_diagnostic.csv`](results/iot_prefix_dead_end_diagnostic.csv)

## Rebuild derived tables

From this directory:

```bash
python3 code/aggregate_rq3_robot_iot.py
```

This reads the retained IoT run and regenerates its CSV summaries; it does not
invoke a model or modify raw evidence. See the JSON run configurations in
`code/` for model, seeds, hardware and command templates. Model weights and the
pinned upstream AgentSpec checkout are external prerequisites for inference.
