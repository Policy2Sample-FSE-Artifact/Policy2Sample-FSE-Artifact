# RQ3 — Safety and task utility

This artifact presents the Hard-IoT-96 controlled workload and its results.

Hard-IoT-96 uses a study-authored finite action lattice and abstract energy
coefficients (`6h + 4d + 5w + 8e ≤ B`) to compare the four methods under a
controlled workload.

## Hard-IoT-96 structural-complexity workload

- Qwen3-VL-32B-Instruct-Q8_0, greedy decoding, temperature 0, fixed seed
  20260929; run 2026-09-29; 96 main cases × four methods = 384 method/case rows.
- 24 cases each in C1–C4, crossed with loose/medium/tight admissible-domain
  retention bins (8 cases per cell). The complete discrete action space has
  3,888 candidates.
- Methods: Prompt-only, AgentSpec (up to two repairs), and Policy2Sample.

| Method | Safe-action yield | Safe task completion | Mean generation latency |
|---|---:|---:|---:|
| Prompt-only | 69/96 (71.9%) | 63/96 (65.6%) | 1302.8 ms |
| AgentSpec | 90/96 (93.8%) | 77/96 (80.2%) | 1941.2 ms |
| Policy2Sample | 96/96 (100%) | 88/96 (91.7%) | 1362.4 ms |

The energy model uses study-authored abstract load units.

Files:

- Raw 96-case result and frozen preflight input:
  [`data/raw/iot_hard96/`](data/raw/iot_hard96/)
- Runner and typed-domain dependency:
  [`code/hard_iot_complexity_experiment.py`](code/hard_iot_complexity_experiment.py),
  [`code/rq3_8b_z3_domains.py`](code/rq3_8b_z3_domains.py)
- Paper-facing result summary:
  [`results/Hard-IoT-96_Results.md`](results/Hard-IoT-96_Results.md)
- Derived tables and per-case/method export:
  [`results/iot_hard96_summary.csv`](results/iot_hard96_summary.csv),
  [`results/iot_hard96_by_complexity.csv`](results/iot_hard96_by_complexity.csv),
  [`results/iot_hard96_by_retention.csv`](results/iot_hard96_by_retention.csv),
  [`results/iot_hard96_case_method_results.csv`](results/iot_hard96_case_method_results.csv)

## Rebuild derived tables

From this directory:

```bash
python3 code/aggregate_rq3_robot_iot.py
```

This reads the retained IoT run and regenerates its CSV summaries; it does not
invoke a model or modify raw evidence. See the JSON run configurations in
`code/` for model, seeds, hardware and command templates. Model weights and the
pinned upstream AgentSpec checkout are external prerequisites for inference.
