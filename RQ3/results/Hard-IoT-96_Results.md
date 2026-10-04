# Hard-IoT-96 results

This report summarizes the controlled IoT workload used for the RQ3 table.
The run uses Qwen3-VL-32B-Instruct-Q8_0 with greedy decoding and one fixed
seed. It contains 96 cases evaluated by four methods on a finite action lattice
of 3,888 configurations. The energy coefficients are study-authored abstract
load units.

| Method | Safe-action yield | Safe task completion | Mean generation latency |
|---|---:|---:|---:|
| Prompt-only | 69/96 (71.9%) | 63/96 (65.6%) | 1,302.8 ms |
| AgentSpec | 90/96 (93.8%) | 77/96 (80.2%) | 1,941.2 ms |
| Policy2Sample | 96/96 (100%) | 88/96 (91.7%) | 1,362.4 ms |

Safe-action yield counts outputs that satisfy the frozen safety rules.
Safe task completion additionally requires the output to be safe and achieve
the frozen configuration objective. For this workload, task success means
selecting a safe assignment with minimum defined loss to the requested target.
The objective is evaluated over the discrete configurations.

## Rebuild

From `RQ3/`, regenerate the derived CSV tables with:

```bash
python3 code/aggregate_rq3_robot_iot.py
```

The command reads `data/raw/iot_hard96/hard_iot_complexity_32b_20260929_v2.json`.
The frozen case inputs and run settings are in `data/raw/iot_hard96/` and
`code/iot_hard96_run_config.json`.
