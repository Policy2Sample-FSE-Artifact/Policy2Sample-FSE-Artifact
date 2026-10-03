# Paper claims and artifact map

This guide maps the Evaluation section to the included evidence and gives the
offline commands for rebuilding derived tables. Run each command from the
artifact root. Python 3.10+ is sufficient for aggregation; model inference
requires separately obtained Qwen3-VL weights and the RQ-specific inference
environment.

| Paper claim | Included data | Rebuild / inspect |
|---|---|---|
| **RQ1, Table 1:** 54/55 Contracts validate; 47/50 emitted actions are source-safe; 47/54 masks match the source-rule catalog | `RQ1/data/rq1b_final_eval_input_55_2026-09-24.jsonl`; contributing raw runs in `RQ1/data/raw/` | `cd RQ1 && python3 code/aggregate_rq1_true_e2e_55case.py`; inspect `RQ1/results/rq1_true_e2e_8b_55case_merged.json` and `rq1_true_e2e_8b_55case_rows.csv` |
| **RQ2, Fig. 5:** safe-action yield by model size and retry budget | `RQ2/data/raw/results_moving_narrow_window_{2b,4b,8b}.json`; dedicated `results_agentspec_budget3_{2b,4b,8b}.json`; 2B budget-20 sensitivity file | `cd RQ2 && python3 code/aggregate_rq2_summary.py`; inspect `RQ2/results/rq2_moving_narrow_window_summary.csv` and `RQ2/results/RQ2_MOVING_NARROW_WINDOW_REPORT.md` |
| **RQ3, Fig. 6:** IoT-96 STC of 65.6%, 80.2%, and 91.7% | `RQ3/data/raw/iot_hard96/hard_iot_complexity_32b_20260929_v2.json` | `cd RQ3 && python3 code/aggregate_rq3_robot_iot.py`; inspect `RQ3/results/iot_hard96_summary.csv` and `RQ3/results/Hard-IoT_完整实验报告.md` |
| **RQ4, Fig. 7:** Robot and IoT latency overhead and preprocessing cost | `RQ4/data/raw/final_5reps_v3.json`; frozen inputs in `RQ4/data/raw/` | `cd RQ4 && python3 code/aggregate_rq4_latency.py`; inspect `RQ4/results/rq4_e2e_latency_summary.csv` and `RQ4/results/RQ4_E2E_LATENCY_8B_FULL_REPORT.md` |

The archived RQ3 package currently contains the IoT-96 evidence. The Robot-30
STC values are reported in the paper, but the matching raw Robot run is not
included in this package. The Robot-30 row therefore cannot be rebuilt from
the included files.

## Annotation provenance

The frozen RQ1 references were annotated independently by two human annotators
and disagreements were adjudicated by a third human reviewer, as described in
the paper. The 55-case input contains the final adjudicated references. The
separate pre-adjudication worksheets are not included, so the reported
56.4%–92.7% field-agreement range cannot be recalculated from this package.

## Inference requirements

The artifact does not include model weights. Raw-run aggregation does not
invoke a model. To rerun inference, follow the `Re-run inference` sections in
each RQ README and provide the matching model weights and inference runtime.
