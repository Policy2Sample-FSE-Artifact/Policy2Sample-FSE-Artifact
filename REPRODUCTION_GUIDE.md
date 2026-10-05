# Paper claims and artifact map

This guide maps the evaluation claims to the included evidence and gives offline commands for rebuilding derived tables. Run each command from the artifact root. Python 3.10+ is sufficient for aggregation; model inference requires separately obtained Qwen3-VL weights and the RQ-specific inference environment.

| Paper claim | Included data | Rebuild / inspect |
|---|---|---|
| **RQ1 runtime diagnostics:** 20,000 differential queries, 350 fail-closed probes, 1,648 relational-composition queries (824 contain false admissions under independent projection), and 66,313 tokenizer transitions | `RQ1/data/raw/`; Qwen3-VL-8B tokenizer record shared under `RQ4/data/raw/` | From `RQ1/`, run `python3 code/summarize_runtime_diagnostics.py` to print all archived table counts. To regenerate deterministic differential records, run `python3 code/run_differential_correctness.py`, `PYTHONPATH=code python3 code/run_policy2sample_differential.py`, and `PYTHONPATH=code python3 code/run_policy2sample_failure_modes.py`; see `RQ1/README.md` for the archived joint-projection and tokenizer records |
| **RQ2, Fig. 5:** safe-action yield by model size and generation budget | `RQ2/data/raw/results_moving_narrow_window_{2b,4b,8b}.json`; dedicated `results_agentspec_budget3_{2b,4b,8b}.json`; 2B budget-20 sensitivity file | `cd RQ2 && python3 code/aggregate_rq2_summary.py`; inspect `RQ2/results/rq2_moving_narrow_window_summary.csv` and `RQ2/results/RQ2_MOVING_NARROW_WINDOW_REPORT.md` |
| **RQ3, Fig. 6 (Hard-IoT-96):** safe-action yield and task completion | `RQ3/data/raw/iot_hard96/hard_iot_complexity_32b_20260929_v2.json` | `cd RQ3 && python3 code/aggregate_rq3_robot_iot.py`; inspect `RQ3/results/iot_hard96_summary.csv` and `RQ3/results/Hard-IoT-96_Results.md` |
| **RQ4, Fig. 7:** request-level Robot and IoT latency | `RQ4/data/raw/final_5reps_v3.json`; frozen inputs in `RQ4/data/raw/` | `cd RQ4 && python3 code/aggregate_rq4_latency.py`; inspect `RQ4/results/rq4_e2e_latency_summary.csv` |
| **RQ4:** solver scaling and tokenizer-facing semantic-mask cost | `RQ4/data/raw/rq4_realizer_scaling_20260928.json`, `rq4_adapter_scaling_20260928.json`, and `rq4_qwen8b_tokenizer_mask_differential_20260928.json` | See `RQ4/README.md` for the three replay commands and the reported percentile fields. |

## Annotation provenance

The frozen RQ1 references were annotated independently by two human annotators and disagreements were adjudicated by a third human reviewer, as described in the paper. The 55-case input contains the final adjudicated references.

## Inference requirements

The artifact does not include model weights. Raw-run aggregation does not invoke a model. To rerun inference, follow the `Re-run inference` sections in each RQ README and provide the matching model weights and inference runtime.
