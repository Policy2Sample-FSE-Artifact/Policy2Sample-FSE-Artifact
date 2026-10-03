# RQ4 — End-to-end action-generation latency

This directory archives the 8B system-level latency experiment comparing the
schema-only finite-lattice decoder with Policy2Sample. The final reported run
is the timing-correct `v3` run. No RQ1 files or frozen RQ1 labels were changed.

## Final result

| Workload | Schema-only mean / P50 / P95 (ms) | Policy2Sample mean / P50 / P95 (ms) | Mean overhead |
|---|---:|---:|---:|
| Robot Core-30 v2 | 707.85 / 758.27 / 780.58 | 741.28 / 780.07 / 867.32 | +33.44 ms / +4.72% |
| Hard-IoT Core-30 | 422.89 / 422.78 / 440.96 | 506.88 / 508.57 / 533.92 | +83.99 ms / +19.86% |

Each domain contains 30 cases; each case/method was measured five times with
paired, alternating order: 600 measured generations total, plus four excluded
warmups. The full report discusses the IoT output-token difference (28.0 vs
33.5 mean tokens): a case-level audit found identical canonical serialization
rules but different selected action values, so E2E latency is not a pure mask
overhead measurement.

## Package contents

- `code/remote_rq4_e2e_latency_8b.py`: runner.
- `code/policy2sample/`: the two runtime modules imported by the runner.
- `code/run_config.json`: model identity, experiment settings, timing boundary,
  and an invocation template.
- `code/requirements-rq4.txt`: inference dependencies; XGrammar is intentionally
  unpinned because the recorded runtime metadata did not preserve its version.
- `code/aggregate_rq4_latency.py`: exports the archived JSON summaries to CSV.
- `data/raw/`: final v3 result, original per-run progress logs, frozen input
  snapshots, and the prior exploratory/invalid-timing v2 result.
- `results/`: concise update, full report, and generated summary CSV.

The archived JSON preserves all run rows and timing observations. To follow the
artifact's privacy rules, its remote model path and host name are omitted; model
alias and SHA-256 are retained. The inference script likewise avoids emitting
host/account paths.

## Rebuild the summary table

From this directory:

```bash
python3 code/aggregate_rq4_latency.py
```

The script reads `data/raw/final_5reps_v3.json` and writes
`results/rq4_e2e_latency_summary.csv`.

## Re-run inference

Inference requires the Qwen3-VL-capable `llama-cpp-python`, XGrammar, NumPy,
and a separately obtained GGUF whose SHA-256 matches `code/run_config.json`.
The recorded host used two RTX 3090 GPUs, Python 3.11.15,
llama-cpp-python 0.3.33, and tensor split `0.45,0.55`. The archived result
recorded XGrammar's runtime version as `unknown`; this is a provenance
limitation and is not silently filled with an assumed version.

Set model, vocabulary-cache, and output paths for the target machine, then run
the invocation template in `code/run_config.json` from `RQ4/`. This is a finite
discrete action-lattice experiment, not continuous numeric token enforcement
or simulator/physical replay. The timing boundary includes per-request P2S
projection, safe-candidate serialization and prefix-trie specialization,
prompt prefill, constrained decoding, and JSON parsing; it excludes model
loading and one-time vocabulary/grammar/lattice preparation.

## Raw-run status

- `final_5reps_v3.json` and its progress JSONL are the final timing-correct run
  and are used by the report.
- `final_5reps_v2_exploratory_invalid_timing.json` and its progress JSONL are
  retained for audit only. The v2 timer excluded part of per-request P2S
  specialization; do not cite or aggregate its latency.
