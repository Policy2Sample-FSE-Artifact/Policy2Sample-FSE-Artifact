# RQ4 — End-to-end action-generation latency

This directory archives the 8B system-level latency experiment comparing a
structured-generation configuration with runtime safety constraints disabled
to Policy2Sample. The archived runner labels this arm `schema-only`; it uses
the same structured action grammar without the safety mask. The reported run is
`v3`.

## Final result

| Workload | Schema-only mean / P50 / P95 (ms) | Policy2Sample mean / P50 / P95 (ms) | Mean overhead |
|---|---:|---:|---:|
| Robot Core-30 v2 | 707.85 / 758.27 / 780.58 | 741.28 / 780.07 / 867.32 | +33.44 ms / +4.72% |
| Hard-IoT Core-30 | 422.89 / 422.78 / 440.96 | 506.88 / 508.57 / 533.92 | +83.99 ms / +19.86% |

Each domain contains 30 cases; each case/method was measured five times with
paired, alternating order: 600 measured generations total, plus four excluded
warmups. In IoT, mean output length changes from 28.0 to 33.5 tokens, so this
is a system-level latency comparison.

## Package contents

- `code/remote_rq4_e2e_latency_8b.py`: runner.
- `code/policy2sample/`: the two runtime modules imported by the runner.
- `code/run_config.json`: model identity, experiment settings, timing boundary,
  and an invocation template.
- `code/requirements-rq4.txt`: request-level inference dependencies.
- `code/aggregate_rq4_latency.py`: exports the archived JSON summaries to CSV.
- `data/raw/`: final v3 result and frozen input snapshots.
- `results/`: paper-facing latency and scaling summary, and generated latency
  CSV.
- `data/raw/rq4_realizer_scaling_20260928.json`: 250 repetitions for each
  active-contract, finite-branch, and cross-field-relation configuration.
- `data/raw/rq4_adapter_scaling_20260928.json`: 250 repetitions for each of 90
  controlled field-mask configurations.
- `data/raw/rq4_qwen8b_tokenizer_mask_differential_20260928.json`: differential
  measurements using the tokenizer embedded in Qwen3-VL-8B. The recorded
  machine-local model path is omitted from this public copy.
- `code/scaling/`: runners and minimal runtime modules for those measurements.

The archived JSON preserves all run rows and timing observations. Model names
are kept without machine-local paths.

## Rebuild the component measurements

The raw summaries in `data/raw/` are the measurements cited in the paper. To
repeat the controlled solver and adapter sweeps, run from `RQ4/` with Python,
`z3-solver`, and the included runtime modules:

```bash
python3 code/scaling/run_method_alignment_realizer_scaling.py \
  --repetitions 250 --branch-budget 256 \
  --output data/raw/reproduced_realizer_scaling.json

PYTHONPATH=code/scaling/runtime python3 code/scaling/run_method_alignment_adapter_scaling.py \
  --repetitions 250 --output data/raw/reproduced_adapter_scaling.json
```

To repeat the tokenizer differential, obtain the same Qwen3-VL-8B GGUF and a
compatible `llama-cpp-python` build, then run:

```bash
python3 code/scaling/run_real_tokenizer_differential.py \
  --source-root code/scaling/runtime \
  --model-path /path/to/Qwen3VL-8B-Instruct-Q8_0.gguf \
  --seed 20260928 --transitions-per-case 100 --random-candidates 40 \
  --output data/raw/reproduced_tokenizer_mask_differential.json
```

The runner records only the model filename, not the supplied local path.

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
llama-cpp-python 0.3.33, and tensor split `0.45,0.55`.

Set model, vocabulary-cache, and output paths for the target machine, then run
the invocation template in `code/run_config.json` from `RQ4/`. This is a finite
discrete action-lattice experiment, not continuous numeric token enforcement
or simulator/physical replay. The timing boundary includes per-request P2S
projection, safe-candidate serialization and prefix-trie specialization,
prompt prefill, constrained decoding, and JSON parsing; it excludes model
loading and one-time vocabulary/grammar/lattice preparation.
