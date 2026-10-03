# RQ1 — Natural-language policy to masked structured action

This folder contains the complete RQ1 evidence package: frozen source-policy
records, raw run outputs, runner/validator code, a deterministic aggregation
script, and the final per-case and narrative results.

## Result at a glance

The 55-policy slice contains 25 Robot and 30 IoT policies. In the recorded
Qwen3-VL-8B, temperature-0.2 run, 54/55 Contracts passed structural/type
validation, 50 actions were emitted, and 47/50 emitted actions satisfied the
independent source-rule oracle. Three compiled masks admitted four unsafe
catalog candidates; the full report separates compile rejection, empty domains,
selected-action violations, and exact-mask accuracy.

## Where to find things

- `data/rq1b_final_eval_input_55_2026-09-24.jsonl`: frozen 55-policy input.
- `data/raw/`: only the canonical 15 prior rows, the final IO07 row, and the
  fixed-seed 39-row completion output.
- `code/remote_rq1_true_e2e_8b.py`: core Contract generation, typed validation,
  source-rule oracle, finite-catalog comparison, and trie-masked action choice.
- `code/action_trie_kv_rollback_demo.py` and
  `code/rq1_masked_action_decoder.py`: the minimal trie/logit primitives used
  by this RQ1 runner; unrelated development scenarios are omitted.
- `code/remote_rq1_true_e2e_8b_remaining38.py`: completion runner. It loads the
  frozen corpus, builds the 38 previously uncovered fixtures plus the IO06
  replacement, and supports `--only-case` for a focused diagnostic.
- `code/remote_rq1_io07_compound_guard_regression.py`: runner for the IO07
  compound-guard case used in the final corpus result.
- `code/aggregate_rq1_true_e2e_55case.py`: rebuilds the merged JSON, CSV, and
  both Markdown reports from the frozen input and raw result files.
- `results/`: final merged output and paper-oriented reports.

## Reproduce the reported aggregation

Python 3.10+ is sufficient for the offline aggregation; it uses only the
standard library:

```bash
python3 code/aggregate_rq1_true_e2e_55case.py
```

Run this command from `RQ1/` (or pass the script path while preserving its
relative `data/` and `results/` layout). It asserts that the merged result has
exactly one row for every frozen policy ID.

## Re-run the 39-row completion inference

Inference requires a compatible `llama-cpp-python` build with Qwen3-VL support
and a separately obtained Qwen3-VL-8B GGUF model. Set the model path without
editing the package:

```bash
export QWEN3_VL_8B_MODEL=/path/to/Qwen3VL-8B-Instruct-Q8_0.gguf
python3 code/remote_rq1_true_e2e_8b_remaining38.py \
  --temperature 0.2 --seed 20261006 \
  --output data/raw/rq1_true_e2e_8b_remaining39_20261006.json
```

The recorded experiment ran on a dual-RTX-3090 host; inference dependencies and
model weights are intentionally not bundled. Re-running model inference is
stochastic and need not reproduce byte-identical Contract text. The included
raw outputs are precisely the run records contributing to the final 55 rows.

## Important interpretation

The oracle evaluates the source-rule semantics directly against the fixed
Context and each catalog action; it does not inspect the generated Contract or
mask. A syntactically/type-valid Contract can still be semantically wrong, so
selected-action compliance and candidate-mask soundness are reported
separately. Empty predicted domains count as no-action/failure, not as successful
task completion. This experiment has no simulator replay or task-goal metric.

The 55 frozen semantic references were produced by two independent human
annotators; a third human reviewer adjudicated disagreements. The exact
pre-adjudication field agreement reported in the paper ranges from 56.4% to
92.7%. The frozen input records the final adjudicated reference for each rule.
