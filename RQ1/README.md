# RQ1 — Runtime diagnostics

This directory contains the archived runtime diagnostics cited in the RQ1 evaluation table: two 10,000-query domain comparisons, 350 fail-closed probes, 1,648 relational-composition queries, and tokenizer transition records. The Qwen3-VL-8B tokenizer record is shared with [RQ4](../RQ4/data/raw/rq4_qwen8b_tokenizer_mask_differential_20260928.json); combined with the TinyLlama tokenizer record here, the two experiments cover 66,313 transitions with no observed false admissions or false rejections.

## Human annotation provenance

The frozen 55-case source-rule references were annotated independently by two human annotators. One separate human reviewer resolved disagreements before the references were frozen. The case input is `data/rq1b_final_eval_input_55_2026-09-24.jsonl`.

## Archived evidence

- `data/raw/continuous_domain_differential_10000.json`: continuous PPDC differential result, 10,000 queries, independent repeated Fourier–Motzkin projection oracle.
- `data/raw/policy2sample_differential_10000.json`: DSL-to-sample differential result, 10,000 queries, independent interval/enumeration oracle.
- `data/raw/fail_closed_350_cases.json`: seven fail-closed categories, 50 probes each.
- `data/raw/joint_projection_1648_queries.json`: joint composition before projection; independent projection has false admissions in 824 queries; joint composition eliminates them in the same batch.
- `data/raw/tinyllama_tokenizer_differential_33160.json`: tokenizer-only differential at TinyLlama revision `5243d15`; no language-model inference was run for this check.
- The Qwen3-VL-8B transition record, including its full-mask latency measurements, is in `RQ4/data/raw/rq4_qwen8b_tokenizer_mask_differential_20260928.json`.

## Rebuild the paper-table counts

From `RQ1/`, run `python3 code/summarize_runtime_diagnostics.py`. It reads the archived summaries and prints the diagnostic counts, including the cross-folder tokenizer total.

## Reproduce the archived deterministic checks

From `RQ1/`, run the continuous-domain check with Python 3.10+:

```bash
python3 code/run_differential_correctness.py
```

Run the DSL and fail-closed checks with:

```bash
PYTHONPATH=code python3 code/run_policy2sample_differential.py
PYTHONPATH=code python3 code/run_policy2sample_failure_modes.py
```

These commands regenerate result files under `code/` from their recorded seeds. The relational-composition JSON is the archived controlled batch; its query-level rows are retained in the raw file.

To repeat the TinyLlama tokenizer check, obtain `tokenizer.json` from the upstream model repository at revision `5243d15` and use the RQ4 tokenizer runner with the archived runtime modules:

```bash
curl -L https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/5243d15/tokenizer.json -o /tmp/tinyllama-tokenizer.json
python3 ../RQ4/code/scaling/run_real_tokenizer_differential.py \
  --source-root ../RQ4/code/scaling/runtime \
  --tokenizer-json /tmp/tinyllama-tokenizer.json \
  --tokenizer-source "TinyLlama 1.1B Chat tokenizer.json, revision 5243d15" \
  --seed 20260929 --transitions-per-case 100 --random-candidates 40 \
  --output data/raw/reproduced_tinyllama_tokenizer_differential.json
```

The [upstream TinyLlama model repository](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) identifies the model under Apache-2.0; the tokenizer fixture is not included in this package. The test uses its tokenizer only, not its model weights.
