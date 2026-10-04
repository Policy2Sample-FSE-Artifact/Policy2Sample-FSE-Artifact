# RQ4 results

This package contains the Qwen3-VL-8B request-level comparison and the
component measurements reported in the paper. The request-level comparator
uses the same structured action setup with runtime safety constraints
disabled.

## Request-level latency

| Workload | Safety disabled (mean, ms) | Policy2Sample (mean, ms) | Paired mean difference | Mean output tokens |
|---|---:|---:|---:|---:|
| Robot Core-30 | 707.85 | 741.28 | +33.44 ms (+4.72%) | 41.9 → 41.9 |
| Hard-IoT Core-30 | 422.89 | 506.88 | +83.99 ms (+19.86%) | 28.0 → 33.5 |

Each workload contains 30 cases and five measured repetitions per case and
method, after four warm-up generations. Per-request timing includes setup
through parsing the completed action and excludes shared one-time
initialization. The IoT difference includes the observed output-length change.

## Constraint Solver scaling

Each configuration was measured 250 times on the recorded Apple M2 Pro
environment. Backend construction is excluded; the measurement is one exact
fresh-context domain projection.

| Factor | Values | P50 range (ms) | P95 range (ms) |
|---|---|---:|---:|
| Active Contracts | 1, 2, 4, 8, 16 | 2.25–4.71 | 2.52–5.56 |
| Finite branches | 1, 2, 4, 8, 16 | 2.51–35.08 | 3.02–36.87 |
| Cross-field relations | 0, 1, 2, 4, 8 | 2.05–4.54 | 2.46–4.87 |

## Tokenizer-facing semantic mask

Using the tokenizer embedded in Qwen3-VL-8B, the complete semantic mask over
the current structural candidate tokens took 0.761 ms at P50 and 1.366 ms at
P95. The differential checked 33,153 transitions across the recorded numeric
and delimiter cases, with exact agreement and no false admissions or
rejections. Each full mask covered an average of 41.43 candidate token IDs.

The separate 90-configuration adapter study used synthetic tokenizer fixtures
and measured 250 repetitions per configuration. Its per-configuration P50 for
one lexical mask transition ranged from 54.8 to 550.0 μs; exact projection was
excluded from this measurement.

The raw records and replay commands are listed in this directory's
[`README.md`](../README.md).
