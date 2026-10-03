# Policy2Sample — Anonymous FSE Artifact

This package is a reviewer-oriented, anonymized artifact for the Policy2Sample
submission. It organizes experiment code, raw evidence, and derived results by
research question. RQ1 is frozen; RQ2 contains the paper-reported
Context-moving narrow safety-domain study; RQ4 contains the archived 8B
end-to-end latency experiment. RQ3 retains the paper-aligned Hard-IoT-96 data;
its mismatched Robot Core-30 bundle was removed pending identification of the
paper's matching raw run.

## Start here

- **Paper claim-to-evidence map and rebuild commands:** [`REPRODUCTION_GUIDE.md`](REPRODUCTION_GUIDE.md)
- **License and reuse scope:** [`LICENSE`](LICENSE)
- **RQ1 instructions and reproduction:** [`RQ1/README.md`](RQ1/README.md)
- **RQ1 full experimental report:** [`RQ1/results/RQ1_TRUE_E2E_8B_55CASE_FULL_REPORT.md`](RQ1/results/RQ1_TRUE_E2E_8B_55CASE_FULL_REPORT.md)
- **RQ1 concise update:** [`RQ1/results/RQ1_TRUE_E2E_8B_55CASE_UPDATE.md`](RQ1/results/RQ1_TRUE_E2E_8B_55CASE_UPDATE.md)
- **Per-case results:** [`RQ1/results/rq1_true_e2e_8b_55case_rows.csv`](RQ1/results/rq1_true_e2e_8b_55case_rows.csv)
- **Machine-readable merged results:** [`RQ1/results/rq1_true_e2e_8b_55case_merged.json`](RQ1/results/rq1_true_e2e_8b_55case_merged.json)
- **RQ2 scope, reproduction and status:** [`RQ2/README.md`](RQ2/README.md)
- **RQ2 report:** [`RQ2/results/RQ2_MOVING_NARROW_WINDOW_REPORT.md`](RQ2/results/RQ2_MOVING_NARROW_WINDOW_REPORT.md)
- **RQ2 machine-readable summary:** [`RQ2/results/rq2_moving_narrow_window_summary.csv`](RQ2/results/rq2_moving_narrow_window_summary.csv)
- **RQ2 original per-case data:** [`RQ2/data/raw/`](RQ2/data/raw/)
- **RQ3 instructions, scope and results:** [`RQ3/README.md`](RQ3/README.md)
- **RQ3 data status and mismatch note:** [`RQ3/README.md`](RQ3/README.md)
- **RQ3 Hard-IoT-96 table:** [`RQ3/results/iot_hard96_summary.csv`](RQ3/results/iot_hard96_summary.csv)
- **RQ4 instructions and reproduction:** [`RQ4/README.md`](RQ4/README.md)
- **RQ4 full-action latency report:** [`RQ4/results/RQ4_E2E_LATENCY_8B_FULL_REPORT.md`](RQ4/results/RQ4_E2E_LATENCY_8B_FULL_REPORT.md)
- **RQ4 summary update:** [`RQ4/results/RQ4_E2E_LATENCY_8B_UPDATE.md`](RQ4/results/RQ4_E2E_LATENCY_8B_UPDATE.md)
- **RQ4 raw run data:** [`RQ4/data/raw/final_5reps_v3.json`](RQ4/data/raw/final_5reps_v3.json)

## Layout

```text
Policy2Sample_FSE_Artifact/
├── README.md
├── requirements.txt
├── requirements-inference.txt
├── RQ1/
│   ├── README.md
│   ├── code/
│   ├── data/
│   │   └── raw/
│   └── results/
├── RQ2/
│   ├── README.md
│   ├── code/
│   ├── data/raw/
│   └── results/
├── RQ3/
│   ├── README.md
│   ├── code/
│   ├── data/raw/
│   └── results/
└── RQ4/
    ├── README.md
    ├── code/
    ├── data/raw/
    └── results/
```

## Adding RQ2–RQ4 material

When adding experiment material, place it only under its matching `RQx/`
directory. Keep executable scripts/configuration in `code/`, original logs and
per-case outputs in `data/raw/`, and derived tables/figures/reports in
`results/`. Preserve raw outputs; do not replace them with aggregated values.
Record benchmark/case IDs, model and decoding configuration, seed, hardware,
run date, metric definitions, and exact command in that RQ's `README.md` or a
linked run manifest. Mark exploratory runs as exploratory and keep them
separate from final paper results. Do not modify frozen RQ1 or RQ4 while adding
another RQ. Update this top-level README with links and a short status note
when an RQ is added.

## RQ3 archive status

RQ3 currently retains Hard-IoT-96 (one greedy run), a controlled finite action
lattice using study-authored abstract energy units, not a SimuHome physical
replay. The earlier Robot Core-30 package had valid records for a different
utility metric, but did not support the paper's Robot Safe Task Completion
table; those copies were removed instead of relabeling them.

RQ4's v2 latency run is retained but explicitly marked exploratory/invalid for
latency reporting because its timing boundary omitted part of specialization;
only v3 supplies the reported result. The archived result removes remote host
and model-path metadata while retaining all case-level outputs and measurements.

For a new chat/session, provide the local artifact path and this README as the
source of truth before asking it to add results. The session does not
automatically inherit file locations or decisions from other chats.

## Privacy and scope

The package intentionally excludes local paths, account names, host addresses,
credentials, editor metadata, non-corpus exploratory cases, diagnostic-only
runs, and unrelated project files. Model weights are not included. The reported RQ1 experiment covers one finite-catalog fixture per
policy in the frozen 55-policy corpus; it is not a full simulator replay and
does not report task-goal success. See the RQ1 report for exact denominators,
failure analysis, and timing boundaries.
