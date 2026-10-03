#!/usr/bin/env python3
"""Export the frozen RQ4 latency summaries as a compact CSV."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "final_5reps_v3.json"
OUT = ROOT / "results" / "rq4_e2e_latency_summary.csv"

data = json.loads(RAW.read_text(encoding="utf-8"))
rows = []
for domain in ("robot", "iot"):
    summary = data["summaries"][domain]
    for method in ("schema_only", "policy2sample"):
        item = summary[method]
        rows.append({
            "domain": domain,
            "method": method,
            "n": item["n"],
            "mean_latency_ms": item["mean_ms"],
            "p50_latency_ms": item["p50_ms"],
            "p95_latency_ms": item["p95_ms"],
            "mean_output_tokens": item["mean_output_tokens"],
            "p50_output_tokens": item["p50_output_tokens"],
            "p95_output_tokens": item["p95_output_tokens"],
            "paired_mean_delta_ms": "",
            "paired_p50_delta_ms": "",
            "paired_p95_delta_ms": "",
            "relative_mean_overhead_pct": ""
        })
    paired = summary["paired"]
    rows.append({
        "domain": domain,
        "method": "paired_delta_policy2sample_minus_schema_only",
        "n": paired["n_pairs"],
        "mean_latency_ms": "",
        "p50_latency_ms": "",
        "p95_latency_ms": "",
        "mean_output_tokens": "",
        "p50_output_tokens": "",
        "p95_output_tokens": "",
        "paired_mean_delta_ms": paired["mean_absolute_delta_ms"],
        "paired_p50_delta_ms": paired["p50_absolute_delta_ms"],
        "paired_p95_delta_ms": paired["p95_absolute_delta_ms"],
        "relative_mean_overhead_pct": paired["relative_overhead_pct_of_schema_mean"]
    })

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(f"wrote {OUT}")
