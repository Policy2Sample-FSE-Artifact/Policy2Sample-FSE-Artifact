#!/usr/bin/env python3
"""Rebuild the paper-facing RQ2 summary from archived per-case JSON runs."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "results" / "rq2_moving_narrow_window_summary.csv"
FIELDS = [
    "model", "method", "safe_actions", "total_cases", "unsafe_outputs",
    "no_action", "max_permitted_speed_hits", "mean_calls", "mean_latency_ms",
    "result_source", "status",
]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def summary_row(model_size: str, method_label: str, source: str, summary: dict) -> dict:
    return {
        "model": model_size.upper(),
        "method": method_label,
        "safe_actions": summary["safe_success"],
        "total_cases": summary["n"],
        "unsafe_outputs": summary["unsafe_emitted"],
        "no_action": summary["no_action"],
        "max_permitted_speed_hits": summary["max_speed_choice"],
        "mean_calls": f"{summary['mean_calls']:.2f}",
        "mean_latency_ms": f"{summary['mean_latency_ms']:.1f}",
        "result_source": source,
        "status": "paper-reported controlled result",
    }


rows = []
for size in ("2b", "4b", "8b"):
    base_name = f"results_moving_narrow_window_{size}.json"
    base = load(RAW / base_name)
    for method_key, label in (
        ("prompt_only", "Prompt-only"),
        ("policy2sample_oracle_mask", "Policy2Sample"),
    ):
        rows.append(summary_row(size, label, base_name, base["summary"][method_key]))

    retry_name = f"results_agentspec_budget3_{size}.json"
    retry = load(RAW / retry_name)
    rows.append(summary_row(
        size, "AgentSpec (3-call setting)", retry_name,
        retry["summary"]["agentspec_llm_self_examine"],
    ))

budget_name = "results_agentspec_budget20_2b.json"
budget = load(RAW / budget_name)
rows.append(summary_row(
    "2b", "AgentSpec (20-call sensitivity)", budget_name,
    budget["summary"]["agentspec_llm_self_examine"],
))

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", encoding="utf-8-sig", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)
print(f"Wrote {len(rows)} RQ2 summary rows to {OUT}")
