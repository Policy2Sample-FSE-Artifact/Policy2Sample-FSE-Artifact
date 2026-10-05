"""Print the counts cited in the RQ1 runtime-diagnostics table."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    continuous = read(RAW / "continuous_domain_differential_10000.json")
    dsl = read(RAW / "policy2sample_differential_10000.json")
    fail_closed = read(RAW / "fail_closed_350_cases.json")
    joint = read(RAW / "joint_projection_1648_queries.json")
    tiny = read(RAW / "tinyllama_tokenizer_differential_33160.json")
    qwen = read(ROOT.parent / "RQ4" / "data" / "raw" / "rq4_qwen8b_tokenizer_mask_differential_20260928.json")

    print(f"Differential domain queries: {continuous['queries'] + dsl['queries']}")
    print(f"Differential mismatches: {continuous['mismatch_count'] + dsl['mismatch_count']}")
    print(f"Fail-closed probes: {fail_closed['n_cases']}; expected outcomes: {fail_closed['overall_accuracy']:.0%}")
    print(f"Tokenizer transitions: {qwen['checked_transitions'] + tiny['checked_transitions']}")
    print(f"Tokenizer false admissions: {qwen['false_admission_count'] + tiny['false_admission_count']}")
    print(f"Tokenizer false rejections: {qwen['false_rejection_count'] + tiny['false_rejection_count']}")
    print(f"Relational-composition queries: {joint['queries']}")
    print(f"Queries with independent-projection false admissions: {joint['queries_with_false_admission']}")
    print(f"False-admitted grid values: {joint['total_false_admission_grid_values']}")


if __name__ == "__main__":
    main()
