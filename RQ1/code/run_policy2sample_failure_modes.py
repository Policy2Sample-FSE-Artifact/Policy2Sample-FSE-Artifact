"""RQ2 mutation and fail-closed validation experiment."""

from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path

from policy2sample.compiler import PolicyCompiler
from policy2sample.dsl import DSLValidationError, validate_dsl_document
from policy2sample.pipeline import Policy2SamplePipeline
from semantic_ir import ContextSnapshot, OnlineExactProjectionBackend, ProjectionStatus


ROOT = Path(__file__).parent
OUTPUT = ROOT / "policy2sample_failure_modes_result.json"


def run(n_each: int = 50) -> dict[str, object]:
    compiler = PolicyCompiler()
    pipeline = Policy2SamplePipeline(compiler)
    policy = "靠近人时慢速向左移动，速度不超过 0.25 m/s。"
    compiled = compiler.compile(policy)
    assert compiled.dsl is not None and compiled.sketch is not None
    base_dsl = compiled.dsl
    base_snapshot = ContextSnapshot(
        {"human_distance": 1.2, "free_left": 0.8, "free_right": 1.6, "motion_budget": 1.8, "fragile": True},
        "ctx-valid",
    )
    deployment = compiler.deployment
    backend = OnlineExactProjectionBackend(
        deployment.schema,
        deployment.binding,
        deployment.base_contracts,
        deployment.context_schema,
    )
    rows: list[dict[str, object]] = []

    for index in range(n_each):
        result = pipeline.run_with_context_version(policy, base_snapshot, expected_context_version="ctx-new")
        rows.append({"kind": "stale_context", "expected": "STALE", "observed": result.realization.status if result.realization else None})

        invalid_unit = copy.deepcopy(base_dsl)
        invalid_unit["action_schema"]["fields"][1]["unit"] = "not-a-unit"  # type: ignore[index]
        try:
            validate_dsl_document(invalid_unit)
            observed = "ACCEPTED"
        except DSLValidationError:
            observed = "INVALID"
        rows.append({"kind": "invalid_unit", "expected": "INVALID", "observed": observed})

        unresolved = copy.deepcopy(base_dsl)
        unresolved["contracts"][0]["scope"]["fields"].append("ghost_field")  # type: ignore[index]
        try:
            validate_dsl_document(unresolved)
            observed = "ACCEPTED"
        except DSLValidationError:
            observed = "INVALID"
        rows.append({"kind": "unresolved_reference", "expected": "INVALID", "observed": observed})

        nonlinear = copy.deepcopy(base_dsl)
        nonlinear["contracts"][0]["capability"] = "unsupported_nonlinear"  # type: ignore[index]
        try:
            validate_dsl_document(nonlinear)
            observed = "ACCEPTED"
        except DSLValidationError:
            observed = "UNSUPPORTED"
        rows.append({"kind": "unsupported_nonlinear", "expected": "UNSUPPORTED", "observed": observed})

        contradictory = compiler.compile("向左或向右移动，靠近人时减速。")
        rows.append({"kind": "contradictory_policy", "expected": "AMBIGUOUS", "observed": contradictory.status.value})

        empty = pipeline.run(
            "靠近人时慢速向左移动。",
            ContextSnapshot(
                {"human_distance": 1.2, "free_left": 0.8, "free_right": 1.6, "motion_budget": 0.0, "fragile": True},
                f"ctx-empty-{index}",
            ),
        )
        rows.append({"kind": "empty_domain", "expected": "EMPTY", "observed": empty.realization.status if empty.realization else None})

        try:
            invalid_prefix = backend.project(base_snapshot, ("left", 0.8, 0.2, 99.0))
            observed_prefix = invalid_prefix.status.value
        except ValueError:
            observed_prefix = "INVALID"
        rows.append({"kind": "invalid_prefix", "expected": "INVALID", "observed": observed_prefix})

    counts = Counter((row["kind"], row["observed"]) for row in rows)
    return {
        "experiment": "rq2_fail_closed_failure_modes",
        "n_each": n_each,
        "n_cases": len(rows),
        "overall_accuracy": sum(row["expected"] == row["observed"] for row in rows) / len(rows),
        "by_kind": {
            kind: {
                "n": sum(row["kind"] == kind for row in rows),
                "accuracy": sum(row["kind"] == kind and row["expected"] == row["observed"] for row in rows) / sum(row["kind"] == kind for row in rows),
                "observed": {observed: count for (observed_kind, observed), count in counts.items() if observed_kind == kind},
            }
            for kind in sorted({row["kind"] for row in rows})
        },
        "rows": rows,
    }


def main() -> None:
    result = run()
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
