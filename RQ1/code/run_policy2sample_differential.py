"""Independent DSL-to-sample differential test for the move vertical slice."""

from __future__ import annotations

import json
import math
import random
import argparse
from pathlib import Path

from policy2sample.compiler import PolicyCompiler
from policy2sample.pipeline import Policy2SamplePipeline
from semantic_ir import ContextSnapshot, ProjectionStatus


SEED = 20260917
TARGET_QUERIES = 10_000
OUTPUT = Path(__file__).with_name("policy2sample_differential_result.json")


def independent_domain(sketch, context, prefix):
    """Independent closed-form oracle, intentionally separate from Z3 backend."""
    requested = sketch.requested_direction
    directions = (requested,) if requested else ("left", "right")
    require_motion = sketch.require_motion
    distance_lower = 0.01 if require_motion else 0.0
    # The deployment vertical slice has a base human-proximity contract in
    # addition to any policy-specific guard.
    active_human_guard = float(context["human_distance"]) < 1.5
    if sketch.human_distance_guard is not None:
        active_human_guard = active_human_guard or (
            float(context["human_distance"]) < float(sketch.human_distance_guard)
        )
    speed_upper = 1.0
    if active_human_guard:
        speed_upper = min(speed_upper, 0.3)
    if bool(context["fragile"]):
        speed_upper = min(speed_upper, 0.5)
    if sketch.speed_cap is not None:
        speed_upper = min(speed_upper, float(sketch.speed_cap))

    if len(prefix) == 0:
        feasible = []
        for direction in directions:
            clearance = float(context["free_left" if direction == "left" else "free_right"])
            if distance_lower <= min(clearance, float(context["motion_budget"])) + 1e-12:
                feasible.append(direction)
        return tuple(feasible) if feasible else None

    direction = prefix[0]
    clearance = float(context["free_left" if direction == "left" else "free_right"])
    if len(prefix) == 1:
        upper = min(3.0, clearance, float(context["motion_budget"]))
        return None if upper + 1e-12 < distance_lower else (distance_lower, upper)

    distance = float(prefix[1])
    if distance > clearance + 1e-12 or distance < distance_lower - 1e-12:
        return None
    upper = min(speed_upper, (float(context["motion_budget"]) - distance) / 2.0)
    return None if upper < -1e-12 else (0.0, upper)


def same_domain(actual, expected) -> bool:
    if expected is None:
        return actual.status == ProjectionStatus.EMPTY
    if isinstance(expected, tuple):
        if actual.status != ProjectionStatus.EXACT or actual.domain is None:
            return False
        if isinstance(expected[0], str):
            return tuple(actual.domain.values) == expected
        return math.isclose(actual.domain.lower, expected[0], abs_tol=1e-9) and math.isclose(
            actual.domain.upper, expected[1], abs_tol=1e-9
        )
    return False


def random_policy(rng: random.Random) -> str:
    direction = rng.choice(("向左", "向右", ""))
    guard = rng.choice(("靠近人时", "", "人距离 1.5 m 内"))
    cap = rng.choice(("速度不超过 0.2 m/s", "速度不超过 0.3 m/s", "", "低速"))
    return f"{guard}{direction}移动，{cap}。"


def random_context(rng: random.Random) -> dict[str, object]:
    return {
        "human_distance": round(rng.uniform(0.5, 4.0), 3),
        "free_left": round(rng.uniform(-0.2, 2.5), 3),
        "free_right": round(rng.uniform(-0.2, 2.5), 3),
        "motion_budget": round(rng.uniform(0.0, 2.8), 3),
        "fragile": bool(rng.randrange(2)),
    }


def run() -> dict[str, object]:
    rng = random.Random(SEED)
    compiler = PolicyCompiler()
    pipeline = Policy2SamplePipeline(compiler)
    mismatches = []
    status_counts = {"EXACT": 0, "EMPTY": 0}
    queries = 0
    policy_counts = {"generated": 0, "executable": 0}

    while queries < TARGET_QUERIES:
        policy = random_policy(rng)
        compiled = compiler.compile(policy)
        policy_counts["generated"] += 1
        if compiled.sketch is None or compiled.dsl is None:
            continue
        policy_counts["executable"] += 1
        context = random_context(rng)
        # A fresh backend is built by the pipeline, while the oracle is
        # computed independently from the sketch and context.
        backend = pipeline._backend_for(compiled)
        snapshot = ContextSnapshot(context, f"diff-{queries}")
        prefix_options = [()]
        requested = compiled.sketch.requested_direction
        direction_options = (requested,) if requested else ("left", "right")
        for direction in direction_options:
            prefix_options.append((direction,))
            clearance = context["free_left" if direction == "left" else "free_right"]
            upper = min(float(clearance), float(context["motion_budget"]))
            if upper >= (0.01 if compiled.sketch.require_motion else 0.0):
                prefix_options.append((direction, round(rng.uniform(max(0.0, 0.01 if compiled.sketch.require_motion else 0.0), upper), 2)))
        for prefix in prefix_options:
            actual = backend.project(snapshot, prefix)
            expected = independent_domain(compiled.sketch, context, prefix)
            if not same_domain(actual, expected):
                mismatches.append({"policy": policy, "context": context, "prefix": prefix, "actual": repr(actual), "expected": repr(expected)})
            if actual.status.value in status_counts:
                status_counts[actual.status.value] += 1
            queries += 1
            if queries >= TARGET_QUERIES:
                break

    return {
        "experiment": "policy2sample_dsl_to_sample_differential",
        "seed": SEED,
        "queries": queries,
        "policy_counts": policy_counts,
        "status_counts": status_counts,
        "exact_match_rate": 1.0 if not mismatches else (queries - len(mismatches)) / queries,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:20],
        "oracle": "independent closed-form interval/enumeration oracle, not Realizer or Z3",
        "coverage": ["direction enum", "context guards", "human cap", "fragile cap", "policy cap", "motion budget", "positive-distance task constraint", "empty domains", "prefixes"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
