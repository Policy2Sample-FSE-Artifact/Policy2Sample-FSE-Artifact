"""Large differential test for continuous PPDC prefix-domain exactness."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

from continuous_polyhedral import (
    AffineContext,
    ContinuousCompiledPPDC,
    ContinuousLinearModel,
    ContinuousVariable,
    LinearInequality,
    online_exact_interval,
)


SEED = 20260916
TARGET_QUERIES = 10_000


def make_random_model(n_joint: int, rng: random.Random, model_id: int) -> ContinuousLinearModel:
    variables = tuple(ContinuousVariable(f"x{i}", 0.0, 1.0) for i in range(n_joint))
    constraints: list[LinearInequality] = []
    n_constraints = 2 + rng.randrange(4)
    for constraint_id in range(n_constraints):
        coefficients = {}
        for variable in variables:
            if rng.random() < 0.65:
                coefficients[variable.name] = float(rng.choice((-2, -1, 1, 2)))
        if not coefficients:
            coefficients[variables[rng.randrange(n_joint)].name] = 1.0
        rhs = AffineContext.lookup(("rhs", constraint_id))
        inequality = LinearInequality.make(coefficients, rhs)
        constraints.append(inequality)
        # Equality is represented exactly as two inequalities.
        if (model_id + constraint_id) % 7 == 0:
            constraints.append(
                LinearInequality.make(
                    {name: -coefficient for name, coefficient in coefficients.items()},
                    rhs.scale(-1.0),
                )
            )
    return ContinuousLinearModel(variables, tuple(constraints))


def make_context(model: ContinuousLinearModel, rng: random.Random) -> dict[str, object]:
    return {"rhs": [round(rng.uniform(-0.3, 2.0), 3) for _ in range(len(model.constraints) + 2)]}


def random_prefix(n_joint: int, rng: random.Random) -> tuple[float, ...]:
    length = rng.randrange(n_joint)
    values = []
    for _ in range(length):
        if rng.random() < 0.2:
            values.append(rng.choice((0.0, 1.0)))
        else:
            values.append(round(rng.uniform(-0.25, 1.25), 6))
    return tuple(values)


def same_interval(left, right) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return math.isclose(left.lower, right.lower, abs_tol=1e-8) and math.isclose(
        left.upper, right.upper, abs_tol=1e-8
    )


def run() -> dict[str, object]:
    rng = random.Random(SEED)
    quotas = {n_joint: TARGET_QUERIES // 6 for n_joint in range(1, 7)}
    for n_joint in range(1, TARGET_QUERIES % 6 + 1):
        quotas[n_joint] += 1
    rows = []
    mismatches = []
    empty_matches = 0
    total = 0
    for n_joint, quota in quotas.items():
        model_queries = 0
        while model_queries < quota:
            model = make_random_model(n_joint, rng, total)
            compiled = ContinuousCompiledPPDC.compile(model)
            for _ in range(min(10, quota - model_queries)):
                context = make_context(model, rng)
                specialized_context = compiled.specialize(context)
                prefix = random_prefix(n_joint, rng)
                actual = specialized_context.next_interval(prefix)
                expected = online_exact_interval(model, context, prefix)
                if actual is None and expected is None:
                    empty_matches += 1
                if not same_interval(actual, expected):
                    mismatches.append(
                        {"n_joint": n_joint, "prefix": prefix, "actual": repr(actual), "expected": repr(expected)}
                    )
                model_queries += 1
                total += 1
    for row_n in range(1, 7):
        rows.append({
            "n_joint": row_n,
            "queries": quotas[row_n],
            "mismatches": sum(item["n_joint"] == row_n for item in mismatches),
        })
    return {
        "experiment": "continuous_ppdc_differential_correctness",
        "seed": SEED,
        "queries": total,
        "n_joint_range": [1, 6],
        "coverage": ["random linear inequalities", "exact equality pairs", "bounds", "joint coupling", "empty domains", "changing contexts", "prefix lengths 0..n-1"],
        "exact_match_rate": 1.0 if not mismatches else (total - len(mismatches)) / total,
        "mismatch_count": len(mismatches),
        "empty_domain_matches": empty_matches,
        "by_n_joint": rows,
        "mismatches": mismatches[:20],
        "oracle": "independent repeated Fourier-Motzkin exact projection",
    }


def main() -> None:
    result = run()
    output = Path(__file__).with_name("differential_correctness_result.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
