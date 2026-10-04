"""Controlled finite-branch and relation-count scaling for the exact Realizer."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from fractions import Fraction
from pathlib import Path
import subprocess

import semantic_ir as ir


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * p
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def backend_for(kind: str, size: int, branch_budget: int):
    if kind == "active_contracts":
        schema = ir.ActionSchema((ir.FieldSpec.real("x0", 0.0, 1.0),), name="scale")
        contracts = tuple(
            ir.SafetyContract(
                name=f"cap_{i}", scope="scale", action_roles=(), context_roles=(),
                guard=lambda _ctx, _bind: True,
                constraint=lambda action, _ctx, _bind, cap=1.0 - (i + 1) / (size + 2): action["x0"] <= ir.z3.RealVal(str(cap)),
            )
            for i in range(size)
        )
        return ir.OnlineExactProjectionBackend(schema, ir.BindingEnvironment((), ()), contracts, {}, branch_budget=branch_budget)

    if kind == "finite_branches":
        enum_values = tuple(f"b{i}" for i in range(size))
        schema = ir.ActionSchema((ir.FieldSpec.real("x0", 0.0, 1.0), ir.FieldSpec.enum("mode", enum_values)), name="scale")
        def constraint(action, _ctx, _bind):
            return ir.z3.Or(*(
                ir.z3.And(
                    action["mode"] == i,
                    action["x0"] >= ir.z3.RealVal(str(Fraction(i * 2 + 1, size * 2 + 2))),
                    action["x0"] <= ir.z3.RealVal(str(Fraction(i * 2 + 2, size * 2 + 2))),
                ) for i in range(size)
            ))
        contract = ir.SafetyContract(
            "mode_bands", "scale", (), (), lambda _c, _b: True, constraint,
            fragment="bounded_linear_with_enum_branch",
        )
        return ir.OnlineExactProjectionBackend(schema, ir.BindingEnvironment((), ()), (contract,), {}, branch_budget=branch_budget)

    if kind == "cross_field_relations":
        fields = tuple(ir.FieldSpec.real(f"x{i}", 0.0, 1.0) for i in range(size + 1))
        schema = ir.ActionSchema(fields, name="scale")
        contract = ir.SafetyContract(
            "cross_relations", "scale", (), (), lambda _c, _b: True,
            lambda action, _ctx, _bind: ir.z3.And(*(
                action["x0"] <= action[f"x{i}"] for i in range(1, size + 1)
            )),
        )
        return ir.OnlineExactProjectionBackend(schema, ir.BindingEnvironment((), ()), (contract,), {}, branch_budget=branch_budget)

    raise ValueError(kind)


def run(repetitions: int, branch_budget: int) -> dict:
    snapshot = ir.ContextSnapshot({}, "scaling-fixed-context")
    sweeps = {
        "active_contracts": (1, 2, 4, 8, 16),
        "finite_branches": (1, 2, 4, 8, 16),
        "cross_field_relations": (0, 1, 2, 4, 8),
    }
    summaries = []
    raw = []
    for kind, sizes in sweeps.items():
        for size in sizes:
            backend = backend_for(kind, size, branch_budget)
            prefix = ()
            timings = []
            statuses = {}
            for _ in range(repetitions):
                start = time.perf_counter_ns()
                result = backend.project(snapshot, prefix)
                timings.append((time.perf_counter_ns() - start) / 1e6)
                statuses[result.status.value] = statuses.get(result.status.value, 0) + 1
            summaries.append({
                "factor": kind,
                "factor_value": size,
                "n_fields": len(backend.schema.fields),
                "repetitions": repetitions,
                "status_counts": statuses,
                "p50_ms": percentile(timings, 0.50),
                "p95_ms": percentile(timings, 0.95),
                "p99_ms": percentile(timings, 0.99),
                "mean_ms": statistics.mean(timings),
                "max_ms": max(timings),
            })
            raw.append({"factor": kind, "factor_value": size, "latency_ms": timings})
    def sysctl(name: str) -> str | None:
        try:
            completed = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, check=True)
            return completed.stdout.strip()
        except Exception:
            return None

    memory_bytes = sysctl("hw.memsize")
    return {
        "experiment": "method_alignment_exact_realizer_scaling",
        "measurement": "one fresh-context exact next-domain projection; backend construction excluded",
        "branch_budget": branch_budget,
        "repetitions_per_configuration": repetitions,
        "sweeps": list(sweeps),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "z3": ir.z3.get_version_string(),
            "cpu": sysctl("machdep.cpu.brand_string") or platform.processor() or platform.machine(),
            "ram_bytes": int(memory_bytes) if memory_bytes and memory_bytes.isdigit() else None,
            "logical_cpus": __import__("os").cpu_count(),
        },
        "summaries": summaries,
        "raw": raw,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=250)
    parser.add_argument("--branch-budget", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.repetitions, args.branch_budget)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for row in result["summaries"]:
        print(row)


if __name__ == "__main__":
    main()
