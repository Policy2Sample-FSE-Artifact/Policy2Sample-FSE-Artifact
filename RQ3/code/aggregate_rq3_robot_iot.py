#!/usr/bin/env python3
"""Export derived RQ3 tables from the archived Robot Core-30 and Hard-IoT-96."""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
OUT = ROOT / "results"


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def robot_exports() -> list[dict]:
    files = sorted((RAW / "robot_core30").glob("robot_core30_result_seed*.json"))
    per_seed: list[dict] = []
    case_rows: list[dict] = []
    for path in files:
        d = json.loads(path.read_text(encoding="utf-8"))
        seed = int(d["seed"])
        for method in ("prompt_only", "agentspec_llm_self_examine", "policy2sample"):
            group = [r for r in d["rows"] if r["method"] == method]
            per_seed.append({
                "seed": seed, "method": method, "n": len(group),
                "first_safe": sum(bool(r["first_action_safe"]) for r in group),
                "final_safe": sum(bool(r["final_safe"]) for r in group),
                "nearest_safe_target_matches": sum(bool(r["nearest_safe_target_match"]) for r in group),
                "mean_calls": statistics.mean(r["model_calls"] for r in group),
                "mean_retries": statistics.mean(r["retries"] for r in group),
                "mean_latency_ms": statistics.mean(r["latency_ms"] for r in group),
            })
        for r in d["rows"]:
            case_rows.append({
                "case_id": r["case_id"], "seed": seed, "family": r["family"], "method": r["method"],
                "first_action": r["first_action"], "first_action_safe": r["first_action_safe"],
                "final_action": r["final_action"], "final_safe": r["final_safe"],
                "nearest_safe_target_match": r["nearest_safe_target_match"],
                "target_loss": r["target_loss"], "oracle_optimal_loss": r["oracle_optimal_loss"],
                "oracle_loss_gap": r["oracle_loss_gap"], "model_calls": r["model_calls"],
                "retries": r["retries"], "latency_ms": r["latency_ms"],
                "safety_violations": json.dumps(r["safety_violations"], ensure_ascii=False),
                "active_rules": json.dumps(r["active_rules"], ensure_ascii=False),
            })
    write_csv(OUT / "robot_core30_seed_summary.csv", per_seed, list(per_seed[0]))
    write_csv(OUT / "robot_core30_case_seed_results.csv", case_rows, list(case_rows[0]))

    summary: list[dict] = []
    for method in ("prompt_only", "agentspec_llm_self_examine", "policy2sample"):
        rows = [r for r in per_seed if r["method"] == method]
        rate_cols = ("first_safe", "final_safe", "nearest_safe_target_matches")
        out = {"method": method, "n_seed_case_rows": sum(r["n"] for r in rows)}
        for col in rate_cols:
            vals = [100 * r[col] / r["n"] for r in rows]
            m, sd = mean_sd(vals)
            out[f"{col}_mean_pct"] = m
            out[f"{col}_sd_pp"] = sd
        for col in ("mean_calls", "mean_retries", "mean_latency_ms"):
            m, sd = mean_sd([r[col] for r in rows])
            out[f"{col}_mean"] = m
            out[f"{col}_sd"] = sd
        out["unsafe_first_total"] = sum(r["n"] - r["first_safe"] for r in rows)
        out["unsafe_final_total"] = sum(r["n"] - r["final_safe"] for r in rows)
        summary.append(out)
    write_csv(OUT / "robot_core30_5seed_summary.csv", summary, list(summary[0]))
    return summary


def iot_exports() -> list[dict]:
    src = RAW / "iot_hard96/hard_iot_complexity_32b_20260929_v2.json"
    d = json.loads(src.read_text(encoding="utf-8"))
    case_rows = []
    diagnostic_rows = []
    for r in d["rows"]:
        row = {
            "case_id": r["case_id"], "complexity": r["complexity"], "rho_bin": r["rho_bin"],
            "method": r["method"], "first_admissible": r["first_admissible"],
            "final_admissible": r["final_admissible"], "task_success": r["task_success"],
            "unsafe_proposals": r["unsafe_proposals"], "dead_end": r["dead_end"],
            "repairs": r["repairs"], "model_calls": r["model_calls"],
            "latency_ms": r["total_generation_latency_ms"],
            "final_action": json.dumps(r["final_action"], ensure_ascii=False),
            "final_validation": json.dumps(r["final_validation"], ensure_ascii=False),
            "generations": json.dumps(r["generations"], ensure_ascii=False),
        }
        (diagnostic_rows if r["diagnostic_case"] else case_rows).append(row)
    write_csv(OUT / "iot_hard96_case_method_results.csv", case_rows, list(case_rows[0]))
    write_csv(OUT / "iot_prefix_dead_end_diagnostic.csv", diagnostic_rows, list(diagnostic_rows[0]))

    overall: list[dict] = []
    by_complexity: list[dict] = []
    by_rho: list[dict] = []
    for method, s in d["summary"].items():
        overall.append({
            "method": method, "n": s["n"], "first_admissible": s["first_admissible"],
            "final_admissible": s["final_admissible"], "task_success": s["task_success"],
            "unsafe_proposals": s["unsafe_proposals"], "dead_end_first_proposal": s["dead_end_first_proposal"],
            "mean_repairs": s["mean_repairs"], "mean_model_calls": s["mean_model_calls"],
            "mean_latency_ms": s["mean_latency_ms"],
        })
        for level, v in s["by_complexity"].items():
            by_complexity.append({"method": method, "complexity": level, "n": v["n"],
                                  "first_admissible": v["first_admissible"], "final_admissible": v["final_admissible"],
                                  "task_success": v["task_success"], "mean_model_calls": v["mean_calls"],
                                  "mean_latency_ms": v["mean_latency_ms"]})
        for level, v in s["by_rho"].items():
            by_rho.append({"method": method, "retention_bin": level, "n": v["n"],
                           "first_admissible": v["first_admissible"], "final_admissible": v["final_admissible"],
                           "task_success": v["task_success"], "mean_model_calls": v["mean_calls"],
                           "mean_latency_ms": v["mean_latency_ms"]})
    write_csv(OUT / "iot_hard96_summary.csv", overall, list(overall[0]))
    write_csv(OUT / "iot_hard96_by_complexity.csv", by_complexity, list(by_complexity[0]))
    write_csv(OUT / "iot_hard96_by_retention.csv", by_rho, list(by_rho[0]))
    return overall


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    robot_files = sorted((RAW / "robot_core30").glob("robot_core30_result_seed*.json"))
    robot = robot_exports() if robot_files else None
    iot = iot_exports()
    print(json.dumps({"robot_methods": robot, "iot_methods": iot}, indent=2))


if __name__ == "__main__":
    main()
