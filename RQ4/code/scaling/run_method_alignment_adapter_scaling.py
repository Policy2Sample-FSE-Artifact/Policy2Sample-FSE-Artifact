"""Measure lexical mask cost against interval, precision, prefix and token complexity."""

from __future__ import annotations

import argparse
import itertools
import json
import platform
import statistics
import subprocess
import sys
import time
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

from policy2sample.fieldwise_adapter import NumericFieldMaskSession
from policy2sample.token_adapter import SimpleCharacterTokenizer
from tokenizer_fixtures import GreedyPieceTokenizer
from semantic_ir import IntervalUnion, RealInterval


def percentile(values, p):
    values = sorted(values)
    return values[min(len(values) - 1, round((len(values) - 1) * p))]


def make_domain(count, precision):
    intervals = []
    # Thin disjoint decimal bands; range cardinality rises rapidly with q,
    # while the new transition automaton operates on interval endpoints.
    for index in range(count):
        lo = Fraction(index * 5, 100)
        hi = lo + Fraction(4, 1000)
        intervals.append(RealInterval(float(lo), float(hi), True, True, lo, hi))
    return intervals[0] if count == 1 else IntervalUnion(tuple(intervals))


def run(repetitions):
    configs = itertools.product((1, 2, 4, 8, 16), (1, 2, 3), (0, 32, 128), ("char", "subword"))
    rows = []
    for interval_count, precision, prefix_length, token_complexity in configs:
        domain = make_domain(interval_count, precision)
        filler = '{"prior":"' + ("x" * max(0, prefix_length - 12)) + '","value": '
        if token_complexity == "char":
            tokenizer = SimpleCharacterTokenizer()
            encode = tokenizer
            decode = lambda ids: "".join(chr(i) for i in ids)
            candidates = tuple(ord(c) for c in "0123456789.-,xyz}")
        else:
            tokenizer = GreedyPieceTokenizer((
                '"value": ', "0.00", "0.05", "0.10", "0.15", "0.20", "0.", ".0", ".05", ".10", ".15",
                "00,", "05,", "10,", "15,", ",\"next\":", "0.05,", "0.10,", "0.15,",
                *tuple("prior"), *tuple("value"), *tuple("0123456789.-,xyz{}\" :"),
            ))
            encode = tokenizer.encode
            decode = tokenizer.decode
            candidates = tuple(tokenizer.ids.values())
        prefix_ids = encode(filler)
        current = encode(filler + "0.")
        build_ms = []
        transition_ms = []
        for _ in range(repetitions):
            started = time.perf_counter_ns()
            session = NumericFieldMaskSession(
                encode=encode,
                decode=decode,
                field_prefix_text=filler,
                field_prefix_token_ids=prefix_ids,
                domain=domain,
                precision=precision,
                delimiter=",",
            )
            build_ms.append((time.perf_counter_ns() - started) / 1e6)
            started = time.perf_counter_ns()
            session.mask(current, candidate_token_ids=candidates)
            transition_ms.append((time.perf_counter_ns() - started) / 1e6)
        rows.append({
            "interval_count": interval_count,
            "precision": precision,
            "prefix_characters": len(filler),
            "token_complexity": token_complexity,
            "candidate_tokens": len(candidates),
            "repetitions": repetitions,
            "representable_integer_ranges": len(session._scaled),
            "session_build_p50_us": percentile(build_ms, .50) * 1000,
            "session_build_p95_us": percentile(build_ms, .95) * 1000,
            "transition_p50_us": percentile(transition_ms, .50) * 1000,
            "transition_p95_us": percentile(transition_ms, .95) * 1000,
            "transition_p99_us": percentile(transition_ms, .99) * 1000,
            "mean_transition_us": statistics.mean(transition_ms) * 1000,
        })
    def sysctl(name):
        try:
            return subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            return None

    memory = sysctl("hw.memsize")
    return {
        "experiment": "fieldwise_adapter_scaling",
        "measurement": "field session compilation and one lexical mask step; exact projection excluded",
        "representation": "integer-lattice range automaton; no per-value candidate enumeration in tokenizer-aware path",
        "repetitions_per_configuration": repetitions,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "cpu": sysctl("machdep.cpu.brand_string") or platform.processor() or platform.machine(),
            "ram_bytes": int(memory) if memory and memory.isdigit() else None,
        },
        "factors": {
            "interval_count": [1, 2, 4, 8, 16],
            "precision": [1, 2, 3],
            "prefix_characters": [0, 32, 128],
            "token_complexity": ["char", "subword"],
        },
        "configuration_count": len(rows),
        "results": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=250)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.repetitions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"configuration_count": result["configuration_count"], "repetitions_per_configuration": args.repetitions}, indent=2))


if __name__ == "__main__":
    main()
