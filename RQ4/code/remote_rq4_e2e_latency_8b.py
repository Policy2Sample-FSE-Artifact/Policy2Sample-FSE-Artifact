"""Matched XGrammar-only vs XGrammar+Policy2Sample action latency (Qwen3-VL-8B).

This runner reuses the frozen Robot Core-30 v2 and Hard-IoT Core-30 inputs.
It measures complete JSON action generation over each benchmark's declared
finite action lattice; it is not a continuous-domain or physical-replay test.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import xgrammar as xgr
import llama_cpp
from llama_cpp import Llama

sys.path.insert(0, str(Path(__file__).resolve().parent / "policy2sample"))
from dynamic_matcher import Policy2SampleDynamicMatcher  # noqa: E402
from token_adapter import TokenMask  # noqa: E402


ROBOT_DIRECTIONS = ("forward", "left", "right")
ROBOT_DISTANCES = (0.10, 0.50, 0.90, 1.20)
ROBOT_SPEEDS = (0.10, 0.25, 0.40, 0.60, 0.80)
ROBOT_YAWS = (-30, -15, 0, 15, 30)
ROBOT_LIFTS = (0.10, 0.30, 0.50, 0.70)
ROBOT_EXTENSIONS = (0.00, 0.10, 0.20, 0.30, 0.40)
PAYLOAD_MARGIN = {"none": 0.00, "rigid": 0.05, "heavy": 0.10, "fragile": 0.15}
PAYLOAD_SIDE = {"none": 0.10, "rigid": 0.10, "heavy": 0.15, "fragile": 0.20}
IOT_LEVELS = (0, 20, 40, 60, 80, 100)
IOT_MODES = ("eco", "normal", "boost")


def compact(obj: object) -> str:
    # Match the stable property order used by the JSON-schema compiler input.
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def fmt2(value: float) -> str:
    return f"{float(value):.2f}"


def robot_action_obj(action: tuple | str) -> dict[str, object]:
    if action == "STOP":
        return {"action": "STOP"}
    direction, distance, speed, yaw, lift, extension = action
    return {
        "action": "MOVE_CARRY", "direction": direction,
        "distance": fmt2(distance), "speed": fmt2(speed),
        "yaw_rate": str(int(yaw)), "lift_height": fmt2(lift),
        "extension": fmt2(extension),
    }


def robot_safe(ctx: dict[str, object], a: tuple) -> bool:
    direction, dist, speed, yaw, lift, ext = a
    if ctx["human_in_corridor"]:
        hd = float(ctx["human_distance_m"])
        if hd <= 0.40:
            return False
        if hd < 1.60 and speed > min(0.60, 0.50 * (hd - 0.40)) + 1e-9:
            return False
        if hd >= 1.60 and speed > 0.60 + 1e-9:
            return False
    elif speed > 0.80 + 1e-9:
        return False
    payload = str(ctx["payload"])
    clearance = float(ctx["clearance_m"][direction])
    if dist + ext + 0.10 + PAYLOAD_MARGIN[payload] > clearance + 1e-9:
        return False
    if ctx["zone"] == "aisle" and 0.55 + 2 * ext + PAYLOAD_SIDE[payload] > float(ctx["aisle_width_m"]) + 1e-9:
        return False
    if payload in ("heavy", "fragile") and speed / 0.60 + abs(yaw) / 90.0 + lift / 1.20 > 1.0 + 1e-9:
        return False
    if ctx["floor"] == "low_friction" and (speed > 0.25 + 1e-9 or abs(yaw) > 15):
        return False
    if ctx["corner"] == "occluded" and (speed > 0.20 + 1e-9 or ext > 0.20 + 1e-9):
        return False
    if ctx["corner"] == "occluded" and payload == "fragile" and abs(yaw) > 15:
        return False
    if ctx["zone"] == "handover" and ctx["human_in_corridor"] and float(ctx["human_distance_m"]) <= 0.50:
        return False
    return True


def robot_universe():
    return tuple(itertools.product(ROBOT_DIRECTIONS, ROBOT_DISTANCES, ROBOT_SPEEDS,
                                   ROBOT_YAWS, ROBOT_LIFTS, ROBOT_EXTENSIONS))


def iot_universe():
    return tuple((m, h, d, w, e) for m in IOT_MODES for h in IOT_LEVELS
                 for d in IOT_LEVELS for w in IOT_LEVELS for e in IOT_LEVELS)


def iot_power(a: tuple) -> int:
    _, h, d, w, e = a
    return 6 * h + 4 * d + 5 * w + 8 * e


def iot_safe(case: dict[str, object], a: tuple) -> bool:
    mode, h, d, w, e = a
    c = case["context"]
    level = int(str(case.get("complexity", case.get("family")))[1])
    if iot_power(a) > c["budget"]:
        return False
    if level >= 2:
        if c["window"] == "open" and h > 40:
            return False
        if c["grid"] == "constrained" and e > 20:
            return False
        if c["water_temp_c"] >= 55 and w > 20:
            return False
    if level >= 3 and (h + d > 180 or e + w > 160):
        return False
    if level >= 4:
        if mode == "eco" and (h > 80 or d > 80 or e > 80):
            return False
        if mode == "normal" and not (20 <= h <= 100):
            return False
        if mode == "boost" and not (h >= 80 and d >= 60):
            return False
    return True


def iot_action_obj(a: tuple) -> dict[str, object]:
    mode, h, d, w, e = a
    return {"action": "CONFIGURE_ENERGY", "mode": mode,
            "h": int(h), "d": int(d), "w": int(w), "e": int(e)}


def iot_policy(case: dict[str, object]) -> str:
    c = case["context"]
    level = int(str(case.get("complexity", case.get("family")))[1])
    s = f"Shared circuit load 6*h+4*d+5*w+8*e must be <= {c['budget']}."
    if level >= 2:
        s += (f" With window={c['window']}, grid={c['grid']}, water temperature={c['water_temp_c']} C: "
              "if window=open then h<=40; if grid=constrained then e<=20; "
              "if water temperature>=55 C then w<=20.")
    if level >= 3:
        s += " Couplings: h+d<=180 and e+w<=160."
    if level >= 4:
        s += " Mode rules: eco requires h<=80,d<=80,e<=80; normal requires 20<=h<=100; boost requires h>=80,d>=60."
    return s


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fast_ids_from_bitmask(bitmask, vocab_size: int) -> set[int]:
    # XGrammar stores packed uint32 words in a CPU tensor. Vectorized expansion
    # avoids Python-level scans over ~150k vocabulary positions on every token.
    words = np.asarray(bitmask[0], dtype=np.int32).view(np.uint32)
    bits = np.unpackbits(words.view(np.uint8), bitorder="little")[:vocab_size]
    return set(np.flatnonzero(bits).tolist())


def _fast_adapter_bitmask_ids(bitmask, index: int = 0) -> set[int]:
    words = np.asarray(bitmask[index], dtype=np.int32).view(np.uint32)
    vocab_size = int(bitmask.shape[1]) * 32
    bits = np.unpackbits(words.view(np.uint8), bitorder="little")[:vocab_size]
    return set(np.flatnonzero(bits).tolist())


# The adapter's reference implementation expands packed masks in Python. Use
# an equivalent vectorized unpacker here so this end-to-end run measures the
# same mask semantics without spending most time on Python bit iteration.
Policy2SampleDynamicMatcher._ids_from_bitmask = staticmethod(_fast_adapter_bitmask_ids)


def ids_for_text(llm: Llama, text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in llm.tokenize(text.encode("utf-8"), add_bos=False, special=True))


def prefix_map_for(ids_by_candidate: dict[str, tuple[int, ...]], safe_texts: list[str]):
    mapping: dict[tuple[int, ...], set[int]] = {}
    for text in safe_texts:
        ids = ids_by_candidate[text]
        for i, token_id in enumerate(ids):
            mapping.setdefault(ids[:i], set()).add(token_id)
    return mapping


class PrefixProvider:
    def __init__(self, prefix_map: dict[tuple[int, ...], set[int]]) -> None:
        self.prefix_map = prefix_map

    def __call__(self, binding, prefix_text, prefix_token_ids):
        del binding, prefix_text
        return TokenMask(frozenset(self.prefix_map.get(tuple(prefix_token_ids), set())))


def json_schema_for(domain: str, kind: str) -> dict[str, object]:
    if domain == "robot" and kind == "stop":
        return {"type": "object", "properties": {"action": {"const": "STOP"}},
                "required": ["action"], "additionalProperties": False}
    if domain == "robot":
        props = {
            "action": {"const": "MOVE_CARRY"},
            "direction": {"type": "string", "enum": list(ROBOT_DIRECTIONS)},
            "distance": {"type": "string", "enum": [fmt2(x) for x in ROBOT_DISTANCES]},
            "speed": {"type": "string", "enum": [fmt2(x) for x in ROBOT_SPEEDS]},
            "yaw_rate": {"type": "string", "enum": [str(x) for x in ROBOT_YAWS]},
            "lift_height": {"type": "string", "enum": [fmt2(x) for x in ROBOT_LIFTS]},
            "extension": {"type": "string", "enum": [fmt2(x) for x in ROBOT_EXTENSIONS]},
        }
        return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}
    props = {
        "action": {"const": "CONFIGURE_ENERGY"},
        "mode": {"type": "string", "enum": list(IOT_MODES)},
        "h": {"type": "integer", "enum": list(IOT_LEVELS)},
        "d": {"type": "integer", "enum": list(IOT_LEVELS)},
        "w": {"type": "integer", "enum": list(IOT_LEVELS)},
        "e": {"type": "integer", "enum": list(IOT_LEVELS)},
    }
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def compile_grammars(llm: Llama, vocab: list[bytes]):
    info = xgr.TokenizerInfo(vocab, xgr.VocabType.RAW, vocab_size=llm.n_vocab(),
                             stop_token_ids=[llm.token_eos()])
    compiler = xgr.GrammarCompiler(info, cache_enabled=True)
    result = {}
    timings = {}
    for key in (("robot", "move"), ("robot", "stop"), ("iot", "all")):
        schema = json_schema_for(*key)
        t0 = time.perf_counter_ns()
        result[key] = compiler.compile_json_schema(json.dumps(schema, sort_keys=True))
        timings[f"{key[0]}_{key[1]}_compile_ms"] = (time.perf_counter_ns() - t0) / 1e6
    return result, timings


def build_prompt(domain: str, case: dict[str, object]) -> str:
    if domain == "robot":
        return ("Choose one structured action for the request. Follow the safety policy and current context. "
                f"Safety policy: {case['safety_policy']}\nContext: {compact(case['context'])}\n"
                f"Task: {case['goal']}\nReturn only the action JSON.")
    return ("Choose one structured device action for the request. Follow the safety policy and current context. "
            f"Safety policy: {iot_policy(case)}\nContext: {compact(case['context'])}\n"
            f"Task: configure devices while satisfying the stated request; target={compact(case['target'])}. "
            "Return only the action JSON.")


def run_one(llm: Llama, grammar, domain: str, case: dict[str, object], mode: str,
            universe: tuple, ids_by_candidate: dict[str, tuple[int, ...]],
            universe_prefix_maps: dict[tuple[str, str], dict[tuple[int, ...], set[int]]],
            max_tokens: int = 160,
            warmup: bool = False) -> dict[str, object]:
    start_ns = time.perf_counter_ns()
    case_id = str(case.get("case_id", case.get("id")))
    is_stop = domain == "robot" and bool(case.get("expected_stop", False))
    if domain == "robot":
        kind = "stop" if is_stop else "move"
        expected_count = 1 if is_stop else int(case.get("safe_move_count", 0))
    else:
        kind = "all"
        expected_count = int(case.get("safe_domain_size", 0))

    safe_actions = ()
    candidate_texts: list[str] | None = None
    project_ms = serialization_ms = prefix_trie_ms = 0.0
    prefix_map = None
    if mode == "policy2sample":
        projection_started = time.perf_counter_ns()
        if is_stop:
            safe_actions = ("STOP",)
        elif domain == "robot":
            safe_actions = tuple(a for a in universe if robot_safe(case["context"], a))
        else:
            safe_actions = tuple(a for a in universe if iot_safe(case, a))
        project_ms = (time.perf_counter_ns() - projection_started) / 1e6
        if len(safe_actions) != expected_count:
            raise RuntimeError(f"safe-domain count mismatch for {case_id}: {len(safe_actions)} != {expected_count}")
        if not safe_actions:
            raise RuntimeError(f"empty safe action domain: {case_id}")
        serialization_started = time.perf_counter_ns()
        if domain == "robot":
            candidate_texts = [compact(robot_action_obj(a)) for a in safe_actions]
        else:
            candidate_texts = [compact(iot_action_obj(a)) for a in safe_actions]
        serialization_ms = (time.perf_counter_ns() - serialization_started) / 1e6
        trie_started = time.perf_counter_ns()
        prefix_map = prefix_map_for(ids_by_candidate, candidate_texts)
        prefix_trie_ms = (time.perf_counter_ns() - trie_started) / 1e6
    specialize_ms = project_ms + serialization_ms + prefix_trie_ms

    prompt = build_prompt(domain, case)
    llm.reset()
    prompt_ids = llm.tokenize(prompt.encode("utf-8"), add_bos=True, special=True)
    prefill0 = time.perf_counter_ns()
    llm.eval(prompt_ids)
    prefill_ms = (time.perf_counter_ns() - prefill0) / 1e6

    matcher = xgr.GrammarMatcher(grammar[(domain, kind)], terminate_without_stop_token=True)
    if mode == "policy2sample":
        provider = PrefixProvider(prefix_map)
        runtime_matcher = Policy2SampleDynamicMatcher(
            matcher, llm.n_vocab(), provider,
            token_decoder=lambda tid: llm.detokenize([int(tid)]).decode("utf-8", errors="replace"),
            bitmask_factory=xgr.allocate_token_bitmask,
        )
        runtime_matcher.bind(scene_id=domain, rule_set_id=f"{domain}-frozen-v1",
                             context_version=case_id, context=case.get("context", {}))
    else:
        runtime_matcher = None

    generated_ids: list[int] = []
    generated = ""
    mask_total_ms = 0.0
    llm_eval_ms = 0.0
    mask_steps = 0
    status = "MAX_TOKENS"
    for _ in range(max_tokens):
        m0 = time.perf_counter_ns()
        if mode == "policy2sample":
            _, decision = runtime_matcher.fill_next_token_mask()
            allowed = runtime_matcher._last_allowed
        else:
            bitmask = xgr.allocate_token_bitmask(1, llm.n_vocab())
            matcher.fill_next_token_bitmask(bitmask)
            structural_ids = fast_ids_from_bitmask(bitmask, llm.n_vocab())
            canonical_ids = universe_prefix_maps[(domain, kind)].get(tuple(generated_ids), set())
            allowed = structural_ids & canonical_ids
        mask_total_ms += (time.perf_counter_ns() - m0) / 1e6
        mask_steps += 1
        if not allowed:
            status = "EMPTY_MASK"
            break
        logits = np.asarray(llm.scores[-1], dtype=np.float32)
        next_id = max(allowed, key=lambda tid: float(logits[tid]))
        e0 = time.perf_counter_ns()
        llm.eval([int(next_id)])
        llm_eval_ms += (time.perf_counter_ns() - e0) / 1e6
        if mode == "policy2sample":
            runtime_matcher.accept_token(int(next_id))
            generated = runtime_matcher.prefix_text
        else:
            if not matcher.accept_token(int(next_id)):
                status = "GRAMMAR_REJECTED"
                break
            generated_ids.append(int(next_id))
            generated += llm.detokenize([int(next_id)]).decode("utf-8", errors="replace")
        if matcher.is_terminated():
            status = "COMPLETE"
            break

    parse0 = time.perf_counter_ns()
    try:
        parsed = json.loads(generated) if status == "COMPLETE" else None
        parse_ok = isinstance(parsed, dict)
        if not parse_ok:
            status = "PARSE_FAILED"
    except Exception:
        parsed, parse_ok = None, False
        status = "PARSE_FAILED"
    parse_ms = (time.perf_counter_ns() - parse0) / 1e6
    total_ms = (time.perf_counter_ns() - start_ns) / 1e6
    output_text_safe = generated in set(candidate_texts) if candidate_texts is not None else None
    return {
        "case_id": case_id, "domain": domain, "method": mode,
        "status": status, "parsed": parsed, "parse_ok": parse_ok,
        "policy_candidate_match": output_text_safe,
        "safe_domain_size": expected_count, "universe_size": len(universe) if not is_stop else 1,
        "output_tokens": len(generated_ids) if mode == "schema_only" else len(runtime_matcher.prefix_token_ids),
        "prompt_tokens": len(prompt_ids), "output_text": generated,
        "project_ms": project_ms, "serialization_ms": serialization_ms,
        "prefix_trie_ms": prefix_trie_ms, "specialize_ms": specialize_ms,
        "prompt_prefill_ms": prefill_ms,
        "mask_ms": mask_total_ms, "model_eval_ms": llm_eval_ms, "parse_ms": parse_ms,
        "total_action_latency_ms": total_ms,
        "mask_steps": mask_steps,
        "warmup": warmup,
    }


def read_hardware() -> dict[str, object]:
    # Keep the anonymous artifact free of the remote account/host identifier.
    d = {"platform": platform.platform(), "python": platform.python_version()}
    try:
        d["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                                  capture_output=True, text=True, timeout=5, check=True).stdout.strip().splitlines()
    except Exception:
        d["gpu"] = "unavailable"
    return d


def load_cases(robot_path: Path, iot_path: Path):
    robot_doc = json.loads(robot_path.read_text(encoding="utf-8"))
    iot_doc = json.loads(iot_path.read_text(encoding="utf-8"))
    robot_cases = robot_doc["cases"]
    iot_cases = iot_doc["cases"]
    if len(robot_cases) != 30 or len(iot_cases) != 30:
        raise ValueError(f"expected 30+30 cases, got {len(robot_cases)}+{len(iot_cases)}")
    for case in robot_cases:
        case["safety_policy"] = robot_doc["safety_policy"]
    return robot_cases, iot_cases


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    out = {}
    for domain in ("robot", "iot"):
        out[domain] = {}
        for method in ("schema_only", "policy2sample"):
            vals = [float(r["total_action_latency_ms"]) for r in rows
                    if r["domain"] == domain and r["method"] == method and not r["warmup"]]
            tokens = [float(r["output_tokens"]) for r in rows
                      if r["domain"] == domain and r["method"] == method and not r["warmup"]]
            out[domain][method] = {
                "n": len(vals), "mean_ms": statistics.mean(vals), "p50_ms": percentile(vals, 50),
                "p95_ms": percentile(vals, 95), "mean_output_tokens": statistics.mean(tokens),
                "p50_output_tokens": percentile(tokens, 50), "p95_output_tokens": percentile(tokens, 95),
            }
        b = out[domain]["schema_only"]
        p = out[domain]["policy2sample"]
        by_key = {}
        for row in rows:
            if row["domain"] == domain and not row["warmup"]:
                by_key.setdefault((row["case_id"], row["repetition"]), {})[row["method"]] = float(row["total_action_latency_ms"])
        deltas = [pair["policy2sample"] - pair["schema_only"]
                  for pair in by_key.values() if set(pair) == {"policy2sample", "schema_only"}]
        out[domain]["paired"] = {
            "n_pairs": len(deltas),
            "mean_absolute_delta_ms": statistics.mean(deltas) if deltas else None,
            "p50_absolute_delta_ms": percentile(deltas, 50) if deltas else None,
            "p95_absolute_delta_ms": percentile(deltas, 95) if deltas else None,
            "relative_overhead_pct_of_schema_mean": 100 * (p["mean_ms"] - b["mean_ms"]) / b["mean_ms"],
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--robot-cases", required=True)
    ap.add_argument("--iot-cases", required=True)
    ap.add_argument("--vocab-cache", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--repetitions", type=int, default=5)
    ap.add_argument("--warmups", type=int, default=1)
    ap.add_argument("--case-limit", type=int, default=0, help="pilot only: truncate each domain; 0 uses all 30")
    ap.add_argument("--tensor-split", default="0.45,0.55")
    args = ap.parse_args()

    robot_cases, iot_cases = load_cases(Path(args.robot_cases), Path(args.iot_cases))
    if args.case_limit:
        robot_cases, iot_cases = robot_cases[:args.case_limit], iot_cases[:args.case_limit]
    model_sha = file_sha256(Path(args.model))
    tensor_split = [float(x) for x in args.tensor_split.split(",")]
    llm = Llama(model_path=args.model, n_ctx=4096, n_batch=512, n_ubatch=128,
                n_gpu_layers=-1, split_mode=llama_cpp.LLAMA_SPLIT_MODE_LAYER, tensor_split=tensor_split,
                offload_kqv=True, flash_attn=True, logits_all=True, verbose=False)
    rows: list[dict[str, object]] = []
    try:
        vocab_started = time.perf_counter_ns()
        vocab_path = Path(args.vocab_cache)
        if vocab_path.exists():
            import pickle
            vocab = pickle.loads(vocab_path.read_bytes())
            vocab_cache_hit = True
        else:
            vocab = []
            for token_id in range(llm.n_vocab()):
                piece = llm.detokenize([token_id])
                vocab.append(piece if isinstance(piece, bytes) else str(piece).encode("utf-8"))
            vocab_path.parent.mkdir(parents=True, exist_ok=True)
            import pickle
            vocab_path.write_bytes(pickle.dumps(vocab, protocol=pickle.HIGHEST_PROTOCOL))
            vocab_cache_hit = False
        vocab_ms = (time.perf_counter_ns() - vocab_started) / 1e6
        grammars, compile_timings = compile_grammars(llm, vocab)

        robot_all = robot_universe()
        iot_all = iot_universe()
        tokenization_started = time.perf_counter_ns()
        robot_token_texts = [compact(robot_action_obj(a)) for a in robot_all]
        robot_token_texts.append(compact(robot_action_obj("STOP")))
        iot_token_texts = [compact(iot_action_obj(a)) for a in iot_all]
        ids_by_candidate: dict[str, tuple[int, ...]] = {}
        for text in robot_token_texts + iot_token_texts:
            ids_by_candidate[text] = ids_for_text(llm, text)
        tokenization_compile_ms = (time.perf_counter_ns() - tokenization_started) / 1e6
        universe_prefix_maps = {
            ("robot", "move"): prefix_map_for(ids_by_candidate, robot_token_texts[:-1]),
            ("robot", "stop"): prefix_map_for(ids_by_candidate, [robot_token_texts[-1]]),
            ("iot", "all"): prefix_map_for(ids_by_candidate, iot_token_texts),
        }

        # One warm-up per domain/method, excluded from all summaries.
        for domain, case in (("robot", robot_cases[0]), ("iot", iot_cases[0])):
            for method in ("schema_only", "policy2sample"):
                universe = robot_all if domain == "robot" else iot_all
                for _ in range(args.warmups):
                    row = run_one(llm, grammars, domain, case, method, universe, ids_by_candidate,
                                  universe_prefix_maps, warmup=True)
                    rows.append(row)

        run_index = 0
        for case_idx, (domain, case) in enumerate(
                [("robot", c) for c in robot_cases] + [("iot", c) for c in iot_cases]):
            universe = robot_all if domain == "robot" else iot_all
            for rep in range(args.repetitions):
                methods = ("schema_only", "policy2sample") if (case_idx + rep) % 2 == 0 else ("policy2sample", "schema_only")
                for method in methods:
                    row = run_one(llm, grammars, domain, case, method, universe, ids_by_candidate,
                                  universe_prefix_maps)
                    row["repetition"] = rep + 1
                    row["order_index"] = run_index
                    rows.append(row)
                    progress_path = Path(args.output + ".progress.jsonl")
                    with progress_path.open("a", encoding="utf-8") as progress_stream:
                        progress_stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                        progress_stream.flush()
                    run_index += 1
                    print(f"{domain} {row['case_id']} rep={rep+1} {method} "
                          f"{row['status']} {row['total_action_latency_ms']:.1f}ms "
                          f"tokens={row['output_tokens']}", flush=True)
    finally:
        llm.close()

    data = {
        "experiment": "rq4_full_action_latency_xgrammar_vs_policy2sample_qwen3vl_8b",
        "model": "Qwen3-VL-8B-Instruct-Q8_0",
        "model_sha256": model_sha,
        "decode": {"temperature": 0.0, "selection": "greedy argmax over allowed token IDs", "repetitions": args.repetitions,
                   "warmups": args.warmups, "paired_order": "alternated by case index and repetition"},
        "timing_boundary": "from request setup/specialization start through prompt prefill, constrained decode and JSON parse; model loading, vocabulary extraction, grammar compilation, and global action tokenization excluded",
        "methods": {"schema_only": "XGrammar structural mask intersected with a once-compiled canonical trie for the full declared action lattice (no safety filtering)", "policy2sample": "same XGrammar grammar intersected with a per-request safe-domain prefix mask via Policy2SampleDynamicMatcher"},
        "case_counts": {"robot": len(robot_cases), "iot": len(iot_cases)},
        "universes": {"robot_move": len(robot_all), "robot_stop": 1, "iot": len(iot_all)},
        "hardware": read_hardware(),
        "versions": {"xgrammar": getattr(xgr, "__version__", "unknown"), "llama_cpp": getattr(__import__("llama_cpp"), "__version__", "unknown")},
        "one_time_costs": {"vocab_cache_hit": vocab_cache_hit, "vocab_prepare_ms": vocab_ms,
                           "grammar_compile_ms": compile_timings, "global_action_tokenization_ms": tokenization_compile_ms},
        "summaries": summarize(rows),
        "rows": rows,
        "notes": [
            "This is a finite-lattice action-generation timing experiment, not continuous-valued token enforcement or simulator replay.",
            "Robot inputs are the frozen 30-case Core v2 metadata; IoT inputs are the frozen Hard-IoT Core-30 selection.",
            "The safety domain is recomputed from each case Context on every Policy2Sample request; action token paths are globally precomputed once.",
            "The model receives identical prompt text under both methods; the baseline is not told the safe candidate set.",
            "The full-domain trie only enforces canonical, whitespace-free serialization and declared finite field values; it does not remove unsafe assignments.",
        ],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), "summaries": data["summaries"], "one_time_costs": data["one_time_costs"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
