"""Differential lexical-mask test against the tokenizer embedded in a GGUF.

Run in the model server's llama-cpp-python environment.  The model is loaded
with vocab_only=True; no neural weights or GPU inference are used.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import statistics
import sys
import time
import types
from fractions import Fraction
from pathlib import Path


def load_adapter(root: Path):
    pkg = types.ModuleType("p2s_adapter_runtime")
    pkg.__path__ = [str(root / "policy2sample")]
    sys.modules[pkg.__name__] = pkg
    for short, filename in (("token_adapter", "token_adapter.py"), ("fieldwise_adapter", "fieldwise_adapter.py")):
        name = pkg.__name__ + "." + short
        spec = importlib.util.spec_from_file_location(name, root / "policy2sample" / filename)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[pkg.__name__ + ".fieldwise_adapter"]


def make_domain(module, intervals):
    pieces = tuple(types.SimpleNamespace(
        lower=float(lo), upper=float(hi), lower_closed=lc, upper_closed=uc,
        exact_lower=Fraction(str(lo)), exact_upper=Fraction(str(hi)),
    ) for lo, hi, lc, uc in intervals)
    if len(pieces) == 1:
        return pieces[0]
    return types.SimpleNamespace(intervals=pieces)


def percentiles(values):
    ordered = sorted(values)
    def at(p):
        return ordered[min(len(ordered) - 1, int(round((len(ordered) - 1) * p)))] if ordered else 0
    return {"p50_us": at(.50) * 1e6, "p95_us": at(.95) * 1e6, "p99_us": at(.99) * 1e6}


def run(args):
    adapter = load_adapter(args.source_root.resolve())
    if args.tokenizer_json:
        from tokenizers import Tokenizer

        hf_tokenizer = Tokenizer.from_file(str(args.tokenizer_json))

        class TokenizerJsonAdapter:
            def __init__(self):
                self.special_token_ids = {
                    token_id for token_id, token in hf_tokenizer.get_added_tokens_decoder().items()
                    if token.special
                }

            def tokenize(self, text, add_bos=False, special=False):
                return hf_tokenizer.encode(text.decode("utf-8"), add_special_tokens=add_bos).ids

            def detokenize(self, ids, special=False):
                return hf_tokenizer.decode(list(ids), skip_special_tokens=not special).encode("utf-8")

            def n_vocab(self):
                return hf_tokenizer.get_vocab_size(with_added_tokens=True)

            def close(self):
                return None

        llm = TokenizerJsonAdapter()
        tokenizer_source = args.tokenizer_source or "Hugging Face tokenizer.json"
    else:
        from llama_cpp import Llama
        if not args.model_path:
            raise ValueError("Provide --model-path or --tokenizer-json")
        llm = Llama(model_path=str(args.model_path), vocab_only=True, n_ctx=128, verbose=False)
        tokenizer_source = "tokenizer embedded in GGUF; llama.cpp vocab-only loader"
    try:
        encode = lambda text: tuple(llm.tokenize(text.encode("utf-8"), add_bos=False, special=False))
        def decode(ids):
            raw = llm.detokenize(list(ids), special=False)
            return raw.decode("utf-8", errors="replace")
        vocab_size = int(llm.n_vocab())
        special = set(getattr(llm, "special_token_ids", ()))
        for name in ("token_bos", "token_eos", "token_eot", "token_nl", "token_fim_prefix", "token_fim_middle", "token_fim_suffix"):
            fn = getattr(llm, name, None)
            if callable(fn):
                try:
                    special.add(int(fn()))
                except Exception:
                    pass
        candidates = [token_id for token_id in range(vocab_size) if token_id not in special]
        rng = random.Random(args.seed)
        specs = [
            ("closed_endpoints", 2, ((0.20, 0.25, True, True),)),
            ("open_endpoints", 2, ((0.20, 0.25, False, False),)),
            ("negative_values", 2, ((-0.30, -0.10, True, True),)),
            ("nonconvex_union", 1, ((0.0, 0.2, True, True), (0.8, 1.0, True, True))),
            ("structural_to_value", 2, ((-0.1, 0.1, True, True),)),
            ("delimiter_boundary", 2, ((0.20, 0.21, True, True),)),
            ("value_delimiter_singleton", 2, ((-0.2, -0.2, True, True),)),
            ("unrepresentable", 2, ((0.001, 0.002, True, True),)),
            ("precision_three", 3, ((0.101, 0.104, True, False),)),
        ]
        case_sessions = []
        for index, (name, precision, ranges) in enumerate(specs):
            prefix = '{"field_' + str(index) + '": '
            domain = make_domain(adapter, ranges)
            session = adapter.NumericFieldMaskSession(
                encode=encode, decode=decode, field_prefix_text=prefix,
                field_prefix_token_ids=(), domain=domain, precision=precision,
                delimiter=",",
            )
            value_strings = adapter.field_value_strings(domain, precision)
            targets = tuple(prefix + value + "," for value in value_strings)
            # Use genuine token-prefix states from the model tokenizer, not
            # character-level approximations. Empty prefix includes JSON-key
            # tokens and therefore covers structural-to-value transitions.
            states = [()]
            for target in targets:
                tokenized = encode(target)
                states.extend(tokenized[:pos] for pos in range(1, len(tokenized)))
            case_sessions.append((name, precision, session, targets, states))

        transitions = []
        mask_batch_timings = []
        mask_batch_sizes = []
        for name, precision, session, targets, states in case_sessions:
            if not session.representable:
                transitions.append({"case": name, "kind": "UNREPRESENTABLE", "checked": 1, "false_admission": 0, "false_rejection": 0})
                continue
            for _ in range(args.transitions_per_case):
                state = rng.choice(states)
                base_text = decode(state)
                # Stratify: include known-valid continuations plus random
                # vocabulary tokens, including multi-character pieces.
                valid_next = {
                    int(encoded[len(state)]) for target in targets
                    if (encoded := encode(target))[:len(state)] == state and len(encoded) > len(state)
                }
                sample = set(rng.sample(candidates, min(args.random_candidates, len(candidates))))
                sample.update(valid_next)
                sampled = tuple(sample)
                mask_started = time.perf_counter_ns()
                adapter_mask, _ = session.mask(state, candidate_token_ids=sampled)
                mask_batch_timings.append((time.perf_counter_ns() - mask_started) / 1e9)
                mask_batch_sizes.append(len(sampled))
                allowed = adapter_mask.allowed_token_ids
                for token_id in sampled:
                    decoded = decode(state + (token_id,))
                    oracle = decoded != base_text and any(target.startswith(decoded) for target in targets)
                    transitions.append({
                        "case": name,
                        "precision": precision,
                        "prefix_token_count": len(state),
                        "state_token_ids": list(state),
                        "token_id": token_id,
                        "decoded_token_text": decode((token_id,)),
                        "adapter_allow": token_id in allowed,
                        "oracle_allow": oracle,
                        "latency_s": 0.0,
                    })
        # Exercise a real token boundary around the comma between two numeric
        # safety fields. If a model token crosses the boundary, the first
        # field session must reject it while the successor domain is unknown.
        cross_prefix = '{"first": '
        cross_text = cross_prefix + "0.20,\"second\": 0.40}"
        cross_domain = make_domain(adapter, ((0.20, 0.20, True, True),))
        cross_session = adapter.NumericFieldMaskSession(
            encode=encode, decode=decode, field_prefix_text=cross_prefix,
            field_prefix_token_ids=(), domain=cross_domain, precision=2, delimiter=",")
        cross_ids = encode(cross_text)
        crossing = []
        comma_pos = len(cross_prefix) + len("0.20,") - 1
        for pos, token_id in enumerate(cross_ids):
            state = cross_ids[:pos]
            decoded_after = decode(state + (token_id,))
            if len(decoded_after) > comma_pos + 1 and decoded_after.startswith(cross_prefix) and "," in decoded_after:
                comma_at = decoded_after.find(",", len(cross_prefix))
                if comma_at < len(decoded_after) - 1:
                    mask, _ = cross_session.mask(state, candidate_token_ids=(token_id,))
                    crossing.append({
                        "token_id": int(token_id),
                        "decoded_token_text": decode((token_id,)),
                        "adapter_allow": token_id in mask.allowed_token_ids,
                        "oracle_allow": False,
                        "rejected_as_cross_field": token_id not in mask.allowed_token_ids,
                    })
        if not crossing:
            # Some tokenizers put a hard token boundary exactly after the
            # comma. At that point the old session must still reject the next
            # field's structural token; the caller must first compute D_(i+1).
            for pos in range(len(cross_ids)):
                state = cross_ids[:pos]
                decoded_state = decode(state)
                if decoded_state == cross_prefix + "0.20,":
                    token_id = cross_ids[pos]
                    mask, _ = cross_session.mask(state, candidate_token_ids=(token_id,))
                    crossing.append({
                        "token_id": int(token_id),
                        "decoded_token_text": decode((token_id,)),
                        "adapter_allow": token_id in mask.allowed_token_ids,
                        "oracle_allow": False,
                        "rejected_as_cross_field": token_id not in mask.allowed_token_ids,
                    })
                    break
        transitions.extend({
            "case": "cross_next_safety_field",
            "precision": 2,
            "prefix_token_count": None,
            **item,
            "latency_s": 0.0,
        } for item in crossing)
        # Replay the adapter path separately for timing; correctness rows above
        # retain every tested transition and are the authoritative record.
        timings = []
        for row in transitions:
            if "token_id" not in row or "state_token_ids" not in row:
                continue
            session = next(item[2] for item in case_sessions if item[0] == row["case"])
            # Reconstruct a valid prefix by picking a target whose tokenization
            # has this length and whose decoded prefix matches the recorded one.
            states = next(item[4] for item in case_sessions if item[0] == row["case"])
            state = tuple(row["state_token_ids"])
            start = time.perf_counter_ns()
            session.mask(state, candidate_token_ids=(row["token_id"],))
            timings.append((time.perf_counter_ns() - start) / 1e9)
            row["latency_s"] = timings[-1]
        false_admissions = [r for r in transitions if r.get("adapter_allow") and not r.get("oracle_allow")]
        false_rejections = [r for r in transitions if not r.get("adapter_allow") and r.get("oracle_allow")]
        checked = sum("token_id" in row for row in transitions)
        cross_rows = [r for r in transitions if r.get("case") == "cross_next_safety_field"]
        result = {
            "experiment": "real_tokenizer_fieldwise_differential",
            "model_artifact": (args.model_path.name if args.model_path else args.tokenizer_json.name),
            "tokenizer_source": tokenizer_source,
            "vocab_size": vocab_size,
            "seed": args.seed,
            "requested_random_transitions_per_case": args.transitions_per_case,
            "random_vocabulary_candidates_per_transition": args.random_candidates,
            "coverage_cases": [name for name, *_ in specs],
            "checked_transitions": checked,
            "unrepresentable_domain_cases": [r["case"] for r in transitions if r.get("kind") == "UNREPRESENTABLE"],
            "cross_next_field_transitions_tested": len(cross_rows),
            "cross_next_field_tokens_rejected": sum(r["rejected_as_cross_field"] for r in cross_rows),
            "false_admission_count": len(false_admissions),
            "false_rejection_count": len(false_rejections),
            "exact_agreement": (checked - len(false_admissions) - len(false_rejections)) / checked if checked else None,
            "adapter_transition_latency": percentiles(timings),
            "adapter_full_mask_latency": percentiles(mask_batch_timings),
            "mean_candidate_ids_per_full_mask": statistics.mean(mask_batch_sizes) if mask_batch_sizes else 0,
            "failures": {"false_admissions": false_admissions[:100], "false_rejections": false_rejections[:100]},
            "transitions": transitions,
        }
        return result
    finally:
        llm.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--tokenizer-json", type=Path)
    parser.add_argument("--tokenizer-source")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--transitions-per-case", type=int, default=100)
    parser.add_argument("--random-candidates", type=int, default=40)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.model_path) == bool(args.tokenizer_json):
        parser.error("provide exactly one of --model-path or --tokenizer-json")
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in {"transitions", "failures"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
