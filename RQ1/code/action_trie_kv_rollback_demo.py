"""Minimal token-trie primitives used by the RQ1 masked-action runner.

This module contains only the trie and incremental-logit operations required by
RQ1; unrelated development benchmarks and scenario prompts are not included.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from llama_cpp import Llama


MODEL_PATHS = {
    "2b": Path(os.environ.get("QWEN3_VL_2B_MODEL", "models/Qwen3VL-2B-Instruct-Q8_0.gguf")),
    "4b": Path(os.environ.get("QWEN3_VL_4B_MODEL", "models/Qwen3VL-4B-Instruct-Q8_0.gguf")),
    "8b": Path(os.environ.get("QWEN3_VL_8B_MODEL", "models/Qwen3VL-8B-Instruct-Q8_0.gguf")),
    "32b": Path(os.environ.get("QWEN3_VL_32B_MODEL", "models/Qwen3VL-32B-Instruct-Q8_0.gguf")),
}


@dataclass
class TrieNode:
    children: dict[int, "TrieNode"] = field(default_factory=dict)
    action: Optional[str] = None
    prefix: tuple[int, ...] = ()


def log_softmax_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = np.asarray(logits, dtype=np.float64) / temperature
    scaled -= np.max(scaled)
    return scaled - np.log(np.exp(scaled).sum())


def build_trie(llm: Llama, actions: list[str]) -> tuple[TrieNode, dict[str, tuple[int, ...]]]:
    root = TrieNode()
    action_tokens: dict[str, tuple[int, ...]] = {}
    for action in actions:
        tokens = tuple(llm.tokenize(f" {action}".encode("utf-8"), add_bos=False, special=True))
        action_tokens[action] = tokens
        node = root
        prefix: tuple[int, ...] = ()
        for token_id in tokens:
            prefix += (token_id,)
            node = node.children.setdefault(token_id, TrieNode(prefix=prefix))
        node.action = action
    return root, action_tokens


class ActionTrieDecoder:
    def __init__(self, llm, root, action_tokens, temperature, safety):
        self.llm = llm
        self.root = root
        self.action_tokens = action_tokens
        self.temperature = temperature
        self.safety = safety
        self.action_start_n_tokens = llm.n_tokens
        self.logprob_cache: dict[tuple[int, ...], np.ndarray] = {}

    def truncate_to_action_start(self) -> bool:
        removed = self.llm._ctx.kv_cache_seq_rm(-1, self.action_start_n_tokens, -1)
        self.llm.n_tokens = self.action_start_n_tokens
        return bool(removed)

    def node_logprobs(self, prefix: tuple[int, ...]) -> np.ndarray:
        cached = self.logprob_cache.get(prefix)
        if cached is not None:
            return cached
        self.truncate_to_action_start()
        if prefix:
            self.llm.eval(list(prefix))
        logits = self.llm._scores[self.llm.n_tokens - 1]
        result = log_softmax_temperature(logits, self.temperature)
        self.logprob_cache[prefix] = result
        return result

    def child_probabilities(self, node: TrieNode) -> tuple[np.ndarray, dict[int, float]]:
        logprobs = self.node_logprobs(node.prefix)
        child_ids = list(node.children)
        child_logprobs = np.asarray([logprobs[token_id] for token_id in child_ids], dtype=np.float64)
        child_logprobs -= np.max(child_logprobs)
        child_weights = np.exp(child_logprobs)
        child_weights /= child_weights.sum()
        return logprobs, {
            token_id: float(probability)
            for token_id, probability in zip(child_ids, child_weights)
        }
