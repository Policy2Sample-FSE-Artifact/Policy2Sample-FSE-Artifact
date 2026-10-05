"""XGrammar integration for the Policy2Sample dynamic matcher.

The old ``fill_intersection_mask`` helper remains for compatibility with the
first checkpoint.  New code should use ``dynamic_matcher``: it binds a
Scene/RuleSet/Context and computes one final mask inside the matcher loop,
without recompiling the static grammar when Context values change.
"""

from __future__ import annotations

from typing import Any

from .token_adapter import TokenMask
from .dynamic_matcher import Policy2SampleDynamicMatcher
from .xgrammar_layer import XGrammarStructuralLayer, XGrammarUnavailable


class XGrammarSemanticBridge:
    def __init__(self, tokenizer_info: Any, vocab_size: int) -> None:
        self.vocab_size = int(vocab_size)
        self.structural = XGrammarStructuralLayer(tokenizer_info)

    def compile_dsl(self, dsl_document: dict[str, object]) -> Any:
        return self.structural.compile_dsl(dsl_document)

    def matcher(self, compiled_grammar: Any) -> Any:
        return self.structural.matcher(compiled_grammar)

    def dynamic_matcher(self, compiled_grammar: Any, dynamic_provider, *, token_decoder=None) -> Policy2SampleDynamicMatcher:
        """Create a Context-aware matcher over one compiled structural grammar."""
        return Policy2SampleDynamicMatcher(
            self.structural.matcher(compiled_grammar),
            self.vocab_size,
            dynamic_provider,
            token_decoder=token_decoder,
        )

    def fill_intersection_mask(self, matcher: Any, semantic_mask: TokenMask, index: int = 0) -> tuple[Any, frozenset[int]]:
        """Return ``XGrammar structural ∩ Policy2Sample semantic``.

        The returned tensor is ready for ``xgrammar.apply_token_bitmask_inplace``.
        ``allowed`` is returned as an auditable CPU-side view for tests/logging.
        """
        try:
            import xgrammar as xgr  # type: ignore
        except ImportError as exc:  # pragma: no cover - exercised by environment setup
            raise XGrammarUnavailable("install xgrammar to enable mask intersection") from exc

        bitmask = xgr.allocate_token_bitmask(1, self.vocab_size)
        matcher.fill_next_token_bitmask(bitmask, index)
        structural_ids = self._ids_from_bitmask(bitmask, index)
        allowed = frozenset(structural_ids & set(semantic_mask.allowed_token_ids))
        self._write_ids_to_bitmask(bitmask, allowed, index)
        return bitmask, allowed

    @staticmethod
    def _ids_from_bitmask(bitmask: Any, index: int = 0) -> set[int]:
        ids: set[int] = set()
        row = bitmask[index]
        for word_index in range(row.shape[0]):
            word = int(row[word_index].item()) & 0xFFFFFFFF
            for bit in range(32):
                token_id = word_index * 32 + bit
                if token_id < bitmask.shape[1] * 32 and word & (1 << bit):
                    ids.add(token_id)
        return ids

    @staticmethod
    def _write_ids_to_bitmask(bitmask: Any, token_ids: frozenset[int], index: int = 0) -> None:
        row = bitmask[index]
        row.zero_()
        words = [0] * row.shape[0]
        for token_id in token_ids:
            if token_id < 0 or token_id >= row.shape[0] * 32:
                continue
            words[token_id // 32] |= 1 << (token_id % 32)
        for word_index, word in enumerate(words):
            # XGrammar stores 32-bit words in signed int32 tensors.
            signed = word - (1 << 32) if word >= (1 << 31) else word
            row[word_index] = signed
