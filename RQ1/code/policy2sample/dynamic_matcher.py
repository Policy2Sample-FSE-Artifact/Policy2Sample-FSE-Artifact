"""Parametric, context-aware matcher for Policy2Sample.

The original bridge exposed ``structural mask ∩ semantic mask`` as a helper
that callers had to orchestrate themselves.  This module makes that
intersection a matcher operation with an explicit runtime binding:

    compile once -> bind(scene, rule_set, context) -> decode -> switch scene

The structural matcher can be an XGrammar ``GrammarMatcher``.  The dynamic
rule provider is called inside ``fill_next_token_mask`` and therefore changes
the mask without recompiling the static grammar.  A later C++ implementation
can move the same transition hook into a forked XGrammar matcher without
changing this runtime contract.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

try:
    from .token_adapter import TokenMask
except ImportError:  # pragma: no cover - allows the remote runner to load this module directly
    from token_adapter import TokenMask  # type: ignore


@dataclass(frozen=True)
class RuntimeRuleBinding:
    """Rules and Context currently bound to one action generation."""

    scene_id: str
    rule_set_id: str
    context_version: str
    context: Mapping[str, object]
    captured_at: float | None = None
    max_age_ms: float | None = None

    def is_fresh(self, now: float | None = None) -> bool:
        if self.captured_at is None or self.max_age_ms is None:
            return True
        current = time.time() if now is None else float(now)
        return (current - self.captured_at) * 1000.0 <= self.max_age_ms


@dataclass(frozen=True)
class DynamicMaskDecision:
    """Auditable result of one unified structural/dynamic mask step."""

    structural_count: int
    dynamic_count: int
    allowed_count: int
    context_version: str
    scene_id: str
    rule_set_id: str


class DynamicMaskProvider(Protocol):
    def __call__(
        self,
        binding: RuntimeRuleBinding,
        prefix_text: str,
        prefix_token_ids: tuple[int, ...],
    ) -> TokenMask:
        """Return tokens with at least one safe completion under the binding."""


class DynamicMaskEmpty(RuntimeError):
    """No token remains after applying the current dynamic rule binding."""


class StaleRuntimeBinding(RuntimeError):
    """A caller attempted to use a matcher with an obsolete Context version."""


class Policy2SampleDynamicMatcher:
    """One matcher API for static structure and dynamic safety rules.

    ``structural_matcher`` is intentionally duck-typed so local tests do not
    need XGrammar installed.  In production it is an XGrammar
    ``GrammarMatcher``.  The caller receives one final bitmask and never has
    to perform a second, external semantic-mask intersection.
    """

    def __init__(
        self,
        structural_matcher: Any,
        vocab_size: int,
        dynamic_provider: DynamicMaskProvider,
        *,
        token_decoder: Callable[[int], str] | None = None,
        bitmask_factory: Callable[[int, int], Any] | None = None,
    ) -> None:
        self.structural_matcher = structural_matcher
        self.vocab_size = int(vocab_size)
        self.dynamic_provider = dynamic_provider
        self.token_decoder = token_decoder or (lambda token_id: chr(token_id))
        self._bitmask_factory = bitmask_factory
        self.binding: RuntimeRuleBinding | None = None
        self.prefix_token_ids: tuple[int, ...] = ()
        self.prefix_text = ""
        self._last_allowed: frozenset[int] = frozenset()

    def bind(
        self,
        *,
        scene_id: str,
        rule_set_id: str,
        context_version: str,
        context: Mapping[str, object],
        expected_context_version: str | None = None,
        captured_at: float | None = None,
        max_age_ms: float | None = None,
    ) -> None:
        """Bind a new scene/rule/context without recompiling the grammar.

        A bind starts a new action prefix.  This is the intended operation
        when action A finishes and action B starts.  The optional expected
        version prevents a stale Context from silently driving decoding.
        """
        if expected_context_version is not None and context_version != expected_context_version:
            raise StaleRuntimeBinding(
                f"Context version {context_version!r} does not match expected {expected_context_version!r}"
            )
        self.binding = RuntimeRuleBinding(
            scene_id,
            rule_set_id,
            context_version,
            dict(context),
            captured_at=captured_at,
            max_age_ms=max_age_ms,
        )
        self.prefix_token_ids = ()
        self.prefix_text = ""
        self._last_allowed = frozenset()
        reset = getattr(self.structural_matcher, "reset", None)
        if callable(reset):
            reset()

    def switch_scene(
        self,
        scene_id: str,
        rule_set_id: str,
        context_version: str,
        context: Mapping[str, object],
        *,
        captured_at: float | None = None,
        max_age_ms: float | None = None,
    ) -> None:
        """Explicit A -> B transition; semantically identical to ``bind``."""
        self.bind(
            scene_id=scene_id,
            rule_set_id=rule_set_id,
            context_version=context_version,
            context=context,
            captured_at=captured_at,
            max_age_ms=max_age_ms,
        )

    def _ensure_bound(self) -> RuntimeRuleBinding:
        if self.binding is None:
            raise RuntimeError("dynamic matcher has no Scene/RuleSet/Context binding")
        if not self.binding.is_fresh():
            raise StaleRuntimeBinding(
                f"Context version {self.binding.context_version!r} is stale for the current decode"
            )
        return self.binding

    def _allocate_bitmask(self) -> Any:
        if self._bitmask_factory is not None:
            return self._bitmask_factory(1, self.vocab_size)
        try:
            import xgrammar as xgr  # type: ignore
        except ImportError as exc:  # pragma: no cover - remote-only path
            raise RuntimeError("install xgrammar or provide bitmask_factory") from exc
        return xgr.allocate_token_bitmask(1, self.vocab_size)

    @staticmethod
    def _ids_from_bitmask(bitmask: Any, index: int = 0) -> set[int]:
        ids: set[int] = set()
        row = bitmask[index]
        for word_index in range(row.shape[0]):
            word = int(row[word_index].item()) & 0xFFFFFFFF
            for bit in range(32):
                token_id = word_index * 32 + bit
                if word & (1 << bit):
                    ids.add(token_id)
        return ids

    @staticmethod
    def _write_ids_to_bitmask(bitmask: Any, token_ids: Sequence[int], index: int = 0) -> None:
        row = bitmask[index]
        row.zero_()
        for token_id in token_ids:
            if token_id < 0 or token_id >= row.shape[0] * 32:
                continue
            word_index, bit = divmod(int(token_id), 32)
            word = int(row[word_index].item()) & 0xFFFFFFFF
            word |= 1 << bit
            row[word_index] = word - (1 << 32) if word >= (1 << 31) else word

    def fill_next_token_mask(self, index: int = 0) -> tuple[Any, DynamicMaskDecision]:
        """Compute one final mask from grammar state and dynamic rules."""
        binding = self._ensure_bound()
        bitmask = self._allocate_bitmask()
        self.structural_matcher.fill_next_token_bitmask(bitmask, index)
        structural_ids = self._ids_from_bitmask(bitmask, index)
        candidate_masker = getattr(self.dynamic_provider, "mask_for_candidates", None)
        if callable(candidate_masker):
            # Field-aware providers receive only structurally possible tokens.
            # Their exact domain is projected once at field entry and cached;
            # subsequent token steps do lexical-prefix checks only.
            dynamic_mask = candidate_masker(
                binding, self.prefix_text, self.prefix_token_ids, structural_ids
            )
        else:
            dynamic_mask = self.dynamic_provider(binding, self.prefix_text, self.prefix_token_ids)
        allowed = frozenset(structural_ids & set(dynamic_mask.allowed_token_ids))
        if not allowed:
            self._last_allowed = frozenset()
            raise DynamicMaskEmpty(
                f"empty dynamic mask for scene={binding.scene_id!r}, rule_set={binding.rule_set_id!r}, "
                f"context_version={binding.context_version!r}, prefix={self.prefix_text!r}"
            )
        self._write_ids_to_bitmask(bitmask, allowed, index)
        self._last_allowed = allowed
        return bitmask, DynamicMaskDecision(
            structural_count=len(structural_ids),
            dynamic_count=len(dynamic_mask.allowed_token_ids),
            allowed_count=len(allowed),
            context_version=binding.context_version,
            scene_id=binding.scene_id,
            rule_set_id=binding.rule_set_id,
        )

    def accept_token(self, token_id: int) -> None:
        """Advance both the structural and dynamic matcher state."""
        self._ensure_bound()
        if self._last_allowed and token_id not in self._last_allowed:
            raise ValueError(f"token {token_id} was not allowed by the last dynamic mask")
        if not self.structural_matcher.accept_token(token_id):
            raise ValueError(f"structural matcher rejected token {token_id}")
        self.prefix_token_ids = self.prefix_token_ids + (int(token_id),)
        self.prefix_text += self.token_decoder(int(token_id))
        self._last_allowed = frozenset()
