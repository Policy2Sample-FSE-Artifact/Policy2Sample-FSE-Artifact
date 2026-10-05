"""Tokenizer-aware semantic mask primitives.

The adapter is intentionally independent of XGrammar and of a specific model
runtime.  It converts a finite canonical value domain into allowed next-token
IDs.  A production engine can intersect this set with the structural grammar
mask before sampling.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_FLOOR, ROUND_CEILING
from enum import Enum
from fractions import Fraction
from typing import Callable, Iterable, Sequence


TokenEncoder = Callable[[str], Sequence[int]]


@dataclass(frozen=True)
class TokenMask:
    allowed_token_ids: frozenset[int]
    terminal: bool = False
    rejected_candidates: int = 0

    def intersect(self, other: "TokenMask") -> "TokenMask":
        return TokenMask(
            self.allowed_token_ids & other.allowed_token_ids,
            terminal=self.terminal and other.terminal,
            rejected_candidates=self.rejected_candidates + other.rejected_candidates,
        )


class NumericMaskStatus(str, Enum):
    REPRESENTABLE = "REPRESENTABLE"
    UNREPRESENTABLE = "UNREPRESENTABLE"


@dataclass(frozen=True)
class NumericMaskResult:
    """Auditable result for one interval-to-token projection."""

    status: NumericMaskStatus
    mask: TokenMask
    representable_values: tuple[str, ...]


class NumericTokenAdapter:
    """Convert an exact numeric interval into tokenizer-compatible next tokens.

    The adapter enumerates the representable canonical decimal lattice, then
    applies tokenizer-prefix filtering.  Runtime interval changes therefore
    do not require recompiling the structural grammar, while the fixed-
    precision boundary remains explicit.
    """

    def __init__(self, encode: TokenEncoder, precision: int = 2) -> None:
        if precision < 0:
            raise ValueError("precision must be non-negative")
        self.trie = TokenTrieAdapter(encode)
        self.precision = precision

    def next_tokens(
        self,
        current_text: str,
        current_token_ids: Sequence[int],
        lower: float,
        upper: float,
        *,
        suffix_builder: Callable[[str], str] | None = None,
        lower_closed: bool = True,
        upper_closed: bool = True,
    ) -> NumericMaskResult:
        domain = CanonicalDecimalDomain(
            Decimal(str(lower)),
            Decimal(str(upper)),
            self.precision,
            lower_closed,
            upper_closed,
        )
        values = domain.values()
        builder = suffix_builder or (lambda value: value)
        mask = self.trie.next_tokens(
            current_text,
            current_token_ids,
            (builder(value) for value in values),
        )
        status = NumericMaskStatus.REPRESENTABLE if mask.allowed_token_ids else NumericMaskStatus.UNREPRESENTABLE
        return NumericMaskResult(status, mask, values)


@dataclass(frozen=True)
class CanonicalDecimalDomain:
    lower: Decimal | Fraction
    upper: Decimal | Fraction
    precision: int = 2
    lower_closed: bool = True
    upper_closed: bool = True

    def values(self) -> tuple[str, ...]:
        lower = self.lower if isinstance(self.lower, Fraction) else Fraction(self.lower)
        upper = self.upper if isinstance(self.upper, Fraction) else Fraction(self.upper)
        scale = 10 ** self.precision
        lo_scaled, hi_scaled = lower * scale, upper * scale
        first = -((-lo_scaled.numerator) // lo_scaled.denominator)
        last = hi_scaled.numerator // hi_scaled.denominator
        if not self.lower_closed and Fraction(first, scale) <= lower:
            first += 1
        if not self.upper_closed and Fraction(last, scale) >= upper:
            last -= 1
        if first > last:
            return ()
        return tuple(f"{Decimal(index) / Decimal(scale):.{self.precision}f}" for index in range(first, last + 1))


class TokenTrieAdapter:
    """Build next-token masks from complete candidate continuations.

    ``encode`` must use the same tokenizer and special-token settings as the
    actual decoder.  Candidates that are not token-prefix-compatible with the
    current generated token sequence are rejected rather than approximated.
    """

    def __init__(self, encode: TokenEncoder) -> None:
        self.encode = encode

    def next_tokens(
        self,
        current_text: str,
        current_token_ids: Sequence[int],
        candidate_suffixes: Iterable[str],
    ) -> TokenMask:
        allowed: set[int] = set()
        rejected = 0
        current = tuple(current_token_ids)
        for suffix in candidate_suffixes:
            encoded = tuple(self.encode(current_text + suffix))
            if encoded[: len(current)] != current:
                rejected += 1
                continue
            if len(encoded) == len(current):
                continue
            allowed.add(encoded[len(current)])
        return TokenMask(frozenset(allowed), rejected_candidates=rejected)

    def numeric_next_tokens(
        self,
        current_text: str,
        current_token_ids: Sequence[int],
        lower: float,
        upper: float,
        precision: int = 2,
        suffix_builder: Callable[[str], str] | None = None,
        *,
        lower_closed: bool = True,
        upper_closed: bool = True,
    ) -> TokenMask:
        domain = CanonicalDecimalDomain(
            Decimal(str(lower)),
            Decimal(str(upper)),
            precision,
            lower_closed,
            upper_closed,
        )
        builder = suffix_builder or (lambda value: value)
        return self.next_tokens(current_text, current_token_ids, (builder(value) for value in domain.values()))


class SimpleCharacterTokenizer:
    """Tiny deterministic tokenizer used for adapter tests and local demos."""

    def __call__(self, text: str) -> tuple[int, ...]:
        return tuple(ord(char) for char in text)
