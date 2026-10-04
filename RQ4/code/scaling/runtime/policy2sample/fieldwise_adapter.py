"""Field-scoped lexical masking for exact numeric domains.

Create one session after the Realizer has projected a field.  Reuse that
session for every token transition inside the field; discard it only after the
field delimiter is accepted and the next field has been projected.  The
candidate language ends at the field delimiter, so a token that also enters a
subsequent field is conservatively rejected until that field's domain exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from typing import Callable, Iterable, Sequence, Any

from .token_adapter import CanonicalDecimalDomain, NumericMaskStatus, TokenMask


Encode = Callable[[str], Sequence[int]]
Decode = Callable[[Sequence[int]], str]


def _ascii_digits(text: str) -> bool:
    return bool(text) and all("0" <= char <= "9" for char in text)


def _intervals(domain: Any):
    return getattr(domain, "intervals", (domain,))


@dataclass(frozen=True)
class FieldMaskStatus:
    status: NumericMaskStatus
    candidate_sequences: int
    rejected_prefixes: int


@dataclass(frozen=True)
class FieldCursor:
    """Parser state for the next safety-relevant numeric field.

    ``start_token_ids`` is the emitted action prefix at the start of this
    field; ``target_prefix_text`` may additionally contain the not-yet-emitted
    canonical key/separator. This lets a single tokenizer token span the
    structural-to-value boundary without bypassing the value domain.
    """

    field_key: tuple[object, ...]
    start_token_ids: tuple[int, ...]
    target_prefix_text: str
    precision: int
    delimiter: str = ","


class FieldwiseDomainMaskProvider:
    """Compute exact projection once per field; reuse a lexical session per token.

    ``locate`` must identify the upcoming constrained field as soon as its
    preceding value prefix is complete, including before any structural token
    that could be merged with the first numeric token. ``project`` is invoked
    only when the context/field/completed-prefix cache key changes.
    """

    def __init__(self, *, locate, project, encode: Encode, decode: Decode, vocab_size: int) -> None:
        self.locate = locate
        self.project = project
        self.encode = encode
        self.decode = decode
        self.vocab_size = int(vocab_size)
        self._cache_key: tuple[object, ...] | None = None
        self._session: NumericFieldMaskSession | None = None
        self.projection_calls = 0

    def mask_for_candidates(self, binding: Any, prefix_text: str, prefix_token_ids: tuple[int, ...], candidates):
        cursor = self.locate(binding, prefix_text, prefix_token_ids)
        if cursor is None:
            # No safety field can be inferred: fail closed, do not silently
            # fall back to the structural grammar alone.
            return TokenMask(frozenset())
        key = (
            binding.scene_id,
            binding.rule_set_id,
            binding.context_version,
            *cursor.field_key,
        )
        if key != self._cache_key:
            result = self.project(binding, cursor)
            self.projection_calls += 1
            status = getattr(result, "status", None)
            domain = getattr(result, "domain", None)
            if getattr(status, "value", status) != "EXACT" or domain is None:
                self._cache_key, self._session = key, None
                return TokenMask(frozenset())
            try:
                self._session = NumericFieldMaskSession(
                    encode=self.encode,
                    decode=self.decode,
                    field_prefix_text=cursor.target_prefix_text,
                    field_prefix_token_ids=cursor.start_token_ids,
                    domain=domain,
                    precision=cursor.precision,
                    delimiter=cursor.delimiter,
                )
            except (ValueError, OverflowError):
                self._cache_key, self._session = key, None
                return TokenMask(frozenset())
            self._cache_key = key
        if self._session is None:
            return TokenMask(frozenset())
        mask, _ = self._session.mask(prefix_token_ids, candidate_token_ids=tuple(candidates))
        return mask

    def invalidate(self) -> None:
        self._cache_key = None
        self._session = None


def _ceil_fraction(value: Fraction) -> int:
    return -((-value.numerator) // value.denominator)


def _scaled_ranges(intervals: Sequence[Any], precision: int) -> tuple[tuple[int, int], ...]:
    scale = 10 ** precision
    ranges = []
    for interval in intervals:
        lower = interval.exact_lower if interval.exact_lower is not None else Fraction(str(interval.lower))
        upper = interval.exact_upper if interval.exact_upper is not None else Fraction(str(interval.upper))
        scaled_lower, scaled_upper = lower * scale, upper * scale
        lo, hi = _ceil_fraction(scaled_lower), scaled_upper.numerator // scaled_upper.denominator
        if not interval.lower_closed and Fraction(lo, scale) == lower:
            lo += 1
        if not interval.upper_closed and Fraction(hi, scale) == upper:
            hi -= 1
        if lo <= hi:
            ranges.append((lo, hi))
    ranges.sort()
    merged: list[list[int]] = []
    for lo, hi in ranges:
        if not merged or lo > merged[-1][1] + 1:
            merged.append([lo, hi])
        else:
            merged[-1][1] = max(merged[-1][1], hi)
    return tuple((lo, hi) for lo, hi in merged)


def _prefix_abs_ranges(prefix: str, precision: int, max_integer_digits: int) -> tuple[tuple[int, int], ...]:
    scale = 10 ** precision
    if "." in prefix:
        if precision == 0 or prefix.count(".") != 1:
            return ()
        integer, fraction = prefix.split(".", 1)
        if not _ascii_digits(integer) or (len(integer) > 1 and integer.startswith("0")):
            return ()
        if not _ascii_digits(fraction) and fraction != "":
            return ()
        if len(fraction) > precision:
            return ()
        frac_lo = int(fraction) * (10 ** (precision - len(fraction))) if fraction else 0
        frac_hi = frac_lo + (10 ** (precision - len(fraction))) - 1
        whole = int(integer)
        return ((whole * scale + frac_lo, whole * scale + frac_hi),)

    if prefix and not _ascii_digits(prefix):
        return ()
    if len(prefix) > max_integer_digits or (len(prefix) > 1 and prefix.startswith("0")):
        return ()
    output = []
    first_len = max(1, len(prefix))
    for digits in range(first_len, max_integer_digits + 1):
        if prefix:
            multiplier = 10 ** (digits - len(prefix))
            whole_lo = int(prefix) * multiplier
            whole_hi = whole_lo + multiplier - 1
            if digits > 1 and whole_lo < 10 ** (digits - 1):
                continue
        elif digits == 1:
            whole_lo, whole_hi = 0, 9
        else:
            whole_lo, whole_hi = 10 ** (digits - 1), 10 ** digits - 1
        output.append((whole_lo * scale, whole_hi * scale + scale - 1))
    return tuple(output)


def _has_intersection(left: Sequence[tuple[int, int]], right: Sequence[tuple[int, int]]) -> bool:
    return any(max(a, c) <= min(b, d) for a, b in left for c, d in right)


def _canonical_scaled_integer(text: str, precision: int) -> int | None:
    if precision == 0:
        if not _ascii_digits(text) and not (text.startswith("-") and _ascii_digits(text[1:])):
            return None
        return int(text)
    if text.count(".") != 1:
        return None
    negative = text.startswith("-")
    body = text[1:] if negative else text
    whole, fractional = body.split(".", 1)
    if not _ascii_digits(whole) or len(fractional) != precision or not _ascii_digits(fractional):
        return None
    if len(whole) > 1 and whole.startswith("0"):
        return None
    scaled = int(whole) * (10 ** precision) + int(fractional)
    if negative:
        if scaled == 0:
            return None
        scaled = -scaled
    return scaled


class NumericFieldMaskSession:
    """A tokenizer-faithful, cached candidate trie for a single numeric field."""

    def __init__(
        self,
        *,
        encode: Encode,
        decode: Decode | None = None,
        field_prefix_text: str,
        field_prefix_token_ids: Sequence[int],
        domain: Any,
        precision: int,
        delimiter: str = ",",
        max_values: int = 100_000,
    ) -> None:
        self.encode = encode
        self.decode = decode
        self.field_prefix_text = field_prefix_text
        self.prefix_text = field_prefix_text
        self.prefix_ids = tuple(int(token) for token in field_prefix_token_ids)
        self.precision = precision
        self.delimiter = delimiter
        intervals = _intervals(domain)
        self._scaled = _scaled_ranges(intervals, precision)
        largest_magnitude = max((max(abs(lo), abs(hi)) for lo, hi in self._scaled), default=0)
        largest_whole = largest_magnitude // (10 ** precision)
        self._max_integer_digits = max(1, len(str(largest_whole)))
        if self.decode is None:
            values: list[str] = []
            for interval in intervals:
                numeric = CanonicalDecimalDomain(
                    interval.exact_lower if interval.exact_lower is not None else Fraction(str(interval.lower)),
                    interval.exact_upper if interval.exact_upper is not None else Fraction(str(interval.upper)),
                    precision,
                    interval.lower_closed,
                    interval.upper_closed,
                ).values()
                values.extend(numeric)
                if len(values) > max_values:
                    raise ValueError(f"numeric field representation exceeds {max_values} canonical values")
            values = sorted(set(values), key=Decimal)
            self.values = tuple(values)
            self._lexical_targets = tuple(field_prefix_text + value + delimiter for value in self.values)
            self._continuations = tuple(tuple(int(token) for token in encode(target)) for target in self._lexical_targets)
        else:
            self.values = ()
            self._lexical_targets = ()
            self._continuations = ()

    @property
    def representable(self) -> bool:
        return bool(self._scaled) if self.decode is not None else bool(self._continuations)

    def _text_is_safe_prefix(self, text: str) -> bool:
        if not self.representable:
            return False
        if self.field_prefix_text.startswith(text):
            return True
        if not text.startswith(self.field_prefix_text):
            return False
        fragment = text[len(self.field_prefix_text):]
        if self.delimiter in fragment:
            numeric, suffix = fragment.split(self.delimiter, 1)
            if suffix:
                return False
            scaled = _canonical_scaled_integer(numeric, self.precision)
            return scaled is not None and any(lo <= scaled <= hi for lo, hi in self._scaled)
        if not fragment:
            return self.representable
        if fragment == "-":
            return any(lo < 0 for lo, _ in self._scaled)
        if fragment.startswith("-"):
            abs_ranges = tuple((-hi, -lo) for lo, hi in self._scaled if lo < 0)
            possible = _prefix_abs_ranges(fragment[1:], self.precision, self._max_integer_digits)
            return _has_intersection(possible, abs_ranges)
        positive_ranges = tuple((max(0, lo), hi) for lo, hi in self._scaled if hi >= 0)
        possible = _prefix_abs_ranges(fragment, self.precision, self._max_integer_digits)
        return _has_intersection(possible, positive_ranges)

    def mask(
        self,
        current_token_ids: Sequence[int],
        *,
        candidate_token_ids: Sequence[int] | None = None,
    ) -> tuple[TokenMask, FieldMaskStatus]:
        current = tuple(int(token) for token in current_token_ids)
        if current[: len(self.prefix_ids)] != self.prefix_ids:
            return TokenMask(frozenset()), FieldMaskStatus(NumericMaskStatus.UNREPRESENTABLE, 0, 0)
        if not self.representable:
            return TokenMask(frozenset()), FieldMaskStatus(NumericMaskStatus.UNREPRESENTABLE, 0, 0)
        if self.decode is not None and candidate_token_ids is not None:
            decoded_prefix = self.decode(current)
            allowed = {
                int(token_id)
                for token_id in candidate_token_ids
                if self.decode(current + (int(token_id),)) != decoded_prefix
                and self._text_is_safe_prefix(self.decode(current + (int(token_id),)))
            }
            status = NumericMaskStatus.REPRESENTABLE if allowed else NumericMaskStatus.UNREPRESENTABLE
            return (
                TokenMask(frozenset(allowed), rejected_candidates=len(candidate_token_ids) - len(allowed)),
                FieldMaskStatus(status, len(self._scaled), len(candidate_token_ids) - len(allowed)),
            )
        allowed: set[int] = set()
        rejected = 0
        for candidate in self._continuations:
            if candidate[: len(current)] != current:
                rejected += 1
                continue
            if len(candidate) == len(current):
                continue
            allowed.add(candidate[len(current)])
        status = NumericMaskStatus.REPRESENTABLE if allowed else NumericMaskStatus.UNREPRESENTABLE
        return (
            TokenMask(frozenset(allowed), rejected_candidates=rejected),
            FieldMaskStatus(status, len(self._continuations), rejected),
        )


def field_value_strings(
    domain: Any, precision: int, *, max_values: int = 100_000
) -> tuple[str, ...]:
    """Independent lexical-oracle helper; intentionally returns strings only."""
    intervals: Iterable[Any] = _intervals(domain)
    output: set[str] = set()
    for interval in intervals:
        output.update(CanonicalDecimalDomain(
            interval.exact_lower if interval.exact_lower is not None else Fraction(str(interval.lower)),
            interval.exact_upper if interval.exact_upper is not None else Fraction(str(interval.upper)),
            precision,
            interval.lower_closed,
            interval.upper_closed,
        ).values())
        if len(output) > max_values:
            raise ValueError(f"numeric field representation exceeds {max_values} canonical values")
    return tuple(sorted(output, key=Decimal))
