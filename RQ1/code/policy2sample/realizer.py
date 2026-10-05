"""DSL-to-sample realization for the Policy2Sample vertical slice."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from semantic_ir import ContextSnapshot, IntervalUnion, OnlineExactProjectionBackend, ProjectionResult, ProjectionStatus, RealInterval

from .compiler import (
    CompilationResult,
    CompilationStatus,
    direction_task_contract,
    positive_distance_task_contract,
    speed_cap_contract,
)


@dataclass(frozen=True)
class FixedPrecisionTokenAdapter:
    """String-level numeric adapter used before a tokenizer-backed adapter.

    The adapter intentionally rejects extra precision.  It is sound for the
    canonical decimal grammar used by the first experiment, but it does not
    claim to implement arbitrary tokenizer prefix masking yet.
    """

    precision: int = 2

    def canonical(self, value: float) -> str:
        return f"{value:.{self.precision}f}"

    def admits(self, value_text: str, lower: float, upper: float) -> bool:
        try:
            value = float(value_text)
        except ValueError:
            return False
        if "." in value_text:
            decimals = len(value_text.split(".", 1)[1])
            if decimals > self.precision:
                return False
        return lower - 1e-12 <= value <= upper + 1e-12


@dataclass(frozen=True)
class RealizationResult:
    status: str
    action: dict[str, object] | None
    action_text: str | None
    domains: tuple[dict[str, object], ...]
    reason: str | None = None


class SamplingRealizer:
    """Generate a complete structured move sample from executable DSL."""

    def __init__(self, compilation: CompilationResult, backend: OnlineExactProjectionBackend) -> None:
        if compilation.status != CompilationStatus.EXECUTABLE or compilation.sketch is None:
            raise ValueError("realizer requires an executable compilation")
        self.compilation = compilation
        self.sketch = compilation.sketch
        self.backend = backend
        self.adapter = FixedPrecisionTokenAdapter()

    @classmethod
    def from_compilation(cls, compilation: CompilationResult, backend_factory) -> "SamplingRealizer":
        if compilation.sketch is None:
            raise ValueError("compilation has no sketch")
        backend = backend_factory(compilation.sketch)
        return cls(compilation, backend)

    def _choose_enum(self, values: Sequence[object]) -> object:
        if self.sketch.requested_direction in values:
            return self.sketch.requested_direction
        return values[0]

    def _choose_real(self, domain: RealInterval | IntervalUnion, field: str) -> float:
        intervals = domain.intervals if isinstance(domain, IntervalUnion) else (domain,)
        for interval in intervals:
            lower, upper = interval.lower, interval.upper
            if field == "distance":
                target = min(0.5, upper)
            elif field == "speed":
                target = min(0.2, upper)
            else:
                target = (lower + upper) / 2.0
            candidate = round(max(lower, target), 2)
            if interval.contains(candidate):
                return candidate
            midpoint = round((lower + upper) / 2.0, 2)
            if interval.contains(midpoint):
                return midpoint
        raise ValueError(f"domain for {field} has no representable two-decimal value")

    def _result_domain(self, field: str, result: ProjectionResult) -> dict[str, object]:
        row: dict[str, object] = {"field": field, "status": result.status.value, "reason": result.reason}
        if result.domain is not None:
            row["domain"] = result.domain.as_decoder_constraint()
        return row

    def _serialize_action(self, action: dict[str, object]) -> str:
        serialized: dict[str, object] = {}
        for field, value in action.items():
            if field in {"distance", "speed"} and isinstance(value, (int, float)):
                serialized[field] = self.adapter.canonical(float(value))
            else:
                serialized[field] = value
        return json.dumps(serialized, ensure_ascii=False, separators=(",", ":"))

    def realize(
        self,
        snapshot: ContextSnapshot,
        *,
        current_context_version: Callable[[], str] | None = None,
    ) -> RealizationResult:
        prefix: list[object] = []
        action: dict[str, object] = {}
        domains: list[dict[str, object]] = []

        for field in self.backend.schema.fields:
            result = self.backend.project(snapshot, prefix)
            domains.append(self._result_domain(field.name, result))
            if result.status != ProjectionStatus.EXACT or result.domain is None:
                return RealizationResult(result.status.value, None, None, tuple(domains), result.reason)

            if field.kind == "enum":
                value = self._choose_enum(result.domain.values)  # type: ignore[union-attr]
            else:
                try:
                    value = self._choose_real(result.domain, field.name)  # type: ignore[arg-type]
                except ValueError as exc:
                    return RealizationResult("UNREPRESENTABLE", None, None, tuple(domains), str(exc))
                if not result.domain.contains(value):  # type: ignore[union-attr]
                    return RealizationResult("UNSUPPORTED", None, None, tuple(domains), f"adapter selected value outside {field.name} domain")
            action[field.name] = value
            prefix.append(value)

        # The action is admitted against one immutable snapshot.  Re-check the
        # snapshot lease and provider version immediately before returning it
        # to the caller for commit; stale actions are discarded, never reused.
        if not snapshot.is_fresh():
            return RealizationResult("STALE", None, None, tuple(domains), "Context snapshot expired before action commit")
        if current_context_version is not None and current_context_version() != snapshot.version:
            return RealizationResult("STALE", None, None, tuple(domains), "Context version changed before action commit")

        # Defensive assertion for implementation verification.  This does not
        # repair, retry, or admit an action rejected by the sampling-time
        # projection path; enforcement has already happened above.
        if not self.backend.validate_action(snapshot, action):
            return RealizationResult("UNSUPPORTED", None, None, tuple(domains), "final independent validation failed")
        return RealizationResult("EXACT", action, self._serialize_action(action), tuple(domains))
