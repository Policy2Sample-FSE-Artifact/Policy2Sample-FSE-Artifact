"""Minimal projection-oriented Safety Contract IR vertical slice.

This module intentionally implements only the first supported fragment:

* typed action fields: finite Enum and bounded Real;
* Context-only guards;
* context-dependent and action-dependent linear constraints;
* exact per-field projection through a persistent Z3-style model;
* a small decoder-domain adapter with explicit EXACT/EMPTY/UNSUPPORTED status.

The IR is kept separate from the projection backend.  It is not a new full
DSL yet; the Python constructors are an executable semantic fixture for the
first end-to-end correctness experiment.
"""

from __future__ import annotations

import math
import re
import sys
import time
from dataclasses import dataclass, field as dataclass_field
from enum import Enum
from fractions import Fraction
from itertools import product
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

sys.path.insert(0, str(__import__("pathlib").Path(__file__).with_name("vendor")))
import z3  # type: ignore  # noqa: E402


Context = Mapping[str, object]
ActionExprs = Mapping[str, Any]
BoundContext = Mapping[str, object]
_MISSING_DEFAULT = object()


class ProjectionStatus(str, Enum):
    EXACT = "EXACT"
    EMPTY = "EMPTY"
    INVALID = "INVALID"
    UNSUPPORTED = "UNSUPPORTED"
    STALE = "STALE"


@dataclass(frozen=True)
class FieldSpec:
    name: str
    kind: str
    values: tuple[object, ...] = ()
    lower: float | None = None
    upper: float | None = None
    unit: str | None = None
    optional: bool = False
    default: object = _MISSING_DEFAULT
    semantic_role: str | None = None

    @classmethod
    def enum(
        cls,
        name: str,
        values: Sequence[object],
        *,
        optional: bool = False,
        default: object = _MISSING_DEFAULT,
        semantic_role: str | None = None,
    ) -> "FieldSpec":
        normalized = tuple(values)
        if not normalized or len(set(normalized)) != len(normalized):
            raise ValueError(f"enum field {name!r} must have unique non-empty values")
        return cls(name=name, kind="enum", values=normalized, optional=optional, default=default, semantic_role=semantic_role)

    @classmethod
    def real(
        cls,
        name: str,
        lower: float,
        upper: float,
        unit: str | None = None,
        *,
        optional: bool = False,
        default: object = _MISSING_DEFAULT,
        semantic_role: str | None = None,
    ) -> "FieldSpec":
        if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
            raise ValueError(f"invalid bounds for real field {name!r}")
        return cls(
            name=name,
            kind="real",
            lower=float(lower),
            upper=float(upper),
            unit=unit,
            optional=optional,
            default=default,
            semantic_role=semantic_role,
        )


@dataclass(frozen=True)
class ActionSchema:
    fields: tuple[FieldSpec, ...]
    name: str | None = None
    scope_aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        names = [field.name for field in self.fields]
        if len(names) != len(set(names)):
            raise ValueError("action field names must be unique")
        for field in self.fields:
            if field.optional and field.default is _MISSING_DEFAULT:
                raise ValueError(f"optional field {field.name!r} must declare a schema default")

    @property
    def order(self) -> tuple[str, ...]:
        return tuple(field.name for field in self.fields)

    def field(self, name: str) -> FieldSpec:
        for field in self.fields:
            if field.name == name:
                return field
        raise KeyError(name)

    def canonicalize(self, action: Mapping[str, object]) -> dict[str, object]:
        """Materialize schema defaults into the safety-relevant action."""
        unknown = set(action) - set(self.order)
        if unknown:
            raise ValueError(f"action contains unknown fields: {sorted(unknown)}")
        normalized = dict(action)
        for field in self.fields:
            if field.name not in normalized:
                if not field.optional:
                    raise ValueError(f"missing required action field {field.name!r}")
                normalized[field.name] = field.default
        return normalized


@dataclass(frozen=True)
class ContextFieldSpec:
    name: str
    kind: str
    unit: str | None = None
    semantic_role: str | None = None


@dataclass(frozen=True)
class SemanticRole:
    name: str
    side: str
    kind: str
    unit: str | None = None


@dataclass(frozen=True)
class BoundRole:
    role: SemanticRole
    concrete_name: str


@dataclass(frozen=True)
class BindingEnvironment:
    action: tuple[BoundRole, ...]
    context: tuple[BoundRole, ...]
    lookups: tuple["LookupRef", ...] = ()

    def _lookup(self, side: str, role_name: str) -> BoundRole:
        entries = self.action if side == "action" else self.context
        for entry in entries:
            if entry.role.name == role_name:
                return entry
        raise KeyError(f"unbound {side} semantic role {role_name!r}")

    def action_field(self, role_name: str) -> str:
        return self._lookup("action", role_name).concrete_name

    def context_field(self, role_name: str) -> str:
        return self._lookup("context", role_name).concrete_name

    def lookup(self, name: str) -> "LookupRef":
        for lookup in self.lookups:
            if lookup.name == name:
                return lookup
        raise KeyError(f"unregistered LookupRef {name!r}")

    def validate(
        self,
        schema: ActionSchema,
        context_schema: Mapping[str, ContextFieldSpec],
        contracts: Sequence["SafetyContract"],
    ) -> None:
        action_fields = {field.name: field for field in schema.fields}
        for entry in self.action:
            field = action_fields.get(entry.concrete_name)
            if field is None:
                raise ValueError(f"binding points to unknown action field {entry.concrete_name!r}")
            if field.kind != entry.role.kind:
                raise ValueError(f"type mismatch for action role {entry.role.name!r}")
            if field.unit != entry.role.unit:
                raise ValueError(f"unit mismatch for action role {entry.role.name!r}")
            if field.semantic_role is not None and field.semantic_role != entry.role.name:
                raise ValueError(
                    f"semantic-role mismatch for action binding {entry.role.name!r}: "
                    f"catalog declares {field.semantic_role!r}"
                )
        for entry in self.context:
            field = context_schema.get(entry.concrete_name)
            if field is None:
                raise ValueError(f"binding points to unknown Context field {entry.concrete_name!r}")
            if field.kind != entry.role.kind:
                raise ValueError(f"type mismatch for Context role {entry.role.name!r}")
            if field.unit != entry.role.unit:
                raise ValueError(f"unit mismatch for Context role {entry.role.name!r}")
            if field.semantic_role is not None and field.semantic_role != entry.role.name:
                raise ValueError(
                    f"semantic-role mismatch for Context binding {entry.role.name!r}: "
                    f"catalog declares {field.semantic_role!r}"
                )
        for contract in contracts:
            if not _scope_matches_schema(contract.scope, schema):
                raise ValueError(
                    f"action-scope mismatch: contract scope {contract.scope!r} "
                    f"does not match schema {schema.name!r}"
                )
            for role in contract.action_roles:
                self._lookup("action", role)
            for role in contract.context_roles:
                self._lookup("context", role)
        lookup_names = [lookup.name for lookup in self.lookups]
        if len(lookup_names) != len(set(lookup_names)):
            raise ValueError("LookupRef names must be unique")
        for lookup in self.lookups:
            if lookup.output_kind not in {"real", "enum", "bool"}:
                raise ValueError(f"unsupported LookupRef output kind {lookup.output_kind!r}")


def _scope_matches_schema(scope: str, schema: ActionSchema) -> bool:
    """Conservatively check that a contract scope can target this schema."""
    if scope == "*":
        return True
    candidates = {name.lower() for name in (schema.name, *schema.scope_aliases) if name}
    if not candidates:
        # Legacy schemas without a declared scope cannot be checked here.
        return True
    normalized = scope.lower()
    return any(normalized == candidate or normalized.endswith("." + candidate) for candidate in candidates)


LookupResolver = Callable[[ActionExprs, BoundContext], Any]


@dataclass(frozen=True)
class LookupRef:
    """A deterministic deployment lookup evaluated against one snapshot.

    The resolver receives the symbolic action expressions and an immutable
    copy of the current snapshot values.  It must not read a live sensor,
    network client, or mutable global state.  This makes the snapshot binding
    explicit while still allowing enum-dependent lookups such as
    ``clearance(direction)``.
    """

    name: str
    signature: str
    output_kind: str
    output_unit: str | None
    resolver: LookupResolver

    def evaluate(self, action: ActionExprs, snapshot_values: BoundContext) -> Any:
        return self.resolver(action, snapshot_values)


@dataclass(frozen=True)
class BoolConst:
    """Explicit DSL-level boolean constant for top/empty contracts."""

    value: bool

    def to_z3(self) -> Any:
        return z3.BoolVal(self.value)


def true_constraint() -> BoolConst:
    return BoolConst(True)


def false_constraint() -> BoolConst:
    return BoolConst(False)


Guard = Callable[[BoundContext, BindingEnvironment], bool]
ConstraintBuilder = Callable[[ActionExprs, BoundContext, BindingEnvironment], Any]


@dataclass(frozen=True)
class SafetyContract:
    name: str
    scope: str
    action_roles: tuple[str, ...]
    context_roles: tuple[str, ...]
    guard: Guard
    constraint: ConstraintBuilder
    fragment: str = "bounded_linear"

    def active(self, context: BoundContext, binding: BindingEnvironment) -> bool:
        return bool(self.guard(context, binding))


@dataclass(frozen=True)
class RealInterval:
    lower: float
    upper: float
    lower_closed: bool = True
    upper_closed: bool = True
    exact_lower: Fraction | None = dataclass_field(default=None, compare=False, repr=False)
    exact_upper: Fraction | None = dataclass_field(default=None, compare=False, repr=False)

    def contains(self, value: object, tolerance: float = 0.0) -> bool:
        try:
            numeric = Fraction(str(value))
        except (TypeError, ValueError):
            return False
        lower = self.exact_lower if self.exact_lower is not None else Fraction(str(self.lower))
        upper = self.exact_upper if self.exact_upper is not None else Fraction(str(self.upper))
        tol = Fraction(str(tolerance))
        lower_ok = numeric >= lower - tol if self.lower_closed else numeric > lower + tol
        upper_ok = numeric <= upper + tol if self.upper_closed else numeric < upper - tol
        return lower_ok and upper_ok

    def as_decoder_constraint(self) -> dict[str, object]:
        result: dict[str, object] = {"type": "number", "minimum": self.lower, "maximum": self.upper}
        exact_lower = self.exact_lower if self.exact_lower is not None else Fraction(str(self.lower))
        exact_upper = self.exact_upper if self.exact_upper is not None else Fraction(str(self.upper))
        # JSON Schema numeric literals are binary-decoded by many consumers;
        # preserve the exact rational endpoint alongside the interoperable
        # numeric approximation for Policy2Sample-aware adapters.
        result["x-policy2sample-exactMinimum"] = f"{exact_lower.numerator}/{exact_lower.denominator}"
        result["x-policy2sample-exactMaximum"] = f"{exact_upper.numerator}/{exact_upper.denominator}"
        if not self.lower_closed:
            result["exclusiveMinimum"] = True
        if not self.upper_closed:
            result["exclusiveMaximum"] = True
        return result


@dataclass(frozen=True)
class EnumDomain:
    values: tuple[object, ...]

    def contains(self, value: object) -> bool:
        return value in self.values

    def as_decoder_constraint(self) -> dict[str, object]:
        return {"enum": list(self.values)}


@dataclass(frozen=True)
class IntervalUnion:
    """Exact finite union of closed/open rational intervals."""

    intervals: tuple[RealInterval, ...]

    def contains(self, value: object) -> bool:
        return any(interval.contains(value) for interval in self.intervals)

    def as_decoder_constraint(self) -> dict[str, object]:
        return {
            "type": "number",
            "anyOf": [interval.as_decoder_constraint() for interval in self.intervals],
        }


@dataclass(frozen=True)
class ProjectionResult:
    status: ProjectionStatus
    domain: RealInterval | IntervalUnion | EnumDomain | None = None
    reason: str | None = None


@dataclass(frozen=True)
class ContextSnapshot:
    values: Mapping[str, object]
    version: str
    captured_at: float | None = None
    max_age_ms: float | None = None

    def __post_init__(self) -> None:
        # Freeze the provider result at the admission boundary.  A caller can
        # mutate its original dictionary later without changing this snapshot.
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))

    def is_fresh(self, now: float | None = None) -> bool:
        if self.captured_at is None or self.max_age_ms is None:
            return True
        current = time.time() if now is None else float(now)
        return (current - self.captured_at) * 1000.0 <= self.max_age_ms

    def age_ms(self, now: float | None = None) -> float | None:
        if self.captured_at is None:
            return None
        current = time.time() if now is None else float(now)
        return max(0.0, (current - self.captured_at) * 1000.0)


def _z3_real(value: object) -> Any:
    return z3.RealVal(str(value))


def _as_fraction(value: Any) -> Fraction:
    if z3.is_rational_value(value):
        return Fraction(value.numerator_as_long(), value.denominator_as_long())
    if z3.is_int_value(value):
        return Fraction(value.as_long())
    text = str(value).rstrip("?")
    # Optimize uses infinitesimals for unattained strict extrema.  Strip only
    # that infinitesimal; endpoint openness is established independently by SAT.
    text = re.sub(r"\s*[+-]\s*-?(?:\d+\s*\*\s*)?epsilon\s*$", "", text)
    if text.endswith("oo"):
        raise ValueError(f"unbounded optimization result: {value}")
    return Fraction(text.strip())


def _as_float(value: Any) -> float:
    """Compatibility helper for older experiment scripts."""
    return float(_as_fraction(value))


class OnlineExactProjectionBackend:
    """Exact projection backend for the bounded linear vertical slice."""

    capabilities = frozenset(
        {
            "enum",
            "bounded_real",
            "linear",
            "context_guard",
            "bounded_linear",
            "bounded_linear_with_enum_branch",
            "guarded_bounded_linear",
            "bounded_component_l1",
            "boolean_constant",
        }
    )

    def __init__(
        self,
        schema: ActionSchema,
        binding: BindingEnvironment,
        contracts: Sequence[SafetyContract],
        context_schema: Mapping[str, ContextFieldSpec],
        *,
        require_fresh_context: bool = False,
        clock: Callable[[], float] = time.time,
        branch_budget: int = 256,
    ) -> None:
        if branch_budget < 1:
            raise ValueError("branch_budget must be positive")
        binding.validate(schema, context_schema, contracts)
        self.schema = schema
        self.binding = binding
        self.contracts = tuple(contracts)
        self.context_schema = dict(context_schema)
        self.require_fresh_context = require_fresh_context
        self.clock = clock
        self.branch_budget = branch_budget
        self.variables = {
            field.name: z3.Int(field.name) if field.kind == "enum" else z3.Real(field.name)
            for field in schema.fields
        }
        self.enum_codes = {
            field.name: {value: index for index, value in enumerate(field.values)}
            for field in schema.fields
            if field.kind == "enum"
        }
        self._base_constraints = self._make_base_constraints()
        self._capability_checker = ProjectionCapabilityChecker(self.capabilities)
        self._declared_capability_report = self._capability_checker.report(self.contracts)

    def _context_status(self, snapshot: ContextSnapshot) -> ProjectionResult | None:
        if self.require_fresh_context and not snapshot.is_fresh(self.clock()):
            return ProjectionResult(
                ProjectionStatus.STALE,
                reason=f"Context snapshot {snapshot.version!r} exceeded its freshness bound",
            )
        for name, spec in self.context_schema.items():
            if name not in snapshot.values:
                return ProjectionResult(ProjectionStatus.INVALID, reason=f"missing Context field {name}")
            value = snapshot.values[name]
            if spec.kind == "real":
                try:
                    if not math.isfinite(float(value)):
                        return ProjectionResult(ProjectionStatus.INVALID, reason=f"non-finite Context field {name}")
                except (TypeError, ValueError):
                    return ProjectionResult(ProjectionStatus.INVALID, reason=f"invalid Context field {name}")
            elif spec.kind == "bool" and not isinstance(value, bool):
                return ProjectionResult(ProjectionStatus.INVALID, reason=f"invalid Boolean Context field {name}")
        return None

    def _active_contracts(self, action_scope: str | None = None) -> tuple[SafetyContract, ...]:
        if action_scope is None:
            return self.contracts
        wanted = action_scope.lower()
        return tuple(
            contract
            for contract in self.contracts
            if contract.scope == "*" or contract.scope.lower() == wanted or contract.scope.lower().endswith("." + wanted)
        )

    def _runtime_capability_report(self, snapshot: ContextSnapshot) -> CapabilityReport:
        context = self._bound_context(snapshot)
        active = self._active_instances(snapshot, context)
        return self._capability_checker.report(active)

    def _active_instances(
        self,
        snapshot: ContextSnapshot,
        context: BoundContext | None = None,
    ) -> tuple[SafetyContract, ...]:
        bound_context = context if context is not None else self._bound_context(snapshot)
        return tuple(
            contract
            for contract in self._active_contracts(self.schema.name)
            if contract.active(bound_context, self.binding)
        )

    def _make_base_constraints(self) -> list[Any]:
        constraints = []
        for field in self.schema.fields:
            variable = self.variables[field.name]
            if field.kind == "enum":
                constraints.append(z3.Or(*(variable == index for index in range(len(field.values)))))
            else:
                constraints.extend((variable >= _z3_real(field.lower), variable <= _z3_real(field.upper)))
        return constraints

    def _bound_context(self, snapshot: ContextSnapshot) -> dict[str, object]:
        return dict(snapshot.values)

    def _formula(self, snapshot: ContextSnapshot) -> Any:
        context = self._bound_context(snapshot)
        action = self.variables
        formulas = []
        active_contracts = self._active_instances(snapshot, context)
        for contract in active_contracts:
            formula = contract.constraint(action, context, self.binding)
            if isinstance(formula, BoolConst):
                formula = formula.to_z3()
            elif isinstance(formula, bool):
                formula = z3.BoolVal(formula)
            if not z3.is_bool(formula):
                raise ValueError(f"contract {contract.name!r} did not produce a Boolean formula")
            formulas.append(formula)
        self._capability_checker.check(active_contracts)
        return z3.And(*formulas) if formulas else z3.BoolVal(True)

    def _prefix_constraints(self, prefix: Sequence[object]) -> list[Any]:
        if len(prefix) > len(self.schema.fields):
            raise ValueError("prefix is longer than action schema")
        result = []
        for field, value in zip(self.schema.fields, prefix):
            variable = self.variables[field.name]
            if field.kind == "enum":
                if value not in self.enum_codes[field.name]:
                    return [z3.BoolVal(False)]
                result.append(variable == self.enum_codes[field.name][value])
            else:
                result.append(variable == _z3_real(value))
        return result

    def _endpoint_attained(self, constraints: Sequence[Any], field: FieldSpec, value: Fraction) -> bool:
        solver = z3.Solver()
        solver.add(constraints)
        solver.add(self.variables[field.name] == _z3_real(value))
        return solver.check() == z3.sat

    def _dnf(self, formula: Any) -> list[list[Any]]:
        """Expand a Boolean formula to bounded DNF without dropping branches."""
        if z3.is_true(formula):
            return [[]]
        if z3.is_false(formula):
            return []
        if z3.is_implies(formula):
            antecedent, consequent = formula.children()
            return self._dnf(z3.Or(z3.Not(antecedent), consequent))
        if z3.is_not(formula):
            child = formula.arg(0)
            if z3.is_and(child):
                return self._dnf(z3.Or(*(z3.Not(item) for item in child.children())))
            if z3.is_or(child):
                return self._dnf(z3.And(*(z3.Not(item) for item in child.children())))
            if z3.is_implies(child):
                antecedent, consequent = child.children()
                return self._dnf(z3.And(antecedent, z3.Not(consequent)))
        if z3.is_or(formula):
            clauses: list[list[Any]] = []
            for child in formula.children():
                clauses.extend(self._dnf(child))
                if len(clauses) > self.branch_budget:
                    raise OverflowError("finite-branch expansion exceeded configured budget")
            return clauses
        if z3.is_and(formula):
            clauses = [[]]
            for child in formula.children():
                child_clauses = self._dnf(child)
                if not child_clauses:
                    return []
                if len(clauses) * len(child_clauses) > self.branch_budget:
                    raise OverflowError("finite-branch expansion exceeded configured budget")
                clauses = [left + right for left, right in product(clauses, child_clauses)]
            return clauses
        return [[formula]]

    @staticmethod
    def _normalize_intervals(intervals: Sequence[RealInterval]) -> tuple[RealInterval, ...]:
        def lower_of(item: RealInterval) -> Fraction:
            return item.exact_lower if item.exact_lower is not None else Fraction(str(item.lower))

        def upper_of(item: RealInterval) -> Fraction:
            return item.exact_upper if item.exact_upper is not None else Fraction(str(item.upper))

        ordered = sorted(intervals, key=lambda item: (lower_of(item), not item.lower_closed))
        merged: list[RealInterval] = []
        for nxt in ordered:
            if not merged:
                merged.append(nxt)
                continue
            cur = merged[-1]
            cur_upper, next_lower = upper_of(cur), lower_of(nxt)
            joins = next_lower < cur_upper or (
                next_lower == cur_upper and (cur.upper_closed or nxt.lower_closed)
            )
            if not joins:
                merged.append(nxt)
                continue
            next_upper = upper_of(nxt)
            if next_upper > cur_upper:
                merged[-1] = RealInterval(
                    cur.lower, nxt.upper, cur.lower_closed, nxt.upper_closed,
                    exact_lower=lower_of(cur), exact_upper=next_upper,
                )
            elif next_upper == cur_upper:
                merged[-1] = RealInterval(
                    cur.lower, cur.upper, cur.lower_closed,
                    cur.upper_closed or nxt.upper_closed,
                    exact_lower=lower_of(cur), exact_upper=cur_upper,
                )
        return tuple(merged)

    def _project_real_union(
        self, constraints: Sequence[Any], field: FieldSpec, prefix_length: int
    ) -> ProjectionResult:
        # Existentially eliminate future finite fields by enumerating their
        # declared values, then project each exact Boolean branch separately.
        remaining_enums = [
            (position, item) for position, item in enumerate(self.schema.fields)
            if position >= prefix_length and item.kind == "enum"
        ]
        branch_count = 1
        for _, enum_field in remaining_enums:
            branch_count *= len(enum_field.values)
            if branch_count > self.branch_budget:
                return ProjectionResult(
                    ProjectionStatus.UNSUPPORTED,
                    reason=f"finite-branch expansion exceeded budget {self.branch_budget}",
                )
        enum_assignments = product(*(range(len(item.values)) for _, item in remaining_enums)) if remaining_enums else [()]
        intervals: list[RealInterval] = []
        total_branches = 0
        try:
            for assignment in enum_assignments:
                substitutions = [
                    (self.variables[enum_field.name], z3.IntVal(code))
                    for (_, enum_field), code in zip(remaining_enums, assignment)
                ]
                branch_formula = z3.simplify(z3.substitute(z3.And(*constraints), *substitutions))
                clauses = self._dnf(branch_formula)
                total_branches += len(clauses)
                if total_branches > self.branch_budget:
                    raise OverflowError("finite-branch expansion exceeded configured budget")
                for clause in clauses:
                    branch = list(clause)
                    sat = z3.Solver()
                    sat.add(branch)
                    if sat.check() != z3.sat:
                        continue
                    lo = z3.Optimize()
                    lo.add(branch)
                    lo_handle = lo.minimize(self.variables[field.name])
                    if lo.check() != z3.sat:
                        continue
                    lower = _as_fraction(lo.lower(lo_handle))
                    hi = z3.Optimize()
                    hi.add(branch)
                    hi_handle = hi.maximize(self.variables[field.name])
                    if hi.check() != z3.sat:
                        continue
                    upper = _as_fraction(hi.upper(hi_handle))
                    intervals.append(RealInterval(
                        float(lower), float(upper),
                        self._endpoint_attained(branch, field, lower),
                        self._endpoint_attained(branch, field, upper),
                        exact_lower=lower, exact_upper=upper,
                    ))
        except OverflowError as exc:
            return ProjectionResult(ProjectionStatus.UNSUPPORTED, reason=str(exc))
        normalized = self._normalize_intervals(intervals)
        if not normalized:
            return ProjectionResult(ProjectionStatus.EMPTY, reason="no feasible completion")
        if len(normalized) == 1:
            return ProjectionResult(ProjectionStatus.EXACT, normalized[0])
        return ProjectionResult(ProjectionStatus.EXACT, IntervalUnion(normalized))

    def project(self, snapshot: ContextSnapshot, prefix: Sequence[object]) -> ProjectionResult:
        context_status = self._context_status(snapshot)
        if context_status is not None:
            return context_status
        capability = self._runtime_capability_report(snapshot)
        if not capability.supported:
            return ProjectionResult(ProjectionStatus.UNSUPPORTED, reason=capability.reason)
        index = len(prefix)
        if index >= len(self.schema.fields):
            if index > len(self.schema.fields):
                return ProjectionResult(ProjectionStatus.INVALID, reason="prefix is longer than action schema")
            return ProjectionResult(ProjectionStatus.UNSUPPORTED, reason="complete prefix")
        for field, value in zip(self.schema.fields, prefix):
            if field.kind == "enum":
                if value not in self.enum_codes[field.name]:
                    return ProjectionResult(ProjectionStatus.INVALID, reason=f"invalid enum value for {field.name}")
            else:
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    return ProjectionResult(ProjectionStatus.INVALID, reason=f"invalid numeric value for {field.name}")
                if not math.isfinite(numeric):
                    return ProjectionResult(ProjectionStatus.INVALID, reason=f"non-finite numeric value for {field.name}")
        field = self.schema.fields[index]
        constraints = self._base_constraints + [self._formula(snapshot)] + self._prefix_constraints(prefix)
        feasibility = z3.Solver()
        feasibility.add(constraints)
        if feasibility.check() != z3.sat:
            return ProjectionResult(ProjectionStatus.EMPTY, reason="no feasible completion")
        if field.kind == "enum":
            feasible_values = []
            for value, code in self.enum_codes[field.name].items():
                solver = z3.Solver()
                solver.add(constraints)
                solver.add(self.variables[field.name] == code)
                if solver.check() == z3.sat:
                    feasible_values.append(value)
            if not feasible_values:
                return ProjectionResult(ProjectionStatus.EMPTY, reason="no feasible enum value")
            return ProjectionResult(ProjectionStatus.EXACT, EnumDomain(tuple(feasible_values)))
        return self._project_real_union(constraints, field, len(prefix))

    def validate_action(self, snapshot: ContextSnapshot, action: Mapping[str, object]) -> bool:
        """Validate one complete schema-shaped action against the exact joint model."""
        if self._context_status(snapshot) is not None:
            return False
        if not self._runtime_capability_report(snapshot).supported:
            return False
        try:
            action = self.schema.canonicalize(action)
        except ValueError:
            return False
        constraints = self._base_constraints + [self._formula(snapshot)]
        for field in self.schema.fields:
            value = action[field.name]
            if field.kind == "enum":
                if value not in self.enum_codes[field.name]:
                    return False
                constraints.append(self.variables[field.name] == self.enum_codes[field.name][value])
            else:
                try:
                    constraints.append(self.variables[field.name] == _z3_real(value))
                except (TypeError, ValueError):
                    return False
        solver = z3.Solver()
        solver.add(constraints)
        return solver.check() == z3.sat


@dataclass(frozen=True)
class DecoderDomainAdapter:
    """Soundness boundary between exact domains and a structured decoder."""

    def constraint_for(self, result: ProjectionResult) -> dict[str, object] | None:
        if result.status != ProjectionStatus.EXACT or result.domain is None:
            return None
        return result.domain.as_decoder_constraint()

    def admits(self, result: ProjectionResult, value: object) -> bool:
        return result.status == ProjectionStatus.EXACT and result.domain is not None and result.domain.contains(value)


@dataclass(frozen=True)
class CapabilityReport:
    supported: bool
    fragments: tuple[str, ...]
    reason: str | None = None


class ProjectionCapabilityChecker:
    """Fail-closed checker for the declared exact projection fragment.

    SafetyContract callbacks are intentionally opaque Python builders in this
    prototype, so the compiler records their declared fragment explicitly.
    The checker validates the *active set* before projection; it never treats
    an unknown fragment as safe merely because Z3 happens to accept it.
    """

    def __init__(self, supported: Sequence[str] | set[str] | frozenset[str]) -> None:
        self.supported = frozenset(supported)

    def report(self, contracts: Sequence[SafetyContract]) -> CapabilityReport:
        fragments = tuple(sorted({contract.fragment for contract in contracts}))
        unsupported = [fragment for fragment in fragments if fragment not in self.supported]
        if unsupported:
            return CapabilityReport(
                False,
                fragments,
                reason=f"unsupported active contract fragment(s): {', '.join(unsupported)}",
            )
        return CapabilityReport(True, fragments)

    def check(self, contracts: Sequence[SafetyContract]) -> CapabilityReport:
        report = self.report(contracts)
        if not report.supported:
            raise ValueError(report.reason or "unsupported projection fragment")
        return report


def make_move_vertical_slice() -> tuple[
    ActionSchema,
    Mapping[str, ContextFieldSpec],
    BindingEnvironment,
    tuple[SafetyContract, ...],
]:
    schema = ActionSchema(
        fields=(
            FieldSpec.enum("direction", ("left", "right")),
            FieldSpec.real("distance", 0.0, 3.0, unit="m"),
            FieldSpec.real("speed", 0.0, 1.0, unit="m/s"),
        ),
        name="move",
    )
    context_schema = {
        "human_distance": ContextFieldSpec("human_distance", "real", "m"),
        "free_left": ContextFieldSpec("free_left", "real", "m"),
        "free_right": ContextFieldSpec("free_right", "real", "m"),
        "motion_budget": ContextFieldSpec("motion_budget", "real", "m"),
        "fragile": ContextFieldSpec("fragile", "bool"),
    }
    binding = BindingEnvironment(
        action=(
            BoundRole(SemanticRole("move.direction", "action", "enum"), "direction"),
            BoundRole(SemanticRole("move.distance", "action", "real", "m"), "distance"),
            BoundRole(SemanticRole("move.speed", "action", "real", "m/s"), "speed"),
        ),
        context=(
            BoundRole(SemanticRole("human.distance", "context", "real", "m"), "human_distance"),
            BoundRole(SemanticRole("space.free_left", "context", "real", "m"), "free_left"),
            BoundRole(SemanticRole("space.free_right", "context", "real", "m"), "free_right"),
            BoundRole(SemanticRole("motion.budget", "context", "real", "m"), "motion_budget"),
            BoundRole(SemanticRole("payload.fragile", "context", "bool"), "fragile"),
        ),
        lookups=(
            LookupRef(
                name="space.directional_clearance",
                signature="Direction -> Length[m]",
                output_kind="real",
                output_unit="m",
                resolver=lambda action, context: z3.If(
                    action["direction"] == 0,
                    _z3_real(context["free_left"]),
                    _z3_real(context["free_right"]),
                ),
            ),
        ),
    )

    def human_guard(context: BoundContext, binding: BindingEnvironment) -> bool:
        return float(context[binding.context_field("human.distance")]) < 1.5

    def fragile_guard(context: BoundContext, binding: BindingEnvironment) -> bool:
        return bool(context[binding.context_field("payload.fragile")])

    def speed_limit(action: ActionExprs, context: BoundContext, binding: BindingEnvironment) -> Any:
        return action[binding.action_field("move.speed")] <= _z3_real(0.3)

    def direction_clearance(action: ActionExprs, context: BoundContext, binding: BindingEnvironment) -> Any:
        direction = action[binding.action_field("move.direction")]
        distance = action[binding.action_field("move.distance")]
        try:
            clearance = binding.lookup("space.directional_clearance").evaluate(action, context)
        except KeyError:
            # Backward-compatible catalog bindings may materialize the same
            # relation directly as two Context fields.  New deployments
            # should register the snapshot-bound LookupRef above.
            free_left = _z3_real(context[binding.context_field("space.free_left")])
            free_right = _z3_real(context[binding.context_field("space.free_right")])
            clearance = z3.If(direction == 0, free_left, free_right)
        return distance <= clearance

    def joint_motion_budget(action: ActionExprs, context: BoundContext, binding: BindingEnvironment) -> Any:
        distance = action[binding.action_field("move.distance")]
        speed = action[binding.action_field("move.speed")]
        budget = _z3_real(context[binding.context_field("motion.budget")])
        return distance + 2 * speed <= budget

    def fragile_speed_limit(action: ActionExprs, context: BoundContext, binding: BindingEnvironment) -> Any:
        return action[binding.action_field("move.speed")] <= _z3_real(0.5)

    contracts = (
        SafetyContract(
            name="human_proximity_speed",
            scope="Mobility.Move",
            action_roles=("move.speed",),
            context_roles=("human.distance",),
            guard=human_guard,
            constraint=speed_limit,
            fragment="guarded_bounded_linear",
        ),
        SafetyContract(
            name="direction_clearance",
            scope="Mobility.Move",
            action_roles=("move.direction", "move.distance"),
            context_roles=("space.free_left", "space.free_right"),
            guard=lambda context, binding: True,
            constraint=direction_clearance,
            fragment="bounded_linear_with_enum_branch",
        ),
        SafetyContract(
            name="joint_motion_budget",
            scope="Mobility.Move",
            action_roles=("move.distance", "move.speed"),
            context_roles=("motion.budget",),
            guard=lambda context, binding: True,
            constraint=joint_motion_budget,
            fragment="bounded_linear",
        ),
        SafetyContract(
            name="fragile_payload_speed",
            scope="Mobility.Move",
            action_roles=("move.speed",),
            context_roles=("payload.fragile",),
            guard=fragile_guard,
            constraint=fragile_speed_limit,
            fragment="guarded_bounded_linear",
        ),
    )
    return schema, context_schema, binding, contracts


def make_bounded_component_deployment(
    action_name: str,
    fields: Sequence[tuple[str, float, float, str]],
    components: Sequence[Sequence[str]],
) -> tuple[
    ActionSchema,
    Mapping[str, ContextFieldSpec],
    BindingEnvironment,
    tuple[SafetyContract, ...],
]:
    """Create a generic executable deployment from a catalog template.

    Each component receives a Context-provided L1 budget.  This is a
    deliberately conservative runtime envelope: it is useful for proving
    that the catalogued DSL can reach the same projection backend, while
    avoiding invented robot- or device-specific limits.
    """
    schema = ActionSchema(
        fields=tuple(FieldSpec.real(name, lower, upper, unit=unit) for name, lower, upper, unit in fields)
    )
    context_schema = {
        f"budget_{index}": ContextFieldSpec(f"budget_{index}", "real", "normalized")
        for index, _ in enumerate(components)
    }
    binding = BindingEnvironment(action=(), context=())
    contracts: list[SafetyContract] = []
    for index, component in enumerate(components):
        names = tuple(component)
        budget_name = f"budget_{index}"

        def component_constraint(
            action: ActionExprs,
            context: BoundContext,
            _binding: BindingEnvironment,
            names: tuple[str, ...] = names,
            budget_name: str = budget_name,
        ) -> Any:
            return z3.Sum(*(z3.Abs(action[name]) for name in names)) <= _z3_real(context[budget_name])

        contracts.append(
            SafetyContract(
                name=f"{action_name}_component_budget_{index}",
                scope=f"Action.{action_name}",
                action_roles=(),
                context_roles=(),
                guard=lambda _context, _binding: True,
                constraint=component_constraint,
                fragment="bounded_component_l1",
            )
        )
    return schema, context_schema, binding, tuple(contracts)
