"""Exact continuous linear-domain backend for the PPDC method.

The backend supports bounded continuous variables and linear inequalities whose
right-hand sides are affine functions of the runtime Context.  Deployment-time
compilation uses Fourier--Motzkin elimination to build symbolic prefix
polyhedra.  Admission-time specialization evaluates Context expressions and
queries the exact interval for the next variable.

This is a self-contained analytical projector, not an SMT/LP wrapper. It
removes finite value enumeration and exposes the compile/runtime boundary used
by the paper method.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


Context = Mapping[str, object]
EPS = 1e-10


def context_value(context: Context, path: tuple[object, ...]) -> float:
    value: object = context
    for key in path:
        value = value[key]  # type: ignore[index]
    return float(value)


@dataclass(frozen=True)
class AffineContext:
    constant: float = 0.0
    terms: tuple[tuple[tuple[object, ...], float], ...] = ()

    @classmethod
    def constant_value(cls, value: float) -> "AffineContext":
        return cls(constant=float(value))

    @classmethod
    def lookup(cls, path: tuple[object, ...]) -> "AffineContext":
        return cls(terms=((path, 1.0),))

    def evaluate(self, context: Context) -> float:
        return self.constant + sum(coef * context_value(context, path) for path, coef in self.terms)

    def __add__(self, other: "AffineContext") -> "AffineContext":
        terms = dict(self.terms)
        for path, coef in other.terms:
            terms[path] = terms.get(path, 0.0) + coef
        return AffineContext(self.constant + other.constant, tuple(sorted(terms.items(), key=repr)))

    def __sub__(self, other: "AffineContext") -> "AffineContext":
        return self + other.scale(-1.0)

    def scale(self, factor: float) -> "AffineContext":
        return AffineContext(
            self.constant * factor,
            tuple((path, coef * factor) for path, coef in self.terms),
        )

    def render(self) -> str:
        pieces = [repr(self.constant)] if abs(self.constant) > EPS else []
        pieces.extend(f"{coef}*{'.'.join(map(str, path))}" for path, coef in self.terms if abs(coef) > EPS)
        return " + ".join(pieces) if pieces else "0"


@dataclass(frozen=True)
class ContinuousVariable:
    name: str
    lower: float
    upper: float


@dataclass(frozen=True)
class LinearInequality:
    """An inequality of the form sum(coeff[var] * var) <= rhs(context)."""

    coefficients: tuple[tuple[str, float], ...]
    rhs: AffineContext

    @classmethod
    def make(cls, coefficients: Mapping[str, float], rhs: AffineContext) -> "LinearInequality":
        normalized = tuple((name, float(coef)) for name, coef in sorted(coefficients.items()) if abs(coef) > EPS)
        return cls(normalized, rhs)

    def coefficient(self, variable: str) -> float:
        return dict(self.coefficients).get(variable, 0.0)

    def without(self, variable: str) -> "LinearInequality":
        return LinearInequality.make(
            {name: coef for name, coef in self.coefficients if name != variable},
            self.rhs,
        )

    def substitute(self, assignment: Mapping[str, float]) -> tuple[float, AffineContext]:
        constant = sum(coef * float(assignment[name]) for name, coef in self.coefficients if name in assignment)
        remaining = {
            name: coef for name, coef in self.coefficients if name not in assignment
        }
        return constant, self.rhs


@dataclass(frozen=True)
class ContinuousLinearModel:
    variables: tuple[ContinuousVariable, ...]
    constraints: tuple[LinearInequality, ...]

    @property
    def order(self) -> tuple[str, ...]:
        return tuple(variable.name for variable in self.variables)

    def base_inequalities(self) -> list[LinearInequality]:
        inequalities = list(self.constraints)
        for variable in self.variables:
            inequalities.append(
                LinearInequality.make(
                    {variable.name: 1.0},
                    AffineContext.constant_value(variable.upper),
                )
            )
            inequalities.append(
                LinearInequality.make(
                    {variable.name: -1.0},
                    AffineContext.constant_value(-variable.lower),
                )
            )
        return inequalities


def dependency_components(model: ContinuousLinearModel) -> tuple[tuple[str, ...], ...]:
    """Return connected safety-variable components in model order."""

    adjacency = {name: set() for name in model.order}
    for inequality in model.constraints:
        names = [name for name, _ in inequality.coefficients]
        for left in names:
            for right in names:
                if left != right:
                    adjacency[left].add(right)
    unseen = set(model.order)
    components: list[tuple[str, ...]] = []
    for seed in model.order:
        if seed not in unseen:
            continue
        stack = [seed]
        unseen.remove(seed)
        component = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbour in adjacency[current]:
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    stack.append(neighbour)
        components.append(tuple(name for name in model.order if name in component))
    return tuple(components)


def decompose_model(model: ContinuousLinearModel) -> tuple[ContinuousLinearModel, ...]:
    """Split a model into independent component models for shared baselines."""

    result = []
    for component in dependency_components(model):
        component_set = set(component)
        variables = tuple(variable for variable in model.variables if variable.name in component_set)
        constraints = tuple(
            inequality
            for inequality in model.constraints
            if all(name in component_set for name, _ in inequality.coefficients)
        )
        result.append(ContinuousLinearModel(variables, constraints))
    return tuple(result)


def eliminate_variable(
    inequalities: Sequence[LinearInequality],
    variable: str,
) -> list[LinearInequality]:
    positive = [inequality for inequality in inequalities if inequality.coefficient(variable) > EPS]
    negative = [inequality for inequality in inequalities if inequality.coefficient(variable) < -EPS]
    zero = [inequality.without(variable) for inequality in inequalities if abs(inequality.coefficient(variable)) <= EPS]
    result = list(zero)
    for upper in positive:
        a_upper = upper.coefficient(variable)
        for lower in negative:
            a_lower = lower.coefficient(variable)
            # (-a_lower) * upper + a_upper * lower eliminates variable.
            combined_rhs = upper.rhs.scale(-a_lower) + lower.rhs.scale(a_upper)
            combined_coefficients: dict[str, float] = {}
            for name, coef in upper.coefficients:
                if name != variable:
                    combined_coefficients[name] = combined_coefficients.get(name, 0.0) + (-a_lower) * coef
            for name, coef in lower.coefficients:
                if name != variable:
                    combined_coefficients[name] = combined_coefficients.get(name, 0.0) + a_upper * coef
            result.append(LinearInequality.make(combined_coefficients, combined_rhs))
    return _deduplicate(result)


def _deduplicate(inequalities: Sequence[LinearInequality]) -> list[LinearInequality]:
    seen: set[tuple[tuple[tuple[str, float], ...], AffineContext]] = set()
    result = []
    for inequality in inequalities:
        key = (inequality.coefficients, inequality.rhs)
        if key not in seen:
            seen.add(key)
            result.append(inequality)
    return result


@dataclass(frozen=True)
class ContinuousCompiledPPDC:
    order: tuple[str, ...]
    prefix_inequalities: Mapping[int, tuple[LinearInequality, ...]]
    compiled_inequality_count: int

    @classmethod
    def compile(cls, model: ContinuousLinearModel) -> "ContinuousCompiledPPDC":
        current = model.base_inequalities()
        prefix_inequalities: dict[int, tuple[LinearInequality, ...]] = {}
        for index in range(len(model.order) - 1, -1, -1):
            if index == len(model.order) - 1:
                prefix_inequalities[index] = tuple(current)
            else:
                current = eliminate_variable(current, model.order[index + 1])
                prefix_inequalities[index] = tuple(current)
        return cls(
            order=model.order,
            prefix_inequalities=prefix_inequalities,
            compiled_inequality_count=sum(len(items) for items in prefix_inequalities.values()),
        )

    def specialize(self, context: Context) -> "ContinuousSpecializedPPDC":
        return ContinuousSpecializedPPDC(self, context)


@dataclass(frozen=True)
class ContinuousInterval:
    lower: float
    upper: float


@dataclass(frozen=True)
class ContinuousSpecializedPPDC:
    compiled: ContinuousCompiledPPDC
    context: Context

    def next_interval(self, prefix: Sequence[float]) -> ContinuousInterval | None:
        index = len(prefix)
        if index >= len(self.compiled.order):
            return None
        variable = self.compiled.order[index]
        assignment = dict(zip(self.compiled.order[:index], prefix))
        lower = float("-inf")
        upper = float("inf")
        for inequality in self.compiled.prefix_inequalities[index]:
            coefficient = inequality.coefficient(variable)
            assigned_constant = sum(
                coef * float(assignment[name])
                for name, coef in inequality.coefficients
                if name in assignment
            )
            rhs = inequality.rhs.evaluate(self.context)
            residual = rhs - assigned_constant
            if coefficient > EPS:
                upper = min(upper, residual / coefficient)
            elif coefficient < -EPS:
                lower = max(lower, residual / coefficient)
            elif assigned_constant > rhs + EPS:
                return None
        if lower > upper + EPS:
            return None
        return ContinuousInterval(lower, upper)


def online_exact_interval(
    model: ContinuousLinearModel,
    context: Context,
    prefix: Sequence[float],
) -> ContinuousInterval | None:
    index = len(prefix)
    if index >= len(model.order):
        return None
    inequalities = model.base_inequalities()
    for future_index in range(len(model.order) - 1, index, -1):
        inequalities = eliminate_variable(inequalities, model.order[future_index])
    compiled = ContinuousCompiledPPDC(model.order, {index: tuple(inequalities)}, len(inequalities))
    return compiled.specialize(context).next_interval(prefix)


def make_continuous_motion_model(n_parameters: int) -> ContinuousLinearModel:
    if n_parameters < 2:
        raise ValueError("n_parameters must be at least 2")
    variables = tuple(
        ContinuousVariable(f"x{i}", 0.0, 1.0)
        for i in range(n_parameters)
    )
    constraints = [
        LinearInequality.make(
            {variable.name: 1.0 for variable in variables},
            AffineContext.lookup(("budget",)),
        ),
    ]
    for variable in variables:
        constraints.append(
            LinearInequality.make(
                {variable.name: 1.0},
                AffineContext.lookup(("limit",)),
            )
        )
    for index in range(n_parameters - 1):
        constraints.append(
            LinearInequality.make(
                {f"x{index}": 1.0, f"x{index + 1}": 1.0},
                AffineContext.lookup(("pair_budget",)),
            )
        )
    return ContinuousLinearModel(variables, tuple(constraints))
