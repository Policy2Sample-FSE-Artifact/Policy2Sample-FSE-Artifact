"""Safety DSL validation and serialization helpers."""

from __future__ import annotations

import re
from typing import Mapping


class DSLValidationError(ValueError):
    pass


REQUIRED_TOP_LEVEL = {
    "dsl_version",
    "policy_id",
    "action_schema",
    "context_schema",
    "contracts",
    "task_admissibility",
    "provenance",
    "backend_fragment",
    "structural_schema",
}
SUPPORTED_CAPABILITIES = {
    "bounded_linear_with_enum_branch",
    "bounded_linear",
    "guarded_bounded_linear",
    "bounded_component_l1",
}
SUPPORTED_UNITS = {
    "%",
    "C",
    "K",
    "N",
    "Nm",
    "m",
    "m/s",
    "m/s2",
    "native",
    "rad",
    "rad/s",
    "rad/s2",
    "s",
}


def validate_dsl_document(document: Mapping[str, object]) -> None:
    """Validate the structural part of DSL v0.1 before execution."""
    missing = REQUIRED_TOP_LEVEL - set(document)
    if missing:
        raise DSLValidationError(f"missing top-level fields: {sorted(missing)}")
    if document["dsl_version"] != "0.1":
        raise DSLValidationError("only DSL version 0.1 is executable")

    structural_schema = document["structural_schema"]
    if not isinstance(structural_schema, Mapping):
        raise DSLValidationError("structural_schema must be an object")
    if structural_schema.get("type") != "object":
        raise DSLValidationError("structural_schema must describe an object")

    action_schema = document["action_schema"]
    if not isinstance(action_schema, Mapping) or not isinstance(action_schema.get("name"), str):
        raise DSLValidationError("action_schema.name must be a non-empty action identifier")
    fields = action_schema.get("fields")
    if not isinstance(fields, list) or not fields:
        raise DSLValidationError("action_schema.fields must be a non-empty list")
    field_names = set()
    for field in fields:
        if not isinstance(field, Mapping):
            raise DSLValidationError("each action field must be an object")
        name = field.get("name")
        if not isinstance(name, str) or name in field_names:
            raise DSLValidationError(f"invalid or duplicate action field: {name!r}")
        field_names.add(name)
        if field.get("type") not in {"enum", "real"}:
            raise DSLValidationError(f"unsupported field type for {name!r}")
        optional = field.get("optional", False)
        if not isinstance(optional, bool):
            raise DSLValidationError(f"field {name!r}.optional must be Boolean")
        if optional and "default" not in field:
            raise DSLValidationError(f"optional field {name!r} must declare a default")
        if field.get("type") == "enum":
            values = field.get("values")
            if not isinstance(values, list) or not values:
                raise DSLValidationError(f"enum field {name!r} must declare values")
            if "default" in field and field["default"] not in values:
                raise DSLValidationError(f"default for enum field {name!r} is outside its domain")
        if field.get("type") == "real":
            domain = field.get("domain")
            if not isinstance(domain, list) or len(domain) != 2:
                raise DSLValidationError(f"real field {name!r} must declare a two-element domain")
            try:
                if float(domain[0]) > float(domain[1]):
                    raise DSLValidationError(f"invalid domain for {name!r}")
            except (TypeError, ValueError) as exc:
                raise DSLValidationError(f"invalid numeric domain for {name!r}") from exc
            if "default" in field:
                try:
                    if not float(domain[0]) <= float(field["default"]) <= float(domain[1]):
                        raise DSLValidationError(f"default for real field {name!r} is outside its domain")
                except (TypeError, ValueError) as exc:
                    raise DSLValidationError(f"invalid default for real field {name!r}") from exc
        if "unit" in field and (not isinstance(field["unit"], str) or field["unit"] not in SUPPORTED_UNITS):
            raise DSLValidationError(f"invalid unit for {name!r}")

    contracts = document["contracts"]
    if not isinstance(contracts, list):
        raise DSLValidationError("contracts must be a list")
    action_names = field_names
    for contract in contracts:
        if not isinstance(contract, Mapping):
            raise DSLValidationError("each contract must be an object")
        for key in ("id", "scope", "guard", "constraint", "capability", "source"):
            if key not in contract:
                raise DSLValidationError(f"contract missing {key!r}")
        if contract["capability"] not in SUPPORTED_CAPABILITIES:
            raise DSLValidationError(f"unsupported contract capability: {contract['capability']!r}")
        scope = contract["scope"]
        if isinstance(scope, Mapping):
            scoped_fields = scope.get("fields", [])
            if not isinstance(scoped_fields, list) or any(field not in action_names for field in scoped_fields):
                raise DSLValidationError("contract references an unresolved action field")
        for clause_name in ("guard", "constraint"):
            clause = contract[clause_name]
            if not isinstance(clause, Mapping) or not isinstance(clause.get("expr"), str):
                raise DSLValidationError(f"contract {clause_name!r} must contain a textual expr")
        guard_expr = contract["guard"]["expr"]
        if re.search(r"\baction\s*\.", guard_expr):
            raise DSLValidationError("guard may only depend on ContextRef, not ActionRef")
        for action_name in action_names:
            if re.search(rf"\b{re.escape(action_name)}\b", guard_expr):
                raise DSLValidationError(
                    f"guard may not reference action field {action_name!r}"
                )

    backend_fragment = document["backend_fragment"]
    if not isinstance(backend_fragment, list) or any(
        not isinstance(fragment, str) or fragment not in SUPPORTED_CAPABILITIES
        for fragment in backend_fragment
    ):
        raise DSLValidationError("backend_fragment contains an unsupported capability")

    bindings = document.get("bindings")
    if bindings is not None:
        if not isinstance(bindings, Mapping):
            raise DSLValidationError("bindings must be an object")
        for side in ("action", "context", "lookup"):
            if side in bindings and not isinstance(bindings[side], Mapping):
                raise DSLValidationError(f"bindings.{side} must be an object")
        lookup_bindings = bindings.get("lookup", {})
        if isinstance(lookup_bindings, Mapping):
            for name, spec in lookup_bindings.items():
                if not isinstance(name, str) or not isinstance(spec, Mapping):
                    raise DSLValidationError("each LookupRef binding must be an object")
                if spec.get("snapshot_bound") is not True:
                    raise DSLValidationError(f"LookupRef {name!r} is not snapshot-bound")

    meta = document.get("meta")
    if meta is not None:
        if not isinstance(meta, Mapping):
            raise DSLValidationError("meta must be an object")
        if meta.get("context_semantics") not in {None, "versioned_snapshot"}:
            raise DSLValidationError("unsupported Context semantics")
        precision = meta.get("canonical_numeric_precision")
        if precision is not None and (not isinstance(precision, int) or precision < 0):
            raise DSLValidationError("canonical_numeric_precision must be a non-negative integer")

    task_admissibility = document["task_admissibility"]
    if not isinstance(task_admissibility, list):
        raise DSLValidationError("task_admissibility must be a list")
    provenance = document["provenance"]
    if not isinstance(provenance, Mapping) or not provenance.get("source_text"):
        raise DSLValidationError("provenance.source_text is required")
