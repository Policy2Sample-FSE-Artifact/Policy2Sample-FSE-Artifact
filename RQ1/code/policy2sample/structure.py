"""Static output structure for the Policy2Sample decoder boundary.

The schema deliberately describes syntax and field types only.  Contextual
numeric bounds remain in the semantic mask, because they change per request.
"""

from __future__ import annotations

from typing import Mapping


def move_json_schema() -> dict[str, object]:
    """Return the static JSON structure accepted by the move decoder."""
    decimal = {
        "type": "string",
        "pattern": r"^[0-9]+\.[0-9]{2}$",
        "description": "canonical fixed-point decimal; semantic bounds are dynamic",
    }
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "direction": {"type": "string", "enum": ["left", "right"]},
            "distance": decimal,
            "speed": decimal,
        },
        "required": ["direction", "distance", "speed"],
        "additionalProperties": False,
    }


def json_schema_for_action(action_schema: Mapping[str, object]) -> dict[str, object]:
    """Build a static JSON schema from a catalogued action schema.

    Context-dependent safety bounds deliberately stay outside this schema. The
    schema only constrains the action's field names and scalar representation;
    the semantic runtime supplies the request-specific domain.
    """
    properties: dict[str, object] = {}
    required: list[str] = []
    fields = action_schema.get("fields")
    if not isinstance(fields, list):
        raise ValueError("action_schema.fields must be a list")
    for field in fields:
        if not isinstance(field, Mapping):
            raise ValueError("action schema field must be an object")
        name = field.get("name")
        kind = field.get("type")
        if not isinstance(name, str) or kind not in {"enum", "real"}:
            raise ValueError("invalid action schema field")
        if kind == "enum":
            values = field.get("values")
            if not isinstance(values, list) or not values:
                raise ValueError(f"enum field {name!r} has no values")
            properties[name] = {"type": "string", "enum": values}
        else:
            properties[name] = {"type": "number"}
        required.append(name)
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }
def structural_schema_for_dsl(document: Mapping[str, object]) -> dict[str, object]:
    """Derive the static JSON schema for a validated v0.1 DSL document."""
    action_schema = document.get("action_schema")
    if not isinstance(action_schema, Mapping):
        raise ValueError("action_schema must be an object")
    if action_schema.get("name") == "move":
        return move_json_schema()
    return json_schema_for_action(action_schema)
