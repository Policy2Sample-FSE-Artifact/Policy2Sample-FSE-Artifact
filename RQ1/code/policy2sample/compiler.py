"""Policy-to-DSL compiler for the first executable Policy2Sample slice.

The compiler deliberately separates three concerns:

1. ``decompose`` extracts a small, auditable policy sketch;
2. ``compile`` binds the sketch to the deployment move schema and emits a
   JSON-serializable Safety DSL;
3. ``CompilationResult`` records an explicit status instead of silently
   guessing when a policy is not supported.

This is the deterministic local compiler used by the first experiment.  The
remote Qwen runner can replace ``decompose`` with an LLM-generated JSON sketch,
then pass the sketch through the same validation and deterministic compiler.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Mapping

from semantic_ir import (
    ActionSchema,
    BindingEnvironment,
    ContextFieldSpec,
    FieldSpec,
    SemanticRole,
    BoundRole,
    SafetyContract,
    OnlineExactProjectionBackend,
    ProjectionCapabilityChecker,
    make_bounded_component_deployment,
    make_move_vertical_slice,
)

from .structure import json_schema_for_action, move_json_schema


class CompilationStatus(str, Enum):
    EXECUTABLE = "EXECUTABLE"
    AMBIGUOUS = "AMBIGUOUS"
    INVALID = "INVALID"
    UNSUPPORTED = "UNSUPPORTED"


@dataclass(frozen=True)
class PolicySketch:
    """A small semantic record that is safe to inspect before DSL creation."""

    action: str
    requested_direction: str | None
    human_distance_guard: float | None
    speed_cap: float | None
    require_motion: bool
    source_text: str
    confidence: float = 1.0
    action_family: str = "motion"
    action_name: str = "move"
    parameters: tuple[str, ...] = ()
    components: tuple[tuple[str, ...], ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CompilationResult:
    status: CompilationStatus
    sketch: PolicySketch | None = None
    dsl: dict[str, object] | None = None
    reason: str | None = None
    deployment: MoveDeployment | None = None


@dataclass(frozen=True)
class MoveDeployment:
    schema: ActionSchema
    context_schema: Mapping[str, ContextFieldSpec]
    binding: BindingEnvironment
    base_contracts: tuple[SafetyContract, ...]


@dataclass(frozen=True)
class ActionTemplate:
    """A conservative, source-grounded action binding template.

    The template supplies only typed fields and bounded component envelopes.
    Device-specific limits remain Context values in the emitted DSL; the
    compiler never invents a collision-free trajectory or a hardware limit.
    """

    action: str
    family: str
    display_name: str
    fields: tuple[tuple[str, float, float, str], ...]
    components: tuple[tuple[str, ...], ...]

    def action_schema(self) -> dict[str, object]:
        return {
            "name": self.action,
            "display_name": self.display_name,
            "fields": [
                {"name": name, "type": "real", "unit": unit, "domain": [lower, upper], "precision": 2}
                for name, lower, upper, unit in self.fields
            ],
        }


def _six_axis_fields(prefix: str, unit: str, lower: float, upper: float) -> tuple[tuple[str, float, float, str], ...]:
    return tuple((axis, lower, upper, unit) for axis in (
        f"{prefix}_x", f"{prefix}_y", f"{prefix}_z",
        f"angular_x", f"angular_y", f"angular_z",
    ))


def _joint_fields(unit: str = "native") -> tuple[tuple[str, float, float, str], ...]:
    return tuple((f"joint_{index}", -3.141592653589793, 3.141592653589793, unit) for index in range(1, 7))


def _catalog() -> tuple[ActionTemplate, ...]:
    six_twist = _six_axis_fields("linear", "native", -2.0, 2.0)
    six_xyz = tuple((name, -10.0, 10.0, "native") for name in ("linear_x", "linear_y", "linear_z", "angular_x", "angular_y", "angular_z"))
    return (
        ActionTemplate("twist_planar", "motion", "publish planar Twist command", (("linear_x", -2.0, 2.0, "m/s"), ("linear_y", -2.0, 2.0, "m/s"), ("angular_z", -4.0, 4.0, "rad/s")), (("linear_x", "linear_y", "angular_z"),)),
        ActionTemplate("twist_cartesian", "motion", "publish Cartesian Twist command", six_twist, (("linear_x", "linear_y", "linear_z"), ("angular_x", "angular_y", "angular_z"))),
        ActionTemplate("servo_twist", "motion", "send Twist command through Servo", six_xyz, (("linear_x", "linear_y", "linear_z", "angular_x", "angular_y", "angular_z"),)),
        ActionTemplate("joint_position", "joint_control", "joint-position command", _joint_fields("rad"), tuple((f"joint_{index}",) for index in range(1, 7))),
        ActionTemplate("navigate_to_pose", "navigation", "NavigateToPose", (("goal_x", -100.0, 100.0, "m"), ("goal_y", -100.0, 100.0, "m"), ("goal_yaw", -3.141592653589793, 3.141592653589793, "rad")), (("goal_x", "goal_y", "goal_yaw"),)),
        ActionTemplate("light_turn_on", "home_automation", "light.turn_on", (("brightness_pct", 0.0, 100.0, "%"), ("color_temp_kelvin", 1500.0, 9000.0, "K")), (("brightness_pct", "color_temp_kelvin"),)),
        ActionTemplate("cover_position", "home_automation", "cover.set_cover_position", (("position", 0.0, 100.0, "%"), ("tilt_position", 0.0, 100.0, "%")), (("position", "tilt_position"),)),
        ActionTemplate("climate_temperature", "home_automation", "climate.set_temperature", (("temperature", 0.0, 50.0, "C"), ("target_temp_low", 0.0, 50.0, "C"), ("target_temp_high", 0.0, 50.0, "C")), (("temperature",), ("target_temp_low", "target_temp_high"))),
        ActionTemplate("navigate_through_poses", "navigation", "NavigateThroughPoses", tuple((f"goal_{goal}_{axis}", -100.0 if axis != "yaw" else -3.141592653589793, 100.0 if axis != "yaw" else 3.141592653589793, "m" if axis != "yaw" else "rad") for goal in (1, 2) for axis in ("x", "y", "yaw")), (("goal_1_x", "goal_1_y", "goal_1_yaw", "goal_2_x", "goal_2_y", "goal_2_yaw"),)),
        ActionTemplate("spin", "rotation", "Spin", (("spin_dist", -6.283185307179586, 6.283185307179586, "rad"),), (("spin_dist",),)),
        ActionTemplate("backup", "motion", "BackUp", (("backup_dist", 0.0, 10.0, "m"), ("backup_speed", 0.0, 2.0, "m/s")), (("backup_dist", "backup_speed"),)),
        ActionTemplate("drive_on_heading", "motion", "DriveOnHeading", (("dist_to_travel", 0.0, 100.0, "m"), ("speed", 0.0, 2.0, "m/s")), (("dist_to_travel", "speed"),)),
        ActionTemplate("dock_robot", "navigation", "DockRobot", (("dock_pose_x", -100.0, 100.0, "m"), ("dock_pose_y", -100.0, 100.0, "m"), ("dock_pose_yaw", -3.141592653589793, 3.141592653589793, "rad"), ("max_staging_time", 0.0, 300.0, "s")), (("dock_pose_x", "dock_pose_y", "dock_pose_yaw", "max_staging_time"),)),
        ActionTemplate("wait", "timing", "Wait", (("wait_duration", 0.0, 300.0, "s"),), (("wait_duration",),)),
        ActionTemplate("accel", "acceleration", "publish Accel command", tuple((name, -10.0, 10.0, "m/s2" if name.startswith("linear") else "rad/s2") for name in ("linear_x", "linear_y", "linear_z", "angular_x", "angular_y", "angular_z")), (("linear_x", "linear_y", "linear_z"), ("angular_x", "angular_y", "angular_z"))),
        ActionTemplate("accel_stamped", "acceleration", "publish AccelStamped command", tuple((name, -10.0, 10.0, "m/s2" if name.startswith("linear") else "rad/s2") for name in ("linear_x", "linear_y", "linear_z", "angular_x", "angular_y", "angular_z")), (("linear_x", "linear_y", "linear_z"), ("angular_x", "angular_y", "angular_z"))),
        ActionTemplate("wrench", "force_torque", "publish Wrench command", tuple((name, -100.0, 100.0, "N" if name.startswith("force") else "Nm") for name in ("force_x", "force_y", "force_z", "torque_x", "torque_y", "torque_z")), (("force_x", "force_y", "force_z"), ("torque_x", "torque_y", "torque_z"))),
        ActionTemplate("wrench_stamped", "force_torque", "publish WrenchStamped command", tuple((name, -100.0, 100.0, "N" if name.startswith("force") else "Nm") for name in ("force_x", "force_y", "force_z", "torque_x", "torque_y", "torque_z")), (("force_x", "force_y", "force_z"), ("torque_x", "torque_y", "torque_z"))),
        ActionTemplate("pose", "navigation", "publish Pose target", (("position_x", -100.0, 100.0, "m"), ("position_y", -100.0, 100.0, "m"), ("position_z", -100.0, 100.0, "m"), ("orientation_roll", -3.141592653589793, 3.141592653589793, "rad"), ("orientation_pitch", -3.141592653589793, 3.141592653589793, "rad"), ("orientation_yaw", -3.141592653589793, 3.141592653589793, "rad")), (("position_x", "position_y", "position_z", "orientation_roll", "orientation_pitch", "orientation_yaw"),)),
        ActionTemplate("pose_stamped", "navigation", "publish PoseStamped target", (("position_x", -100.0, 100.0, "m"), ("position_y", -100.0, 100.0, "m"), ("position_z", -100.0, 100.0, "m"), ("orientation_roll", -3.141592653589793, 3.141592653589793, "rad"), ("orientation_pitch", -3.141592653589793, 3.141592653589793, "rad"), ("orientation_yaw", -3.141592653589793, 3.141592653589793, "rad")), (("position_x", "position_y", "position_z", "orientation_roll", "orientation_pitch", "orientation_yaw"),)),
        ActionTemplate("point", "geometry", "publish Point target", tuple((axis, -100.0, 100.0, "m") for axis in ("x", "y", "z")), (("x", "y", "z"),)),
        ActionTemplate("vector3", "geometry", "publish Vector3 command", tuple((axis, -100.0, 100.0, "native") for axis in ("x", "y", "z")), (("x", "y", "z"),)),
        ActionTemplate("joint_jog", "motion", "JointJog command", _joint_fields("rad"), tuple((f"joint_{index}",) for index in range(1, 7))),
        ActionTemplate("pose_command", "navigation", "Pose command", (("position_x", -100.0, 100.0, "m"), ("position_y", -100.0, 100.0, "m"), ("position_z", -100.0, 100.0, "m"), ("orientation_roll", -3.141592653589793, 3.141592653589793, "rad"), ("orientation_pitch", -3.141592653589793, 3.141592653589793, "rad"), ("orientation_yaw", -3.141592653589793, 3.141592653589793, "rad")), (("position_x", "position_y", "position_z", "orientation_roll", "orientation_pitch", "orientation_yaw"),)),
        ActionTemplate("joint_velocity", "motion", "joint velocity command", tuple((f"joint_{index}", -2.0, 2.0, "rad/s") for index in range(1, 7)), tuple((f"joint_{index}",) for index in range(1, 7))),
        ActionTemplate("joint_acceleration", "acceleration", "joint acceleration command", tuple((f"joint_{index}", -10.0, 10.0, "rad/s2") for index in range(1, 7)), tuple((f"joint_{index}",) for index in range(1, 7))),
        ActionTemplate("joint_effort", "force_torque", "joint effort command", tuple((f"joint_{index}", -100.0, 100.0, "Nm") for index in range(1, 7)), tuple((f"joint_{index}",) for index in range(1, 7))),
        ActionTemplate("vacuum_fan", "home_automation", "vacuum.set_fan_speed", (("fan_speed", 0.0, 100.0, "%"),), (("fan_speed",),)),
        ActionTemplate("climate_humidity", "home_automation", "climate.set_humidity", (("humidity", 0.0, 100.0, "%"),), (("humidity",),)),
        ActionTemplate("valve_position", "home_automation", "valve.set_valve_position", (("position", 0.0, 100.0, "%"),), (("position",),)),
    )


ACTION_TEMPLATES = {template.action: template for template in _catalog()}


def default_move_deployment() -> MoveDeployment:
    schema, context_schema, binding, contracts = make_move_vertical_slice()
    return MoveDeployment(schema, context_schema, binding, contracts)


class PolicyCompiler:
    """Compiler for the move slice plus a conservative real-action catalog.

    The catalog is intentionally explicit: each action family gets a typed
    schema and bounded dependency components.  Unknown actions remain
    ``UNSUPPORTED`` instead of being guessed from arbitrary nouns.
    """

    SUPPORTED_ACTIONS = frozenset({"move", *ACTION_TEMPLATES})
    DIRECTIONS = frozenset({"left", "right"})

    def __init__(self, deployment: MoveDeployment | None = None) -> None:
        self.deployment = deployment or default_move_deployment()

    def deployment_for(self, sketch: PolicySketch):
        """Return the executable IR deployment selected by the action catalog."""
        if sketch.action == "move":
            return self.deployment
        template = ACTION_TEMPLATES.get(sketch.action)
        if template is None:
            raise ValueError(f"no deployment template for action {sketch.action!r}")
        schema, context_schema, binding, contracts = make_bounded_component_deployment(
            template.action, template.fields, template.components
        )
        return MoveDeployment(schema, context_schema, binding, contracts)

    @staticmethod
    def _detect_template(text: str) -> ActionTemplate | None:
        lowered = text.lower()
        detectors = (
            ("navigate_through_poses", ("navigatethroughposes", "navigate through poses")),
            ("navigate_to_pose", ("navigatetopose", "navigate to pose")),
            ("dock_robot", ("dockrobot", "dock robot")),
            ("drive_on_heading", ("driveonheading", "drive on heading")),
            ("backup", ("backup", "back up")),
            ("spin", ("spin", "原地旋转")),
            ("wait", ("wait", "等待")),
            ("light_turn_on", ("light.turn_on", "打开灯")),
            ("cover_position", ("cover.set_cover_position",)),
            ("climate_temperature", ("climate.set_temperature",)),
            ("vacuum_fan", ("vacuum.set_fan_speed", "设置吸力")),
            ("climate_humidity", ("climate.set_humidity",)),
            ("valve_position", ("valve.set_valve_position",)),
            ("twist_planar", ("planar twist", "平面 twist")),
            ("twist_cartesian", ("cartesian twist", "笛卡尔 twist")),
            ("servo_twist", ("twist command", "twist 指令", "twist命令")),
            ("joint_jog", ("jointjog", "joint jog")),
            ("joint_position", ("joint-position", "joint position", "关节位置")),
            ("joint_velocity", ("joint velocity", "关节速度")),
            ("joint_acceleration", ("joint acceleration", "关节加速度")),
            ("joint_effort", ("joint effort", "关节力矩", "关节力")),
            ("accel_stamped", ("accelstamped", "accel stamped")),
            ("accel", ("accel command", "accel 指令", "线加速度")),
            ("wrench_stamped", ("wrenchstamped", "wrench stamped")),
            ("wrench", ("wrench command", "wrench 指令", "力和力矩")),
            ("pose_stamped", ("posestamped", "pose stamped")),
            ("pose_command", ("pose command", "pose 指令")),
            ("pose", ("pose target", "publish pose", "pose 目标")),
            ("point", ("point target", "point 目标")),
            ("vector3", ("vector3", "vector 3")),
        )
        for action, markers in detectors:
            if any(marker in lowered for marker in markers):
                return ACTION_TEMPLATES[action]
        if any(token in text for token in ("移动", "行驶", "走")) or "move" in lowered:
            return None
        return None

    def decompose(self, policy_text: str) -> CompilationResult:
        text = policy_text.strip()
        if not text:
            return CompilationResult(CompilationStatus.INVALID, reason="empty policy")

        lowered = text.lower()
        template = self._detect_template(text)
        if template is None and not any(token in text for token in ("移动", "行驶", "move", "走")):
            return CompilationResult(
                CompilationStatus.UNSUPPORTED,
                reason="no action in the supported deployment catalog was detected",
            )

        if template is not None and template.action != "move":
            sketch = PolicySketch(
                action=template.action,
                requested_direction=None,
                human_distance_guard=None,
                speed_cap=None,
                require_motion=True,
                source_text=text,
                action_family=template.family,
                action_name=template.display_name,
                parameters=tuple(field[0] for field in template.fields),
                components=template.components,
            )
            return CompilationResult(CompilationStatus.EXECUTABLE, sketch=sketch)

        direction: str | None = None
        direction_hits = []
        for token, value in (("左", "left"), ("left", "left"), ("右", "right"), ("right", "right")):
            if token in text or token in lowered:
                direction_hits.append(value)
        if direction_hits and len(set(direction_hits)) > 1:
            return CompilationResult(
                CompilationStatus.AMBIGUOUS,
                reason="policy contains both left and right direction requirements",
            )
        if direction_hits:
            direction = direction_hits[0]

        guard = None
        human_match = re.search(r"(?:人|human)[^0-9]{0,12}(\d+(?:\.\d+)?)\s*(?:米|m)", text, re.IGNORECASE)
        if human_match:
            guard = float(human_match.group(1))
        elif any(token in text or token in lowered for token in ("靠近人", "有人靠近", "near human", "human near")):
            guard = 1.5

        speed_cap = None
        speed_match = re.search(r"(?:速度|speed)[^0-9]{0,12}(?:不超过|小于等于|<=|at most)?\s*(\d+(?:\.\d+)?)\s*(?:米/秒|m/s)?", text, re.IGNORECASE)
        if speed_match:
            speed_cap = float(speed_match.group(1))
        elif any(token in text or token in lowered for token in ("慢速", "低速", "slow")):
            speed_cap = 0.3

        require_motion = not any(token in text or token in lowered for token in ("停车", "停止", "等待", "stop", "idle"))
        sketch = PolicySketch(
            action="move",
            requested_direction=direction,
            human_distance_guard=guard,
            speed_cap=speed_cap,
            require_motion=require_motion,
            source_text=text,
            action_family="motion",
            action_name="move",
            parameters=("direction", "distance", "speed"),
            components=(("direction", "distance", "speed"),),
        )
        if sketch.requested_direction is None and sketch.human_distance_guard is None and sketch.speed_cap is None:
            return CompilationResult(
                CompilationStatus.AMBIGUOUS,
                sketch=sketch,
                reason="no supported action field or numeric safety relation was extracted",
            )
        return CompilationResult(CompilationStatus.EXECUTABLE, sketch=sketch)

    def compile_sketch(self, sketch: PolicySketch) -> CompilationResult:
        if sketch.action != "move":
            template = ACTION_TEMPLATES.get(sketch.action)
            if template is None:
                return CompilationResult(CompilationStatus.UNSUPPORTED, sketch=sketch, reason="unsupported action")
            return self._compile_catalogued_sketch(sketch, template)
        if sketch.action not in self.SUPPORTED_ACTIONS:
            return CompilationResult(CompilationStatus.UNSUPPORTED, sketch=sketch, reason="unsupported action")
        if sketch.requested_direction not in (None, *self.DIRECTIONS):
            return CompilationResult(CompilationStatus.INVALID, sketch=sketch, reason="invalid direction")
        if sketch.speed_cap is not None and not 0.0 <= sketch.speed_cap <= 1.0:
            return CompilationResult(CompilationStatus.INVALID, sketch=sketch, reason="speed cap outside action domain")
        if sketch.human_distance_guard is not None and sketch.human_distance_guard < 0:
            return CompilationResult(CompilationStatus.INVALID, sketch=sketch, reason="negative human-distance guard")

        contracts = [
            {
                "id": "direction_clearance",
                "scope": {"action": "move", "fields": ["direction", "distance"]},
                "guard": {"expr": "true"},
                "constraint": {"expr": "distance <= clearance(direction)"},
                "capability": "bounded_linear_with_enum_branch",
                "source": "deployment move clearance contract",
            },
            {
                "id": "joint_motion_budget",
                "scope": {"action": "move", "fields": ["distance", "speed"]},
                "guard": {"expr": "true"},
                "constraint": {"expr": "distance + 2 * speed <= motion_budget"},
                "capability": "bounded_linear",
                "source": "deployment motion budget contract",
            },
        ]
        if sketch.human_distance_guard is not None:
            contracts.append(
                {
                    "id": "human_proximity_speed",
                    "scope": {"action": "move", "fields": ["speed"]},
                    "guard": {"expr": f"human_distance < {sketch.human_distance_guard}"},
                    "constraint": {"expr": "speed <= 0.3"},
                    "capability": "guarded_bounded_linear",
                    "source": sketch.source_text,
                }
            )
        if sketch.speed_cap is not None:
            contracts.append(
                {
                    "id": "policy_speed_cap",
                    "scope": {"action": "move", "fields": ["speed"]},
                    "guard": {"expr": "true"},
                    "constraint": {"expr": f"speed <= {sketch.speed_cap:g}"},
                    "capability": "bounded_linear",
                    "source": sketch.source_text,
                }
            )

        deployment = self.deployment
        runtime_contracts = list(deployment.base_contracts)
        runtime_contracts.extend(
            item
            for item in (
                direction_task_contract(sketch.requested_direction),
                positive_distance_task_contract() if sketch.require_motion else None,
                speed_cap_contract(sketch.speed_cap),
            )
            if item is not None
        )
        compile_capability = ProjectionCapabilityChecker(OnlineExactProjectionBackend.capabilities).report(
            runtime_contracts
        )
        if not compile_capability.supported:
            return CompilationResult(
                CompilationStatus.UNSUPPORTED,
                sketch=sketch,
                reason=compile_capability.reason,
            )

        task_constraints = []
        if sketch.require_motion:
            task_constraints.append({"id": "must_move", "expr": "distance > 0"})
        if sketch.requested_direction:
            task_constraints.append({"id": "preferred_direction", "expr": f"direction == '{sketch.requested_direction}'"})

        dsl = {
            "dsl_version": "0.1",
            "policy_id": "compiled_move_policy",
            "action_schema": {
                "name": "move",
                "fields": [
                    {"name": "direction", "type": "enum", "values": ["left", "right"]},
                    {"name": "distance", "type": "real", "unit": "m", "domain": [0.0, 3.0], "precision": 2},
                    {"name": "speed", "type": "real", "unit": "m/s", "domain": [0.0, 1.0], "precision": 2},
                ],
            },
            "context_schema": {name: {"type": spec.kind, "unit": spec.unit} for name, spec in self.deployment.context_schema.items()},
            "bindings": {
                "action": {
                    role.name: entry.concrete_name
                    for entry in self.deployment.binding.action
                    for role in (entry.role,)
                },
                "context": {
                    role.name: entry.concrete_name
                    for entry in self.deployment.binding.context
                    for role in (entry.role,)
                },
                "lookup": {
                    lookup.name: {
                        "signature": lookup.signature,
                        "output_type": lookup.output_kind,
                        "unit": lookup.output_unit,
                        "snapshot_bound": True,
                    }
                    for lookup in self.deployment.binding.lookups
                },
            },
            "contracts": contracts,
            "task_admissibility": task_constraints,
            "provenance": {"source_text": sketch.source_text, "compiler": "policy2sample-v0.1"},
            "backend_fragment": sorted({contract["capability"] for contract in contracts}),
            "meta": {
                "context_semantics": "versioned_snapshot",
                "freshness": "provider_precondition",
                "canonical_numeric_precision": 2,
            },
            # Static syntax for XGrammar/JSON-schema engines.  Contextual
            # numeric bounds are intentionally not placed here.
            "structural_schema": move_json_schema(),
        }
        return CompilationResult(
            CompilationStatus.EXECUTABLE,
            sketch=sketch,
            dsl=dsl,
            deployment=MoveDeployment(
                deployment.schema,
                deployment.context_schema,
                deployment.binding,
                tuple(runtime_contracts),
            ),
        )

    def _compile_catalogued_sketch(self, sketch: PolicySketch, template: ActionTemplate) -> CompilationResult:
        """Emit a typed DSL for a known action without inventing device facts."""
        action_schema = template.action_schema()
        context_schema = {
            f"budget_{index}": {"type": "real", "unit": "normalized"}
            for index, _ in enumerate(template.components)
        }
        contracts = []
        for index, component in enumerate(template.components):
            contracts.append(
                {
                    "id": f"component_budget_{index}",
                    "scope": {"action": template.action, "fields": list(component)},
                    "guard": {"expr": "true"},
                    "constraint": {"expr": f"sum_abs({', '.join(component)}) <= budget_{index}"},
                    "capability": "bounded_component_l1",
                    "source": "deployment component envelope; budget is Context supplied",
                }
            )
        runtime_schema, runtime_context_schema, runtime_binding, runtime_contracts = make_bounded_component_deployment(
            template.action,
            template.fields,
            template.components,
        )
        compile_capability = ProjectionCapabilityChecker(OnlineExactProjectionBackend.capabilities).report(
            runtime_contracts
        )
        if not compile_capability.supported:
            return CompilationResult(
                CompilationStatus.UNSUPPORTED,
                sketch=sketch,
                reason=compile_capability.reason,
            )
        deployment = MoveDeployment(
            schema=runtime_schema,
            context_schema=runtime_context_schema,
            binding=runtime_binding,
            base_contracts=runtime_contracts,
        )
        dsl = {
            "dsl_version": "0.1",
            "policy_id": f"compiled_{template.action}",
            "action_schema": action_schema,
            "context_schema": context_schema,
            "bindings": {"action": {}, "context": {}, "lookup": {}},
            "contracts": contracts,
            "task_admissibility": [],
            "provenance": {
                "source_text": sketch.source_text,
                "compiler": "policy2sample-v0.2-action-catalog",
                "action_family": template.family,
                "source_action_name": template.display_name,
                "support_level": "typed_dsl_and_component_envelope",
            },
            "backend_fragment": ["bounded_component_l1"],
            "meta": {
                "context_semantics": "versioned_snapshot",
                "freshness": "provider_precondition",
                "canonical_numeric_precision": 2,
            },
            "structural_schema": json_schema_for_action(action_schema),
        }
        return CompilationResult(CompilationStatus.EXECUTABLE, sketch=sketch, dsl=dsl, deployment=deployment)

    def compile(self, policy_text: str) -> CompilationResult:
        decomposed = self.decompose(policy_text)
        if decomposed.status != CompilationStatus.EXECUTABLE or decomposed.sketch is None:
            return decomposed
        return self.compile_sketch(decomposed.sketch)

    def compile_llm_sketch(self, payload: Mapping[str, Any], source_text: str) -> CompilationResult:
        """Compile an LLM JSON sketch through the deterministic path."""
        action_value = str(payload.get("action", ""))
        if action_value != "move" and action_value in ACTION_TEMPLATES:
            template = ACTION_TEMPLATES[action_value]
            sketch = PolicySketch(
                action=template.action,
                requested_direction=None,
                human_distance_guard=None,
                speed_cap=None,
                require_motion=bool(payload.get("require_motion", True)),
                source_text=source_text,
                confidence=float(payload.get("confidence", 0.0)),
                action_family=template.family,
                action_name=template.display_name,
                parameters=tuple(field[0] for field in template.fields),
                components=template.components,
            )
            return self.compile_sketch(sketch)
        try:
            sketch = PolicySketch(
                action=action_value,
                requested_direction=payload.get("requested_direction"),
                human_distance_guard=(None if payload.get("human_distance_guard") is None else float(payload["human_distance_guard"])),
                speed_cap=(None if payload.get("speed_cap") is None else float(payload["speed_cap"])),
                require_motion=bool(payload.get("require_motion", True)),
                source_text=source_text,
                confidence=float(payload.get("confidence", 0.0)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            return CompilationResult(CompilationStatus.INVALID, reason=f"invalid LLM sketch: {exc}")
        return self.compile_sketch(sketch)


def direction_task_contract(direction: str | None) -> SafetyContract | None:
    """Build a task contract for the existing semantic IR backend."""
    if direction is None:
        return None
    code = 0 if direction == "left" else 1
    return SafetyContract(
        name="task_preferred_direction",
        scope="Task.Move",
        action_roles=("move.direction",),
        context_roles=(),
        guard=lambda context, binding: True,
        constraint=lambda action, context, binding: action[binding.action_field("move.direction")] == code,
    )


def positive_distance_task_contract() -> SafetyContract:
    # v0.1 uses two-decimal canonical numeric strings, so the smallest
    # representable positive motion is 0.01 m.  Encoding that floor avoids an
    # open real interval that the first interval adapter cannot represent.
    return SafetyContract(
        name="task_must_move",
        scope="Task.Move",
        action_roles=("move.distance",),
        context_roles=(),
        guard=lambda context, binding: True,
        constraint=lambda action, context, binding: action[binding.action_field("move.distance")] >= 0.01,
    )


def speed_cap_contract(speed_cap: float | None) -> SafetyContract | None:
    if speed_cap is None:
        return None
    cap = float(speed_cap)
    return SafetyContract(
        name="policy_speed_cap",
        scope="Policy.Move",
        action_roles=("move.speed",),
        context_roles=(),
        guard=lambda context, binding: True,
        constraint=lambda action, context, binding: action[binding.action_field("move.speed")] <= cap,
    )
