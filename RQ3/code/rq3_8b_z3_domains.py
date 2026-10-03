"""Exact finite-lattice domains for the RQ3 8B enforcement comparison.

These masks are derived from typed Z3 constraints, not from the independent
Python evaluators used to score Prompt-only / post-check outputs.  They are
finite-lattice projections for this RQ3 workload (not the general continuous
decoder implementation).
"""
from __future__ import annotations

from pathlib import Path
import os
import sys

if os.environ.get("RQ3_SYSTEM_Z3") != "1":
    sys.path.insert(0, str(Path(__file__).resolve().parent / "vendor"))
import z3  # type: ignore


def _enumerate(solver, variables, decode):
    out = set()
    while solver.check() == z3.sat:
        model = solver.model()
        values = tuple(model.eval(v, model_completion=True).as_long() for v in variables)
        out.add(decode(values))
        solver.add(z3.Or(*(v != value for v, value in zip(variables, values))))
    return out


def robot_boundary_domain(case, actions, parse_action):
    """Return exact safe MOVE/STOP strings for the frozen Boundary-30 lattice."""
    ctx = case.context
    solver = z3.Solver()
    direction = z3.Int("rq3_rb_direction")
    distance_i = z3.Int("rq3_rb_distance_i")
    speed_i = z3.Int("rq3_rb_speed_i")
    is_stop = z3.Bool("rq3_rb_stop")
    distances = (0.4, 0.6, 0.8, 1.0, 1.2)
    speeds = (0.2, 0.3, 0.4, 0.5, 0.6)
    solver.add(direction >= 0, direction < 2, distance_i >= 0, distance_i < len(distances),
               speed_i >= 0, speed_i < len(speeds))
    d = z3.Sum(*(z3.If(distance_i == i, z3.RealVal(str(v)), z3.RealVal("0"))
                 for i, v in enumerate(distances)))
    v = z3.Sum(*(z3.If(speed_i == i, z3.RealVal(str(x)), z3.RealVal("0"))
                 for i, x in enumerate(speeds)))
    human_distance = z3.RealVal(str(float(ctx["human_distance"])))
    close_cap = max(0.15, 0.15 + 0.30 * (float(ctx["human_distance"]) - 0.5))
    speed_cap = 0.60 if float(ctx["human_distance"]) >= 2.0 else close_cap
    if ctx.get("fragile"):
        speed_cap = min(speed_cap, 0.30)
    del human_distance
    left_clear = z3.RealVal(str(float(ctx["free_left"])))
    right_clear = z3.RealVal(str(float(ctx["free_right"])))
    clearance = z3.If(direction == 0, left_clear, right_clear)
    reaction = z3.RealVal(str(float(ctx["reaction_time_s"])))
    braking = z3.RealVal(str(float(ctx["braking_deceleration_mps2"])))
    margin = z3.RealVal(str(float(ctx["safety_margin_m"])))
    occupancy = z3.RealVal(str(float(ctx["max_corridor_occupancy_s"])))
    # STOP is always safe.  Every MOVE obeys the same frozen conjunctive rule.
    solver.add(z3.Implies(z3.Not(is_stop), v <= z3.RealVal(str(speed_cap))))
    solver.add(z3.Implies(z3.Not(is_stop), d <= clearance))
    solver.add(z3.Implies(z3.Not(is_stop),
        d + v * reaction + v * v / (2 * braking) + margin <= clearance))
    solver.add(z3.Implies(z3.Not(is_stop), d <= occupancy * v))
    human_side = ctx.get("human_side")
    if human_side in ("left", "right") and float(ctx["human_distance"]) < 1.5:
        blocked_direction = 0 if human_side == "left" else 1
        solver.add(z3.Implies(z3.Not(is_stop), direction != blocked_direction))

    # First enumerate movement assignments, then include STOP iff it belongs
    # to the lattice. Use separate solver scopes to avoid a Python safe-map mask.
    solver.push()
    solver.add(z3.Not(is_stop))
    move_vars = (direction, distance_i, speed_i)
    def decode(vals):
        side = ("left", "right")[vals[0]]
        return f"MOVE(direction={side},distance={distances[vals[1]]:.1f},speed={speeds[vals[2]]:.1f})"
    result = _enumerate(solver, move_vars, decode)
    solver.pop()
    if "STOP" in actions:
        result.add("STOP")
    return result


def iot_hard_domain(case, actions):
    """Exact Z3 projection over CONFIGURE_ENERGY's discrete field lattice."""
    c = case["context"]
    complexity = case["complexity"]
    solver = z3.Solver()
    mode_i, h, d, w, e = [z3.Int(f"rq3_iot_{n}") for n in ("mode", "h", "d", "w", "e")]
    modes = ("eco", "normal", "boost")
    levels = (0, 20, 40, 60, 80, 100)
    solver.add(mode_i >= 0, mode_i < len(modes))
    for var in (h, d, w, e):
        solver.add(z3.Or(*(var == x for x in levels)))
    solver.add(6*h + 4*d + 5*w + 8*e <= int(c["budget"]))
    if complexity in ("C2", "C3", "C4"):
        if c["window"] == "open": solver.add(h <= 40)
        if c["grid"] == "constrained": solver.add(e <= 20)
        if c["water_temp_c"] >= 55: solver.add(w <= 20)
    if complexity in ("C3", "C4"):
        solver.add(h + d <= 180, e + w <= 160)
    if complexity == "C4":
        solver.add(z3.Implies(mode_i == 0, z3.And(h <= 80, d <= 80, e <= 80)))
        solver.add(z3.Implies(mode_i == 1, z3.And(h >= 20, h <= 100)))
        solver.add(z3.Implies(mode_i == 2, z3.And(h >= 80, d >= 60)))
    def decode(vals):
        return (modes[vals[0]], *vals[1:])
    return _enumerate(solver, (mode_i, h, d, w, e), decode)
