"""Offline ROS-payload projections. Does NOT subscribe, publish, or command motors."""

import numpy as np

from .workflow import ContractError


def named_positions(names, positions, required_names):
    """Reorder by identity, never silently by arrival order or zero-filled gaps."""
    if len(names) != len(set(names)) or len(required_names) != len(set(required_names)):
        raise ContractError("duplicate joint name")
    values = np.asarray(positions, dtype=np.float64)
    if values.shape != (len(names),) or not np.isfinite(values).all():
        raise ContractError("invalid joint positions")
    indices = {name: i for i, name in enumerate(names)}
    if any(name not in indices for name in required_names):
        raise ContractError("missing required joint; do not substitute zero or measured state")
    return values[[indices[name] for name in required_names]]


def project_arm_state(names, positions, wire_map):
    """/joint_states uses URDF names; omit unmodeled neck slots from the candidate."""
    return named_positions(names, positions, [r["urdf"] for r in wire_map if r["urdf"] is not None])


def project_single_target(names, points, wire_map):
    """Require a full, single commanded target, not an interpolated/last-waypoint guess.

    Partial per-arm streams require a separately verified commanded-setpoint cache
    with timestamps/ownership. They are intentionally rejected by this v1 helper.
    """
    if len(points) != 1:
        raise ContractError("require one point; controller consumes points[-1], not an action chunk")
    aliases = {alias: row["urdf"] for row in wire_map for alias in (row["driver"], row["urdf"]) if alias is not None}
    if any(name not in aliases for name in names):
        raise ContractError("unknown joint alias")
    canonical = [aliases[name] if aliases[name] is not None else name for name in names]
    return project_arm_state(canonical, points[0]["positions"], wire_map)


def accepted_open(request, response, *, fixed_mode):
    """Accepted service target is not measured joint state or successful object grasp."""
    if fixed_mode is None or request.get("mode") != fixed_mode:
        raise ContractError("grasp mode not fixed/verified for this session")
    if response.get("success") is not True:
        raise ContractError("grasp command was not accepted")
    value = response.get("open")
    if isinstance(value, bool) or not isinstance(value, float | int) or not np.isfinite(value) or not 0 <= value <= 1:
        raise ContractError("accepted open must be a finite fraction in [0,1]")
    return float(value)
