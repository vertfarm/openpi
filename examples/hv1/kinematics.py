"""Forward and inverse kinematics of one URDF arm, and moving joint angles between arms.

The 2026-10-20 upper body moved the elbow pitch origin forward and the wrist
roll origin back by the same amount. The two cancel with the elbow straight and
not otherwise, so the recorded angles put the new arm's tool a median 28 mm
away at a grasp. Policies trained on the earlier arm speak its joint angles;
`Remap` carries them to the new arm by keeping the tool pose, which is what the
cameras and the object see. Numpy only; the URDFs are read from disk, never kept
in this repository.
"""

from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .artifacts import ContractError
from .artifacts import file_hash

POSITION_TOLERANCE_M = 1e-6
ROTATION_TOLERANCE_RAD = 1e-5
MAX_IK_STEP_RAD = 0.2
# A remapped row may move at most this many times the source row plus slack.
# The two arms differ by a few centimetres, so their Jacobians are nearly equal;
# a larger jump is a solver excursion (near a singularity) and is refused.
STEP_GAIN, STEP_SLACK_RAD = 2.0, 0.01


def _axis_rotation(axis, angle):
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


def _rpy(roll, pitch, yaw):
    x, y, z = np.eye(3)
    return _axis_rotation(z, yaw) @ _axis_rotation(y, pitch) @ _axis_rotation(x, roll)


def _rotation_error(target, current):
    """Axis-angle vector that turns `current` into `target`, in the base frame."""
    r = target @ current.T
    vee = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    angle = np.arccos(np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0))
    return 0.5 * vee if angle < 1e-9 else angle / (2.0 * np.sin(angle)) * vee


def _floats(text, count, what):
    values = [float(v) for v in text.split()]
    if len(values) != count or not np.isfinite(values).all():
        raise ContractError(f"URDF {what} needs {count} finite numbers")
    return np.array(values)


class Chain:
    """The serial chain from the URDF root to `tip`; revolute and fixed joints only."""

    def __init__(self, segments, tip, lower=None, upper=None):
        self.segments, self.tip = segments, tip
        moving = [s for s in segments if s["type"] == "revolute"]
        self.names = tuple(s["name"] for s in moving)
        self.lower = np.array([s["lower"] for s in moving]) if lower is None else lower
        self.upper = np.array([s["upper"] for s in moving]) if upper is None else upper

    @classmethod
    def from_urdf(cls, path, tip):
        by_child = {}
        for joint in ET.parse(Path(path)).getroot().findall("joint"):
            origin, axis, limit = joint.find("origin"), joint.find("axis"), joint.find("limit")
            xyz = _floats(origin.get("xyz", "0 0 0") if origin is not None else "0 0 0", 3, "origin xyz")
            rpy = _floats(origin.get("rpy", "0 0 0") if origin is not None else "0 0 0", 3, "origin rpy")
            segment = dict(
                name=joint.get("name"),
                type=joint.get("type"),
                parent=joint.find("parent").get("link"),
                xyz=xyz,
                rotation=_rpy(*rpy),
                mimic=joint.find("mimic") is not None,
            )
            if segment["type"] == "revolute":
                if limit is None or limit.get("lower") is None or limit.get("upper") is None:
                    raise ContractError(f"revolute joint {segment['name']} has no limit")
                direction = _floats(axis.get("xyz") if axis is not None else "1 0 0", 3, "axis")
                segment.update(
                    axis=direction / np.linalg.norm(direction),
                    lower=float(limit.get("lower")),
                    upper=float(limit.get("upper")),
                )
            by_child[joint.find("child").get("link")] = segment
        if tip not in by_child:
            raise ContractError(f"URDF has no joint into the tool link {tip}")
        segments, link = [], tip
        while link in by_child:
            segment = by_child[link]
            if segment["type"] not in ("revolute", "fixed") or segment["mimic"]:
                raise ContractError(f"unsupported joint on the {tip} chain: {segment['name']}")
            segments.append(segment)
            link = segment["parent"]
        return cls(segments[::-1], tip)

    def with_limits(self, lower, upper):
        """The same chain under other joint limits, e.g. the field-measured ones.

        URDF limits on this arm are CAD placeholders (+-90 deg). The recorded start
        pose needs about 1.62 rad of elbow on the 2026-10-20 arm, so remapping it
        depends on the real limit, not the placeholder.
        """
        lower, upper = np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)
        if (
            lower.shape != self.lower.shape
            or upper.shape != self.upper.shape
            or not (np.isfinite(lower).all() and np.isfinite(upper).all())
            or np.any(lower >= upper)
        ):
            raise ContractError("joint limits must be one finite lower < upper pair per joint")
        return Chain(self.segments, self.tip, lower, upper)

    def _walk(self, q):
        """Tool rotation, tool position and the 6 x n Jacobian at `q`, in one pass."""
        q = np.asarray(q, dtype=np.float64)
        if q.shape != (len(self.names),) or not np.isfinite(q).all():
            raise ContractError(f"expected {len(self.names)} finite joint angles")
        rotation, position, axes, points, index = np.eye(3), np.zeros(3), [], [], 0
        for segment in self.segments:
            position = position + rotation @ segment["xyz"]
            rotation = rotation @ segment["rotation"]
            if segment["type"] == "revolute":
                axes.append(rotation @ segment["axis"])
                points.append(position)
                rotation = rotation @ _axis_rotation(segment["axis"], q[index])
                index += 1
        jacobian = np.array([np.r_[np.cross(a, position - p), a] for a, p in zip(axes, points)]).T
        return rotation, position, jacobian

    def fk(self, q):
        """Tool rotation and position in the root frame."""
        rotation, position, _ = self._walk(q)
        return rotation, position

    def jacobian(self, q):
        return self._walk(q)[2]

    def ik(self, rotation, position, seed, *, iterations=200, damping=1e-3):
        """Joint angles inside the limits that reach the pose, found from `seed`.

        Damped least squares, so a redundant arm moves as little as it can from
        the seed. A joint held at a limit by the step is taken out of that step
        (active set) instead of being clipped back after it, which would stall
        the solve right where the start pose lives. Steps are capped, and a
        solve that stops improving refuses early. Never returns a near miss.
        """
        q = np.clip(np.asarray(seed, dtype=np.float64), self.lower, self.upper)
        best, stalled = np.inf, 0
        for _ in range(iterations):
            current_r, current_p, j = self._walk(q)
            error = np.r_[position - current_p, _rotation_error(rotation, current_r)]
            if np.linalg.norm(error[:3]) < POSITION_TOLERANCE_M and np.linalg.norm(error[3:]) < ROTATION_TOLERANCE_RAD:
                return q
            size = np.linalg.norm(error)
            best, stalled = (size, 0) if size < 0.999 * best else (best, stalled + 1)
            if stalled >= 20:
                break
            free = np.ones(len(q), dtype=bool)
            for _ in range(len(q)):
                jf = j[:, free]
                step = np.zeros(len(q))
                step[free] = jf.T @ np.linalg.solve(jf @ jf.T + damping**2 * np.eye(6), error)
                pinned = free & (((q <= self.lower) & (step < 0)) | ((q >= self.upper) & (step > 0)))
                if not pinned.any():
                    break
                free &= ~pinned
            largest = np.max(np.abs(step))
            if largest > MAX_IK_STEP_RAD:
                step *= MAX_IK_STEP_RAD / largest
            q = np.clip(q + step, self.lower, self.upper)
        raise ContractError(
            f"IK did not reach the pose within the joint limits ({1000 * np.linalg.norm(error[:3]):.2f} mm, "
            f"{np.degrees(np.linalg.norm(error[3:])):.3f} deg)"
        )


def check_step(source_step, target_step):
    """Refuse a remapped row that jumps further than the row it came from allows."""
    bound = STEP_GAIN * np.max(np.abs(source_step)) + STEP_SLACK_RAD
    jump = np.max(np.abs(target_step))
    if jump > bound:
        raise ContractError(f"remap is discontinuous: {jump:.3f} rad step for a {bound:.3f} rad bound")


class Remap:
    """Joint angles of one arm -> the angles that put another arm's tool in the same pose."""

    def __init__(self, source, target):
        if source.names != target.names or source.tip != target.tip:
            raise ContractError("remap needs the same joints and tool link on both arms")
        self.source, self.target = source, target

    def convert(self, q, seed):
        return self.target.ik(*self.source.fk(q), seed=seed)

    def trajectory(self, rows, seed, source_seed=None):
        """Convert consecutive rows, each seeded from the previous answer.

        `seed` is the target-arm angles the first row starts from and
        `source_seed` the source-arm angles they correspond to (the first row
        itself when not given). Every row's step is checked against its source
        step, so a solver excursion is refused rather than passed on.
        """
        rows = np.asarray(rows, dtype=np.float64)
        result = []
        previous_in = rows[0] if source_seed is None else np.asarray(source_seed, dtype=np.float64)
        previous_out = np.asarray(seed, dtype=np.float64)
        for row in rows:
            answer = self.convert(row, previous_out)
            if source_seed is not None or result:
                check_step(row - previous_in, answer - previous_out)
            result.append(answer)
            previous_in, previous_out = row, answer
        return np.array(result)


def field_limits(profile_path):
    """The right arm's measured `joint_min`/`joint_max` from an approved field profile, and its hash."""
    profile = json.loads(Path(profile_path).read_text(encoding="utf-8"))
    if profile.get("approved") is not True or not profile.get("review_id"):
        raise ContractError("joint limits come only from an approved, reviewed field profile")
    low, high = profile.get("joint_min"), profile.get("joint_max")
    if not isinstance(low, list) or not isinstance(high, list) or len(low) != 7 or len(high) != 7:
        raise ContractError("field profile needs 7-joint joint_min and joint_max")
    low, high = np.asarray(low, dtype=np.float64), np.asarray(high, dtype=np.float64)
    if not (np.isfinite(low).all() and np.isfinite(high).all()) or np.any(low >= high):
        raise ContractError("field profile joint limits must be finite with min < max")
    return low, high, file_hash(profile_path)
