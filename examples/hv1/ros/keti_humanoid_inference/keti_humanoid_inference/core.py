"""ROS/GPU-independent deployment contract, scheduling and fail-closed gates."""

from __future__ import annotations

import base64
from collections import deque
import hashlib
import json
import math
from pathlib import Path
import time
import urllib.request

import numpy as np

PROMPT = "Pick up the silver cylindrical part from the table with the right hand and place it on the tray."
CAMERAS = ("head", "hand_l", "hand_r")
ARM_STATE = tuple(
    "Right_" + name + "_Joint"
    for name in (
        "Shoulder_Pitch",
        "Shoulder_Roll",
        "Shoulder_Yaw",
        "Elbow_Pitch",
        "Wrist_Roll",
        "Wrist_Yaw",
        "Wrist_Pitch",
    )
)
ARM_COMMAND = tuple(f"arm_r_joint{i}" for i in range(1, 8))
HAND_STATE = tuple(f"joint_{i}" for i in (10, 11, 12, 21, 22, 30, 31, 32))
CONTRACT = {
    "version": 1,
    "state_names": list(ARM_STATE + HAND_STATE),
    "action_names": list(ARM_COMMAND) + ["right_grasp_intent"],
    "arm_units": "rad",
    "action_space": "absolute_joint_target",
    "cameras": list(CAMERAS),
    "camera_dropout": False,
    "image_pipeline": "rotate0_cv2_area_640x480_RGB_pil_bilinear_pad224",
    "gripper": {"mode": 2, "wrap": True, "close": 1, "open": 0, "release_open": 0.6},
    "prompt": PROMPT,
    "hz": 30,
    "horizon": 15,
}
CONTRACT_SHA = hashlib.sha256(json.dumps(CONTRACT, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
CORE_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
HAND_ENVELOPE_MARGIN_RAD = 0.05


class Rejected(ValueError):
    """Reject the whole command; never silently clamp an arm target."""


def vector(value, width):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (width,) or not np.isfinite(result).all():
        raise Rejected(f"expected {width} finite values")
    return result


def named_positions(names, positions, required):
    if len(names) != len(set(names)):
        raise Rejected("duplicate joint name")
    values = vector(positions, len(names))
    indices = dict(zip(names, range(len(names))))
    if any(name not in indices for name in required):
        raise Rejected("missing joint name")
    return values[[indices[name] for name in required]]


def validate_metadata(meta):
    if meta.get("contract_sha256") != CONTRACT_SHA or meta.get("contract") != CONTRACT:
        raise Rejected("server contract mismatch")
    if meta.get("ready") is not True or meta.get("action_horizon") != 15:
        raise Rejected("server not ready / wrong horizon")
    if not meta.get("snapshot_sha256") or meta.get("denoise") != 10:
        raise Rejected("unqualified snapshot / sampling settings")
    if meta.get("adapter_core_sha256") != CORE_SOURCE_SHA256:
        raise Rejected("server/ROS adapter core source mismatch")
    envelope = meta.get("hand_state_envelope")
    if not isinstance(envelope, dict) or envelope.get("names") != list(HAND_STATE):
        raise Rejected("missing hand-state training envelope")
    q01, q99 = vector(envelope.get("q01"), 8), vector(envelope.get("q99"), 8)
    margin = float(envelope.get("margin_rad", float("nan")))
    if np.any(q01 > q99) or not math.isfinite(margin) or not 0 < margin <= 0.2:
        raise Rejected("invalid hand-state training envelope")


def validate_hand_envelope(hand, metadata):
    """Reject grossly out-of-distribution hand preparation without clipping it."""
    values = vector(hand, 8)
    envelope = metadata.get("hand_state_envelope", {})
    if envelope.get("names") != list(HAND_STATE):
        raise Rejected("hand-state envelope identity mismatch")
    q01, q99 = vector(envelope.get("q01"), 8), vector(envelope.get("q99"), 8)
    margin = float(envelope.get("margin_rad", float("nan")))
    if not math.isfinite(margin) or not 0 < margin <= 0.2:
        raise Rejected("invalid hand-state envelope margin")
    outside = np.flatnonzero((values < q01 - margin) | (values > q99 + margin))
    if len(outside):
        names = ",".join(HAND_STATE[index] for index in outside)
        raise Rejected(f"hand state outside training envelope: {names}")
    return values


def make_request(sequence, state, images):
    if set(images) != set(CAMERAS):
        raise Rejected("all three cameras required")
    packed = {}
    for name, payload in images.items():
        if not isinstance(payload, bytes) or not 0 < len(payload) <= 2_000_000:
            raise Rejected("invalid compressed image")
        packed[name] = base64.b64encode(payload).decode("ascii")
    return {
        "contract_sha256": CONTRACT_SHA,
        "sequence": sequence,
        "state": vector(state, 15).tolist(),
        "images": packed,
        "image_encoding": "ros_compressed",
        "prompt": PROMPT,
    }


class PolicyHTTP:
    """Bounded loopback-only transport, no retry or implicit rearming."""

    def __init__(self, port=8000, timeout=1.0):
        if not 0 < timeout <= 10 or not 1024 <= port <= 65535:
            raise Rejected("invalid transport configuration")
        self.url, self.timeout = f"http://127.0.0.1:{port}", timeout
        # Ignore user proxy environment for robot-host loopback traffic.
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _request(self, endpoint, payload=None):
        body = None if payload is None else json.dumps(payload, allow_nan=False).encode()
        req = urllib.request.Request(self.url + endpoint, data=body, headers={"Content-Type": "application/json"})
        with self.opener.open(req, timeout=self.timeout) as response:
            raw = response.read(4_000_001)
        if len(raw) > 4_000_000:
            raise Rejected("response too large")
        return json.loads(raw)

    def metadata(self):
        result = self._request("/health")
        validate_metadata(result)
        return result

    def infer(self, request, snapshot_sha):
        result = self._request("/infer", request)
        if (
            result.get("sequence") != request["sequence"]
            or result.get("contract_sha256") != CONTRACT_SHA
            or result.get("snapshot_sha256") != snapshot_sha
        ):
            raise Rejected("response identity mismatch")
        actions = np.asarray(result["actions"], dtype=np.float64)
        if actions.shape != (15, 8) or not np.isfinite(actions).all():
            raise Rejected("invalid action chunk")
        return actions, result


class Observations:
    """Receive ages AND publisher stamp ages; row alignment is not exposure sync."""

    def __init__(self, state_age=0.1, image_age=0.15, skew=0.1):
        self.entries = {}
        self.history = {}
        self.state_age, self.image_age, self.skew = state_age, image_age, skew

    def put(self, key, value, received, source_age):
        if not all(math.isfinite(x) for x in (received, source_age)) or source_age < -0.02:
            raise Rejected("invalid/future source timestamp")
        entry = (value, received, max(0.0, source_age))
        self.entries[key] = entry
        self.history.setdefault(key, deque(maxlen=16 if key in CAMERAS else 64)).append(entry)

    def snapshot(self, now):
        keys = ("arm", "hand") + CAMERAS
        missing = [k for k in keys if k not in self.entries]
        if missing:
            raise Rejected("missing: " + ",".join(missing))
        # Never substitute history for a stale/missing newest source. The
        # buffer only aligns otherwise-fresh asynchronous messages.
        for key in keys:
            _, received, initial_age = self.entries[key]
            age = now - received + initial_age
            maximum = self.image_age if key in CAMERAS else self.state_age
            if not 0 <= age <= maximum:
                raise Rejected(f"stale {key}: {age:.3f}s")
        cutoff = min(self.entries[k][1] - self.entries[k][2] for k in keys) + self.skew
        chosen, ages, times = {}, {}, []
        for key in keys:
            maximum = self.image_age if key in CAMERAS else self.state_age
            candidates = [
                e for e in self.history[key] if e[1] - e[2] <= cutoff + 1e-9 and 0 <= now - e[1] + e[2] <= maximum
            ]
            if not candidates:
                raise Rejected("no aligned fresh " + key)
            chosen[key] = max(candidates, key=lambda e: e[1] - e[2])
            ages[key] = now - chosen[key][1] + chosen[key][2]
            times.append(chosen[key][1] - chosen[key][2])
        if max(times) - min(times) > self.skew + 1e-9:
            raise Rejected("observation skew")
        state = np.r_[vector(chosen["arm"][0], 7), vector(chosen["hand"][0], 8)]
        images = {k: chosen[k][0] for k in CAMERAS}
        # Model targets are anchored to the arm observation, not HTTP return time.
        anchor = chosen["arm"][1] - chosen["arm"][2]
        return state, images, anchor, ages


class ChunkQueue:
    """Time-align full predictions; execute three future samples per replan."""

    period = 1.0 / 30
    prefix = 3

    def __init__(self):
        self.queue = deque()
        self.last_sent = None
        self.audit = {}
        self.last_meta = None

    def clear(self):
        self.queue.clear()
        self.last_sent = None
        self.audit.clear()
        self.last_meta = None

    def offer(self, actions, anchor, now, *, sequence=None, lead_s=0.002):
        a = np.asarray(actions, dtype=np.float64)
        if a.shape != (15, 8) or not np.isfinite(a).all():
            raise Rejected("invalid chunk")
        if not all(math.isfinite(t) for t in (anchor, now)) or not 0 <= now - anchor <= 0.2:
            raise Rejected("late/future inference anchor")
        earliest = max(now + lead_s, self.queue[-1][0] + self.period if self.queue else now + lead_s)
        # First arrival establishes the execution grid. Later chunks stay on it.
        due = earliest
        offset = (due - anchor) / self.period
        start = max(0, int(math.floor(offset + 1e-9) if self.queue else math.ceil(offset - 1e-9)))
        if start + self.prefix > len(a):
            raise Rejected("no unexpired action prefix")
        if not self.queue:
            due = anchor + start * self.period
        targets = [{"due": due + i * self.period, "action": a[start + i].tolist()} for i in range(self.prefix)]
        proposal = {"sequence": sequence, "anchor": anchor, "targets": targets}
        proposal_id = hashlib.sha256(json.dumps(proposal, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        for i, target in enumerate(targets):
            self.queue.append((target["due"], a[start + i].copy()))
            self.audit[target["due"]] = {
                "sequence": sequence,
                "model_index": start + i,
                "anchor": anchor,
                "proposal_id": proposal_id,
            }
        return {
            "first_model_index": start,
            "apply_at": due,
            "skipped": start,
            "proposal_id": proposal_id,
            "proposal": proposal,
        }

    def pop(self, now):
        if not self.queue:
            return None
        due, action = self.queue[0]
        if now < due:
            return None
        if now - due >= self.period:
            self.clear()
            raise Rejected("scheduler missed a sample; no catch-up burst")
        if self.last_sent is not None and now - self.last_sent < self.period * 0.5:
            raise Rejected("command burst")
        self.queue.popleft()
        self.last_sent = now
        self.last_meta = self.audit.pop(due, None)
        return due, action


class IntentConditioner:
    """Hysteresis and dwell filtering for the scalar grasp-intent target stream."""

    def __init__(self, *, close_threshold=0.7, open_threshold=0.3, min_hold_s=0.2):
        values = (float(open_threshold), float(close_threshold), float(min_hold_s))
        if not all(math.isfinite(value) for value in values) or not 0 <= values[0] < values[1] <= 1:
            raise Rejected("invalid gripper hysteresis")
        if not 0 <= values[2] <= 2:
            raise Rejected("invalid gripper dwell")
        self.open_threshold, self.close_threshold, self.min_hold_s = values
        self.closed = False
        self.candidate = None
        self.candidate_since = None
        self.last_time = None

    def update(self, value, now):
        value, now = float(value), float(now)
        if not math.isfinite(value) or not math.isfinite(now):
            raise Rejected("nonfinite gripper intent/time")
        if self.last_time is not None and now < self.last_time:
            raise Rejected("gripper time moved backwards")
        self.last_time = now
        desired = None
        if not self.closed and value >= self.close_threshold:
            desired = True
        elif self.closed and value <= self.open_threshold:
            desired = False
        if desired is None:
            self.candidate = self.candidate_since = None
            return None
        if desired != self.candidate:
            self.candidate, self.candidate_since = desired, now
        if now - self.candidate_since + 1e-12 < self.min_hold_s:
            return None
        self.closed = desired
        self.candidate = self.candidate_since = None
        return "close" if desired else "open"


class GripEdges:
    """Condition one pilot open-close-release cycle, never hand joint targets."""

    def __init__(self, **conditioner):
        self.conditioner = IntentConditioner(**conditioner)
        self.phase = 0

    def update(self, value, now=None):
        event = self.conditioner.update(value, time.monotonic() if now is None else now)
        if event == "close":
            if self.phase == 0:
                self.phase = 1
                return event
            raise Rejected("second grasp cycle requires operator reset")
        if event == "open":
            if self.phase == 1:
                self.phase = 2
                return event
            raise Rejected("release requested before grasp intent")
        return None


def qualify_proposal(payload, joint_min, joint_max, max_step):
    """Return a proposal's id if it may be executed, else None.

    The producer side of `LiveGate.check`'s `approved_proposals`: a guardian
    calls this, and the executor will only move on ids it returns.

    The id is re-derived from the contents rather than trusted. Approving the
    id the executor supplied would approve whatever it later chooses to run
    under that name, which is the one thing an independent check must not do.
    """
    try:
        proposal = payload["proposal"]
        encoded = json.dumps(proposal, sort_keys=True, separators=(",", ":")).encode()
        proposal_id = hashlib.sha256(encoded).hexdigest()
        if proposal_id != payload.get("proposal_id"):
            return None
        previous = None
        for target in proposal["targets"]:
            action = vector(target["action"][:7], 7)
            if np.any(action < joint_min) or np.any(action > joint_max):
                return None
            if previous is not None and np.any(np.abs(action - previous) > max_step):
                return None
            previous = action
    except (ValueError, KeyError, TypeError, IndexError, Rejected):
        return None
    return proposal_id


class LiveGate:
    """Physical limits are required inputs, never inferred from demonstrations."""

    def __init__(self, profile):
        self.p = profile
        if profile.get("approved") is not True or not profile.get("review_id"):
            raise Rejected("field qualification missing")
        if profile.get("contract_sha256") != CONTRACT_SHA:
            raise Rejected("field profile contract mismatch")
        for name in (
            "joint_min",
            "joint_max",
            "max_step",
            "max_tracking_error",
            "max_velocity",
            "max_acceleration",
            "start_q",
            "start_tolerance",
        ):
            vector(profile.get(name), 7)
        for name in ("max_step", "max_tracking_error", "max_velocity", "max_acceleration", "start_tolerance"):
            if np.any(np.asarray(profile[name]) <= 0):
                raise Rejected("positive measured bounds required")
        if np.any(np.asarray(profile["joint_min"]) >= profile["joint_max"]):
            raise Rejected("joint range invalid")
        for name in ("stop_service", "guardian_node", "snapshot_sha256", "qualification_evidence"):
            if not profile.get(name):
                raise Rejected("stop/ownership/model qualification missing")
        self.last_q = self.last_v = self.last_t = None
        self.armed = False

    def guardian(self, status, received, now):
        if not 0 <= now - received <= 0.1:
            raise Rejected("guardian lease expired")
        for k in ("exclusive", "workspace_clear", "stop_ready", "hardware_watchdog_ready"):
            if status.get(k) is not True:
                raise Rejected("guardian denies " + k)
        if status.get("review_id") != self.p["review_id"] or status.get("grasp_mode") != 2:
            raise Rejected("guardian profile/mode mismatch")
        age = status.get("mqtt_age_s", float("inf"))
        if not isinstance(age, (int, float)) or not math.isfinite(age) or not 0 <= age <= 0.1:
            raise Rejected("raw MQTT feedback stale")

    def arm(self, q, status, received, now):
        self.guardian(status, received, now)
        if status.get("start_pose_verified") is not True or status.get("narrow_open_verified") is not True:
            raise Rejected("left/head/start pose and narrow preparation not verified")
        q = vector(q, 7)
        if np.any(np.abs(q - self.p["start_q"]) > self.p["start_tolerance"]):
            raise Rejected("outside qualified starting pose")
        self.last_q, self.last_v, self.last_t = q.copy(), np.zeros(7), now
        self.armed = True

    def check(self, target, measured, status, received, now, *, proposal_id=None):
        if not self.armed:
            raise Rejected("not locally armed")
        self.guardian(status, received, now)
        approvals = status.get("approved_proposals", [])
        if not proposal_id or not isinstance(approvals, list) or proposal_id not in approvals:
            raise Rejected("proposed trajectory not qualified by guardian")
        q, measured = vector(target, 7), vector(measured, 7)
        if np.any(q < self.p["joint_min"]) or np.any(q > self.p["joint_max"]):
            raise Rejected("joint limit")
        if np.any(np.abs(q - measured) > self.p["max_tracking_error"]):
            raise Rejected("tracking error")
        dt = now - self.last_t
        if not 0 < dt <= 0.2:
            raise Rejected("command continuity lost")
        step = q - self.last_q
        v = step / dt
        acceleration = (v - self.last_v) / dt
        if np.any(np.abs(step) > self.p["max_step"]):
            raise Rejected("target step")
        if np.any(np.abs(v) > self.p["max_velocity"]) or np.any(np.abs(acceleration) > self.p["max_acceleration"]):
            raise Rejected("velocity/acceleration")
        self.last_q, self.last_v, self.last_t = q.copy(), v, now

    def disarm(self):
        self.armed = False


class JsonLog:
    def __init__(self, path):
        self.file = open(path, "x", encoding="utf-8")

    def write(self, event, **values):
        self.file.write(
            json.dumps(
                {"event": event, "monotonic": time.monotonic(), **values}, allow_nan=False, separators=(",", ":")
            )
            + "\n"
        )
        self.file.flush()

    def close(self):
        self.file.close()
