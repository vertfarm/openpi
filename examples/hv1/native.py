"""The KETI HDF5/HEVC source format: what one recording is, and how to read it.

A library, not a runnable step. The ingestion CLI and its `scan`/`export` pair
served the 2026-09-09 pilot and then the readapt campaign; `pipeline` supersedes
both and nothing called them, so they went on 2026-09-11. No ROS imports.
"""

from __future__ import annotations

from contextlib import ExitStack
import json

import h5py
import numpy as np

from .artifacts import ContractError
from .workflow import validate_profile

PROMPT = "Pick up the silver cylindrical part from the table with the right hand and place it on the tray."
CAMERAS = ("head", "hand_l", "hand_r")
ARM_STATE = [
    f"position/Right_{x}_Joint"
    for x in (
        "Shoulder_Pitch",
        "Shoulder_Roll",
        "Shoulder_Yaw",
        "Elbow_Pitch",
        "Wrist_Roll",
        "Wrist_Yaw",
        "Wrist_Pitch",
    )
]
ARM_ACTION = [f"points/positions/arm_r_joint{i}" for i in range(1, 8)]
HAND_STATE = [f"position/joint_{i}" for i in (10, 11, 12, 21, 22, 30, 31, 32)]


def profile():
    clock = "recorder_row_alignment_not_sensor_exposure_time"

    def stream(path):
        return dict(path=path, timestamps_ns="derived_recorder_row_ns", clock_domain=clock)

    p = dict(
        schema_version=1,
        status="confirmed",
        source_format="keti_hdf5_hevc_v1",
        profile_id="hv1_right_silver_cylinder_3view_v1",
        fps=30,
        action_horizon=15,
        max_skew_ms=100,
        max_gap_ms=100,
        clock_domain=clock,
        state=dict(
            **stream("derived_state"),
            names=[x.removeprefix("position/") for x in ARM_STATE + HAND_STATE],
            units=["rad"] * 15,
        ),
        action=dict(
            **stream("derived_action"),
            names=[f"arm_r_joint{i}" for i in range(1, 8)] + ["right_grasp_intent"],
            units=["rad"] * 7 + ["binary"],
            semantics="commanded_target",
            delta_state_indices=list(range(7)) + [-1],
        ),
        images={c: dict(**stream(c + ".mp4"), shape=[224, 224, 3], color_order="RGB") for c in CAMERAS},
        gripper=dict(
            mode=2,
            wrap=True,
            close_intent=1,
            release_intent=0,
            release_open=0.6,
            provenance="derived_from_fixed_operator_protocol_not_grasp_success",
        ),
        camera_dropout=False,
        robot_motion_authorized=False,
    )
    return validate_profile(p)


def select_columns(f, key, names, labels):
    declared = labels[key]
    if len(declared) != len(set(declared)) or f[key].shape[1] != len(declared):
        raise ContractError(f"{key}: label/width mismatch")
    return f[key][:][:, [declared.index(n) for n in names]].astype(np.float32)


def read_numeric(path, *, allow_multiple_cycles=False):
    with h5py.File(path, "r") as f:
        attrs = f["meta"].attrs
        labels = json.loads(attrs["labels"])
        q = select_columns(f, "observation.state.upper_body.joint", ARM_STATE, labels)
        hand = select_columns(f, "observation.state.hand.joint_r", HAND_STATE, labels)
        command = select_columns(f, "action.upper_body.joint", ARM_ACTION, labels)
        grip = select_columns(f, "action.hand.command_r", ["mode", "open"], labels)
        ts = f["timestamp"][:]
        if len(ts) < 2 or np.any(np.diff(ts) <= 0) or np.max(np.diff(ts)) > 0.1:
            raise ContractError("invalid recorder row timing")
        if not all(np.isfinite(x).all() for x in (q, hand, command, grip, ts)):
            raise ContractError("nonfinite training fields")
        if not np.allclose(grip[:, 0], 2) or not np.all(
            np.isclose(grip[:, 1], 0, atol=1e-4) | np.isclose(grip[:, 1], 0.6, atol=1e-4)
        ):
            raise ContractError("gripper outside confirmed mode2/0/0.6 contract")
        if np.any(f["stale"][:]):
            raise ContractError("stale recorder rows require explicit review")
        if not all(len(x) == len(ts) for x in (q, hand, command, grip)) or int(attrs["num_frames"]) != len(ts):
            raise ContractError("numeric length mismatch")
        state = np.concatenate((q, hand), axis=1)
        action = np.concatenate((command, np.isclose(grip[:, 1:2], 0, atol=1e-4).astype(np.float32)), axis=1)
        changes = np.r_[0, 1 + np.flatnonzero(np.diff(action[:, -1]) != 0)]
        intents = action[changes, -1].astype(int).tolist()
        if not intents or intents[0] != 0 or intents[-1] != 0 or any(a == b for a, b in zip(intents, intents[1:])):
            raise ContractError("expected an open-start/open-end alternating grasp sequence")
        if not allow_multiple_cycles and intents != [0, 1, 0]:
            raise ContractError("expected one open/close/release cycle")
        close_frames = [int(frame) for frame, intent in zip(changes, intents, strict=True) if intent == 1]
        release_frames = [int(frame) for frame, intent in zip(changes, intents, strict=True) if intent == 0][1:]
        if not close_frames or len(close_frames) != len(release_frames):
            raise ContractError("unpaired grasp/release events")
        info = dict(
            frames=len(ts),
            created_at=str(attrs["created_at"]),
            max_command_step_rad=float(np.max(np.abs(np.diff(command, axis=0)))),
            grasp_frame=close_frames[0],
            release_frame=release_frames[-1],
            grasp_frames=close_frames,
            release_frames=release_frames,
            gripper_events=[
                dict(frame=int(frame), intent=int(intent)) for frame, intent in zip(changes, intents, strict=True)
            ],
            stale_rows=0,
            row_timestamp_seconds=ts.tolist(),
            source_stamp_ns=f["stamp_ns"][:].tolist(),
        )
        return state, action, info


def decode_episode(directory, count):
    import av

    with ExitStack() as stack:
        videos = [stack.enter_context(av.open(str(directory / (c + ".mp4")))) for c in CAMERAS]
        streams = [iter(v.decode(video=0)) for v in videos]
        for i in range(count):
            try:
                frames = [next(s).to_ndarray(format="rgb24") for s in streams]
            except StopIteration as exc:
                raise ContractError(f"{directory.name}: video ends before row {i}") from exc
            if any(x.shape != (480, 640, 3) for x in frames):
                raise ContractError("unexpected source camera dimensions")
            yield dict(zip(CAMERAS, frames, strict=True))
        if any(next(s, None) is not None for s in streams):
            raise ContractError("video contains extra rows")


def source_files(directory):
    return [directory / "data.hdf5", directory / "tasks.json", *[directory / (c + ".mp4") for c in CAMERAS]]
