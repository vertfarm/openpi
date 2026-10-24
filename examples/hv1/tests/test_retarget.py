"""Re-solving recorded arm joints for the 2026-10-20 arm, on synthetic arms and sessions."""

import json

import h5py
import numpy as np
import pytest

from examples.hv1 import native
from examples.hv1.artifacts import ContractError
from examples.hv1.artifacts import file_hash
from examples.hv1.kinematics import Chain
from examples.hv1.retarget import retarget_session

GRASP = np.array([0.4, -0.1, 0.2, 0.9, 0.05, 0.3, 0.2], dtype=np.float32)
LIMITS = ([-1.5] * 7, [1.5] * 7, "profile-sha")


def recorded_session(tmp_path, source_session, *, elbow=0.9):
    root = source_session(tmp_path / "raw" / "keti_humanoid_data_260909", "keti_humanoid_data_260909", ["e0", "e1"])
    for index, episode in enumerate(("e0", "e1")):
        with h5py.File(root / "episodes" / episode / "data.hdf5", "r+") as f:
            frames = f["observation.state.upper_body.joint"].shape[0]
            pose = np.tile(GRASP, (frames, 1)) + 0.01 * np.arange(frames, dtype=np.float32)[:, None] + 0.02 * index
            pose[:, 3] = elbow
            f["observation.state.upper_body.joint"][:] = pose
            f["action.upper_body.joint"][:] = pose + 0.005
    return root


def test_retarget_keeps_every_recorded_tool_pose_on_the_new_arm(tmp_path, source_session, arm_urdf):
    old, new = arm_urdf("old"), arm_urdf("new", elbow_x=0.03, wrist_x=-0.03)
    source = recorded_session(tmp_path, source_session)
    before = {p.relative_to(source): file_hash(p) for p in source.rglob("*") if p.is_file()}
    output = tmp_path / "kh261020" / "keti_humanoid_data_260909"
    report = retarget_session(source, output, old, new, LIMITS)

    assert {p.relative_to(source): file_hash(p) for p in source.rglob("*") if p.is_file()} == before
    old_chain, new_chain = Chain.from_urdf(old, "Right_tool0"), Chain.from_urdf(new, "Right_tool0")
    for episode in ("e0", "e1"):
        with (
            h5py.File(source / "episodes" / episode / "data.hdf5") as a,
            h5py.File(output / "episodes" / episode / "data.hdf5") as b,
        ):
            for key in ("observation.state.upper_body.joint", "action.upper_body.joint"):
                for q_old, q_new in zip(a[key][:], b[key][:]):
                    np.testing.assert_allclose(new_chain.fk(q_new)[1], old_chain.fk(q_old)[1], atol=1e-5)
                assert np.max(np.abs(a[key][:] - b[key][:])) > 1e-3
            for key in ("observation.state.hand.joint_r", "action.hand.command_r", "timestamp", "stale"):
                np.testing.assert_array_equal(a[key][:], b[key][:])
        for name in ("tasks.json", *(c + ".mp4" for c in native.CAMERAS)):
            assert file_hash(source / "episodes" / episode / name) == file_hash(output / "episodes" / episode / name)
    assert json.loads((output / "metadata.json").read_text()) == json.loads((source / "metadata.json").read_text())
    provenance = json.loads((output / "retarget.json").read_text())
    assert provenance["from_urdf_sha256"] == file_hash(old) and provenance["to_urdf_sha256"] == file_hash(new)
    assert provenance["limits_profile_sha256"] == "profile-sha"
    assert provenance == report and report["episodes"]["e0"]["max_tool_residual_mm"] < 0.01
    native.read_numeric(output / "episodes" / "e0" / "data.hdf5")


def test_retarget_refuses_to_overwrite_and_reports_unreachable_rows(tmp_path, source_session, arm_urdf):
    old, new = arm_urdf("old", limit=3.0), arm_urdf("new", elbow_x=0.03, wrist_x=-0.03, limit=3.0)
    source = recorded_session(tmp_path, source_session, elbow=2.5)
    output = tmp_path / "kh261020" / "keti_humanoid_data_260909"
    with pytest.raises(ContractError, match="unreachable"):
        retarget_session(source, output, old, new, ([-1.0] * 7, [1.0] * 7))
    assert not output.exists() and not any(output.parent.glob(".*partial"))
    retarget_session(source, output, old, new, ([-3.0] * 7, [3.0] * 7))
    with pytest.raises(ContractError, match="exists"):
        retarget_session(source, output, old, new, ([-3.0] * 7, [3.0] * 7))


def test_retarget_refuses_a_recording_without_the_right_arm_columns(tmp_path, source_session, arm_urdf):
    """A label scheme it does not know must stop the run, not pass the old
    angles through as if they had been re-solved."""
    source = recorded_session(tmp_path, source_session)
    with h5py.File(source / "episodes" / "e1" / "data.hdf5", "r+") as f:
        labels = json.loads(f["meta"].attrs["labels"])
        labels["action.upper_body.joint"] = [f"points/positions/arm_x_joint{i}" for i in range(1, 8)]
        f["meta"].attrs["labels"] = json.dumps(labels)
    output = tmp_path / "kh261020" / "keti_humanoid_data_260909"
    with pytest.raises(ContractError, match="e1: action.upper_body.joint lacks"):
        retarget_session(source, output, arm_urdf("old"), arm_urdf("new", elbow_x=0.03, wrist_x=-0.03), LIMITS)
    assert not output.exists()
