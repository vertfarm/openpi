"""Re-solving recorded arm joints for the 2026-10-20 arm, on synthetic arms and sessions."""

import json

import h5py
import numpy as np
import pytest

from examples.hv1 import native
from examples.hv1.artifacts import ContractError
from examples.hv1.artifacts import file_hash
from examples.hv1.kinematics import Chain
from examples.hv1.kinematics import _rpy
from examples.hv1.retarget import POSE_COLUMNS
from examples.hv1.retarget import RECORDED_POSE
from examples.hv1.retarget import retarget_session

GRASP = np.array([0.4, -0.1, 0.2, 0.9, 0.05, 0.3, 0.2], dtype=np.float32)
LIMITS = ([-1.5] * 7, [1.5] * 7, "profile-sha")
# The recorder logs its controller's FK at a TCP of its own, not the URDF tool link.
LOGGED_TCP = (_rpy(0.1, -0.2, 0.3), np.array([0.01, -0.02, -0.08]))


def _quaternion(r):
    w = np.sqrt(max(1.0 + np.trace(r), 1e-12)) / 2.0
    return np.array([(r[2, 1] - r[1, 2]) / (4 * w), (r[0, 2] - r[2, 0]) / (4 * w), (r[1, 0] - r[0, 1]) / (4 * w), w])


def recorded_session(tmp_path, source_session, recorded_on, *, elbow=0.9):
    """Two episodes as the recorder writes them on the `recorded_on` arm, logged tool pose included."""
    chain = Chain.from_urdf(recorded_on, "Right_tool0")
    root = source_session(tmp_path / "raw" / "keti_humanoid_data_260909", "keti_humanoid_data_260909", ["e0", "e1"])
    for index, episode in enumerate(("e0", "e1")):
        with h5py.File(root / "episodes" / episode / "data.hdf5", "r+") as f:
            frames = f["observation.state.upper_body.joint"].shape[0]
            pose = np.tile(GRASP, (frames, 1)) + 0.02 * np.arange(frames, dtype=np.float32)[:, None] + 0.02 * index
            pose[:, 3] = elbow + 0.02 * np.arange(frames)
            pose[:, 6] += 0.1 * np.arange(frames)
            f["observation.state.upper_body.joint"][:] = pose
            f["action.upper_body.joint"][:] = pose + 0.005
            logged = []
            for q in pose.astype(np.float64):
                r, p = chain.fk(q)
                logged.append(np.r_[p + r @ LOGGED_TCP[1], _quaternion(r @ LOGGED_TCP[0])])
            f.create_dataset(RECORDED_POSE, data=np.array(logged, dtype=np.float32))
            labels = json.loads(f["meta"].attrs["labels"])
            f["meta"].attrs["labels"] = json.dumps({**labels, RECORDED_POSE: POSE_COLUMNS})
    return root


def test_retarget_keeps_every_recorded_tool_pose_on_the_new_arm(tmp_path, source_session, arm_urdf):
    old, new = arm_urdf("old"), arm_urdf("new", elbow_x=0.03, wrist_x=-0.03)
    source = recorded_session(tmp_path, source_session, old)
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
    assert report["recorded_pose_match"]["median_mm"] < 0.01
    assert set(report["recorded_pose_match"]["episode_median_mm"]) == {"e0", "e1"}
    native.read_numeric(output / "episodes" / "e0" / "data.hdf5")


def test_retarget_refuses_to_overwrite_and_reports_unreachable_rows(tmp_path, source_session, arm_urdf):
    old, new = arm_urdf("old", limit=3.0), arm_urdf("new", elbow_x=0.03, wrist_x=-0.03, limit=3.0)
    source = recorded_session(tmp_path, source_session, old, elbow=2.5)
    output = tmp_path / "kh261020" / "keti_humanoid_data_260909"
    with pytest.raises(ContractError, match="unreachable"):
        retarget_session(source, output, old, new, ([-1.0] * 7, [1.0] * 7))
    assert not output.exists() and not any(output.parent.glob(".*partial"))
    retarget_session(source, output, old, new, ([-3.0] * 7, [3.0] * 7))
    with pytest.raises(ContractError, match="exists"):
        retarget_session(source, output, old, new, ([-3.0] * 7, [3.0] * 7))


def test_retarget_refuses_a_source_arm_the_recording_was_not_made_on(tmp_path, source_session, arm_urdf):
    """The 2026-09-09/10 sessions predate the 2026-09-30 wrist remount; today's
    controller URDF carries the remounted wrist. Re-solving from it would teach
    the wrist to bend the wrong way, so the logged tool pose must name the arm."""
    recorded_on, remounted = arm_urdf("old"), arm_urdf("remounted", flip_wrist=True)
    source = recorded_session(tmp_path, source_session, recorded_on)
    output = tmp_path / "kh261020" / "keti_humanoid_data_260909"
    new = arm_urdf("new", elbow_x=0.03, wrist_x=-0.03)
    with pytest.raises(ContractError, match="the --from-urdf arm does not reproduce the recorded tool pose"):
        retarget_session(source, output, remounted, new, LIMITS)
    assert not output.exists()
    with h5py.File(source / "episodes" / "e1" / "data.hdf5", "r+") as f:
        del f[RECORDED_POSE]
    with pytest.raises(ContractError, match=f"e1: {RECORDED_POSE} lacks"):
        retarget_session(source, output, recorded_on, new, LIMITS)


def test_source_arm_check_is_one_tool_fit_over_the_session_and_needs_motion(tmp_path, source_session, arm_urdf):
    """Per-episode fits would let six free parameters absorb a wrong wrist in an
    episode where the arm hardly moves; the whole session shares one fit, every
    joint must move, and padded rows before the first pose message are skipped."""
    from examples.hv1.retarget import check_source_arm

    recorded_on, remounted = arm_urdf("old"), arm_urdf("remounted", flip_wrist=True)
    source = recorded_session(tmp_path, source_session, recorded_on)
    paths = {e: source / "episodes" / e / "data.hdf5" for e in ("e0", "e1")}
    with h5py.File(paths["e0"], "r+") as f:
        f[RECORDED_POSE][0] = [0, 0, 0, 0, 0, 0, 1]  # padded before the first pose message
    chain = Chain.from_urdf(recorded_on, "Right_tool0")
    assert check_source_arm(chain, paths)["median_mm"] < 0.01
    with pytest.raises(ContractError, match="does not reproduce"):
        check_source_arm(Chain.from_urdf(remounted, "Right_tool0"), paths)
    for path in paths.values():
        with h5py.File(path, "r+") as f:
            q = f["observation.state.upper_body.joint"][:]
            f["observation.state.upper_body.joint"][:] = q[:1]
    with pytest.raises(ContractError, match="cannot verify"):
        check_source_arm(chain, paths)


def test_retarget_refuses_a_recording_without_the_right_arm_columns(tmp_path, source_session, arm_urdf):
    """A label scheme it does not know must stop the run, not pass the old
    angles through as if they had been re-solved."""
    source = recorded_session(tmp_path, source_session, arm_urdf("old"))
    with h5py.File(source / "episodes" / "e1" / "data.hdf5", "r+") as f:
        labels = json.loads(f["meta"].attrs["labels"])
        labels["action.upper_body.joint"] = [f"points/positions/arm_x_joint{i}" for i in range(1, 8)]
        f["meta"].attrs["labels"] = json.dumps(labels)
    output = tmp_path / "kh261020" / "keti_humanoid_data_260909"
    with pytest.raises(ContractError, match="e1: action.upper_body.joint lacks"):
        retarget_session(source, output, arm_urdf("old"), arm_urdf("new", elbow_x=0.03, wrist_x=-0.03), LIMITS)
    assert not output.exists()


def test_retarget_only_the_listed_episodes_and_drop_the_rest(tmp_path, source_session, arm_urdf):
    """A session holds episodes no campaign uses (2026-09-09 episode_000000 sits
    at the old elbow limit and is out of reach on the new arm). Re-solve the
    listed ones; drop the others from the output instead of leaving old-arm rows
    in a "retargeted" session. metadata.json stays whole: `prepare` checks the
    inventory and scans only the approved episodes."""
    source = recorded_session(tmp_path, source_session, arm_urdf("old", limit=3.0))
    with h5py.File(source / "episodes" / "e0" / "data.hdf5", "r+") as f:
        f["observation.state.upper_body.joint"][:, 3] = 2.5
    output = tmp_path / "kh261020" / "keti_humanoid_data_260909"
    report = retarget_session(
        source,
        output,
        arm_urdf("old", limit=3.0),
        arm_urdf("new", elbow_x=0.03, wrist_x=-0.03),
        LIMITS,
        episodes={"e1"},
    )
    assert sorted(report["episodes"]) == ["e1"] and report["omitted"] == ["e0"]
    assert not (output / "episodes" / "e0").exists() and (output / "episodes" / "e1" / "data.hdf5").is_file()
    assert (output / "metadata.json").read_bytes() == (source / "metadata.json").read_bytes()
    with pytest.raises(ContractError, match="not in the session"):
        retarget_session(
            source,
            tmp_path / "other" / "keti_humanoid_data_260909",
            arm_urdf("old"),
            arm_urdf("new"),
            LIMITS,
            episodes={"e9"},
        )


def test_campaign_episodes_are_the_ones_its_manifest_names_for_this_session(tmp_path):
    from examples.hv1.retarget import campaign_episodes

    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "episodes": [
                    {"session_id": "keti_humanoid_data_260909", "episode_id": "episode_000003"},
                    {"session_id": "keti_humanoid_data_260910", "episode_id": "episode_000000"},
                ]
            }
        ),
        encoding="utf-8",
    )
    assert campaign_episodes(manifest, "keti_humanoid_data_260909") == {"episode_000003"}
    with pytest.raises(ContractError, match="no episodes"):
        campaign_episodes(manifest, "keti_humanoid_data_999999")
