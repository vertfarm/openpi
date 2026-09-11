"""The source-format contract every campaign reads through.

These were only ever exercised through the retired readapt campaign, so they
moved here when it went: `native` describes the recording, not the experiment,
and both `two_track.prepare` and `two_track.export` depend on it refusing a
recording it cannot faithfully represent.
"""

import json

import h5py
import numpy as np
import pytest

from examples.hv1 import native
from examples.hv1.artifacts import ContractError


def test_the_recorded_contract_a_trained_model_depends_on():
    """Changing any of these silently retrains against a different robot."""
    profile = native.profile()
    assert len(profile["state"]["names"]) == 15 and len(profile["action"]["names"]) == 8
    assert profile["action_horizon"] == 15 and profile["fps"] == 30
    assert profile["gripper"]["mode"] == 2 and profile["gripper"]["release_open"] == 0.6
    assert profile["camera_dropout"] is False
    # The arm is a delta against the measured state; the grasp intent is not.
    assert profile["action"]["delta_state_indices"] == [0, 1, 2, 3, 4, 5, 6, -1]
    assert profile["action"]["names"][7] == "right_grasp_intent"
    assert set(profile["images"]) == set(native.CAMERAS)


def test_numeric_rows_become_state_action_and_grasp_events(tmp_path, source_episode):
    directory = source_episode(tmp_path / "episode_000000", frames=10, cycles=((3, 7),))
    state, action, info = native.read_numeric(directory / "data.hdf5")
    assert state.shape == (10, 15) and action.shape == (10, 8)
    np.testing.assert_allclose(action[:, :7], 0.1)
    assert action[:, 7].tolist() == [0, 0, 0, 1, 1, 1, 1, 0, 0, 0]
    assert info["grasp_frames"] == [3] and info["release_frames"] == [7]
    assert info["grasp_frame"] == 3 and info["release_frame"] == 7
    assert info["frames"] == 10 and info["stale_rows"] == 0
    assert info["max_command_step_rad"] == 0.0


def test_a_second_grasp_cycle_needs_explicit_permission(tmp_path, source_episode):
    """The readapt collection allowed one cycle; two-track recovery episodes
    carry a regrasp, which is why the caller has to say so."""
    directory = source_episode(tmp_path / "episode_000001", frames=12, cycles=((2, 4), (7, 10)))
    with pytest.raises(ContractError, match="one open/close/release cycle"):
        native.read_numeric(directory / "data.hdf5")
    _, _, info = native.read_numeric(directory / "data.hdf5", allow_multiple_cycles=True)
    assert info["grasp_frames"] == [2, 7] and info["release_frames"] == [4, 10]
    assert info["grasp_frame"] == 2 and info["release_frame"] == 10


@pytest.mark.parametrize("cycles", [((0, 4),), ((2, 6),)])
def test_a_recording_that_starts_or_ends_closed_is_refused(tmp_path, source_episode, cycles):
    directory = source_episode(tmp_path / "episode_000002", frames=6, cycles=cycles)
    with pytest.raises(ContractError, match="open-start/open-end"):
        native.read_numeric(directory / "data.hdf5")


def test_stale_rows_require_review(tmp_path, source_episode):
    directory = source_episode(tmp_path / "episode_000003", stale=True)
    with pytest.raises(ContractError, match="stale recorder rows"):
        native.read_numeric(directory / "data.hdf5")


@pytest.mark.parametrize(
    "damage,message",
    [
        ("repeated_row", "invalid recorder row timing"),
        ("long_gap", "invalid recorder row timing"),
        ("half_open_hand", "mode2/0/0.6 contract"),
        ("dropped_label", "label/width mismatch"),
        ("miscounted_rows", "numeric length mismatch"),
    ],
)
def test_a_recording_that_does_not_describe_itself_is_refused(tmp_path, source_episode, damage, message):
    directory = source_episode(tmp_path / "episode_000004", frames=8, cycles=((3, 6),))
    with h5py.File(directory / "data.hdf5", "r+") as handle:
        if damage == "repeated_row":
            handle["timestamp"][4] = handle["timestamp"][3]
        elif damage == "long_gap":
            handle["timestamp"][4:] = handle["timestamp"][4:] + 1.0
        elif damage == "half_open_hand":
            handle["action.hand.command_r"][4, 1] = 0.3
        elif damage == "dropped_label":
            labels = json.loads(handle["meta"].attrs["labels"])
            labels["action.upper_body.joint"] = labels["action.upper_body.joint"][:-1]
            handle["meta"].attrs["labels"] = json.dumps(labels)
        else:
            handle["meta"].attrs["num_frames"] = 7
    with pytest.raises(ContractError, match=message):
        native.read_numeric(directory / "data.hdf5")


def test_source_custody_covers_every_file_the_export_reads(tmp_path, source_episode):
    directory = source_episode(tmp_path / "episode_000005")
    names = [path.name for path in native.source_files(directory)]
    assert names == ["data.hdf5", "tasks.json", "head.mp4", "hand_l.mp4", "hand_r.mp4"]
    assert all(path.is_file() for path in native.source_files(directory))


def test_video_must_have_exactly_the_numeric_rows(tmp_path, source_episode):
    pytest.importorskip("av")
    directory = source_episode(tmp_path / "episode_000006", frames=6, video=True)
    frames = list(native.decode_episode(directory, 6))
    assert len(frames) == 6
    assert all(image.shape == (480, 640, 3) for image in frames[0].values())
    assert set(frames[0]) == set(native.CAMERAS)
    with pytest.raises(ContractError, match="video contains extra rows"):
        list(native.decode_episode(directory, 5))
    with pytest.raises(ContractError, match="video ends before row"):
        list(native.decode_episode(directory, 7))
