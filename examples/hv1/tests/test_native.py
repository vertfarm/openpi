"""The source-format contract every campaign reads through.

These were only ever exercised through the retired readapt campaign, so they
moved here when it went: `native` describes the recording, not the experiment,
and both `pipeline.prepare` and `pipeline.export` depend on it refusing a
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


def _ramp(frames=12, joints=3):
    """A smooth, slowly moving arm series: no frame-to-frame step is near the spike threshold."""
    return np.linspace(0, 0.1, frames, dtype=np.float32)[:, None] * np.arange(1, joints + 1, dtype=np.float32)


def test_a_one_frame_ik_spike_is_replaced_by_the_line_between_its_neighbours():
    series = _ramp()
    expected = series.copy()
    series[5, 1] += 0.7
    cleaned, fixes = native.repair_ik_spikes(series)
    np.testing.assert_allclose(cleaned, expected, atol=1e-6)
    assert [(f["joint"], f["frames"]) for f in fixes] == [(1, [5])]
    assert fixes[0]["peak_rad"] == pytest.approx(0.7 + (expected[5, 1] - expected[4, 1]), abs=1e-5)
    assert native.find_ik_spikes(cleaned) == []


def test_a_two_frame_ik_spike_is_repaired_when_it_returns_within_the_window():
    series = _ramp()
    expected = series.copy()
    series[5:7, 0] += 0.5
    cleaned, fixes = native.repair_ik_spikes(series)
    np.testing.assert_allclose(cleaned, expected, atol=1e-6)
    assert [(f["joint"], f["frames"]) for f in fixes] == [(0, [5, 6])]


def test_a_jump_that_returns_by_exactly_half_still_counts_as_a_spike():
    series = np.zeros((8, 1), dtype=np.float32)
    series[3, 0], series[4, 0] = 0.4, 0.2
    assert [(f["joint"], f["frames"]) for f in native.find_ik_spikes(series)] == [(0, [3])]


@pytest.mark.parametrize(
    "excursion",
    [
        pytest.param([0.5, 0.5, 0.5], id="stays_three_frames"),
        pytest.param([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], id="step_that_never_returns"),
        pytest.param([0.14], id="below_threshold"),
        pytest.param([0.5, 0.45, 0.45], id="returns_less_than_half_in_window"),
    ],
)
def test_real_motion_is_not_mistaken_for_a_spike(excursion):
    series = np.zeros((10, 1), dtype=np.float32)
    series[3 : 3 + len(excursion), 0] = excursion
    cleaned, fixes = native.repair_ik_spikes(series)
    assert fixes == []
    np.testing.assert_array_equal(cleaned, series)


def test_spikes_are_found_per_joint_and_the_input_is_not_modified():
    series = _ramp()
    series[2, 0] += 0.4
    series[8, 2] -= 0.9
    before = series.copy()
    cleaned, fixes = native.repair_ik_spikes(series)
    np.testing.assert_array_equal(series, before)
    assert sorted((f["joint"], tuple(f["frames"])) for f in fixes) == [(0, (2,)), (2, (8,))]
    assert cleaned is not series


RIGHT_ARM_STATE = [
    "position/Right_Shoulder_Pitch_Joint",
    "position/Right_Shoulder_Roll_Joint",
    "position/Right_Shoulder_Yaw_Joint",
    "position/Right_Elbow_Pitch_Joint",
    "position/Right_Wrist_Roll_Joint",
    "position/Right_Wrist_Yaw_Joint",
    "position/Right_Wrist_Pitch_Joint",
]


def test_the_right_arm_state_names_are_exactly_what_models_were_trained_on():
    assert native.ARM_STATE == RIGHT_ARM_STATE
    assert native.arm_state("right") == RIGHT_ARM_STATE


def test_the_left_arm_mirrors_the_right_arm_joint_order():
    left = native.arm_state("left")
    assert left == [name.replace("Right_", "Left_") for name in RIGHT_ARM_STATE]
    assert left[0] == "position/Left_Shoulder_Pitch_Joint" and left[-1] == "position/Left_Wrist_Pitch_Joint"


def test_an_unknown_arm_side_is_refused():
    with pytest.raises(ContractError, match="side"):
        native.arm_state("head")


# The remaining two-arm pieces need names and numbers that only the 2026-10-27 recordings can confirm.
# They stay skipped until the numeric export has been read; nothing here is guessed.
@pytest.mark.skip(reason="needs the recorded action label names for the left arm")
def test_left_arm_action_names_come_from_the_recorded_labels():
    raise AssertionError("fill in from the recordings")


@pytest.mark.skip(reason="needs the recorded left-hand joint labels")
def test_left_hand_state_channels_come_from_the_recorded_labels():
    raise AssertionError("fill in from the recordings")


@pytest.mark.skip(reason="needs the recorded hand mode numbers for circular, tripod and pinch_2f")
def test_hand_modes_are_accepted_by_number_from_the_recordings():
    raise AssertionError("fill in from the recordings")


def test_a_spike_whose_return_would_fall_off_the_end_is_left_alone():
    series = np.zeros((6, 1), dtype=np.float32)
    series[5, 0] = 0.6
    assert native.find_ik_spikes(series) == []
