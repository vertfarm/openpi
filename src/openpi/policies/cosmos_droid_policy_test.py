import numpy as np
import pytest

from openpi.policies import cosmos_droid_policy
from openpi.policies import droid_policy
from openpi import transforms

HORIZON = cosmos_droid_policy.COSMOS_ACTION_HORIZON
SCALE = 0.2


def _cosmos_response(joint_targets: np.ndarray, gripper: np.ndarray | None = None) -> dict:
    """A Cosmos server response: absolute joint angles under the singular key."""
    horizon = joint_targets.shape[0]
    actions = np.zeros((horizon, 8), dtype=np.float32)
    actions[:, :7] = joint_targets
    actions[:, 7] = np.zeros(horizon) if gripper is None else gripper
    return {cosmos_droid_policy.COSMOS_ACTION_KEY: actions}


def _droid_client_reconstruct(
    state: np.ndarray, velocities: np.ndarray, *, scale: float = SCALE, tracking: float = 1.0
) -> np.ndarray:
    """What the robot ends up commanding, given the velocity chunk we send it.

    Mirrors ``RobotEnv(action_space="joint_velocity")``: each command is a normalized per-step
    delta, multiplied by ``scale`` and added to the **measured** joint position at that step.

    ``tracking`` is how much of the commanded motion the arm actually achieves in one step
    (1.0 = perfect). It exists to exercise the failure mode the deployed path's own docs warn
    about: "the server-side conversion cannot correct mid-chunk tracking error".
    """
    measured = np.asarray(state[:7], dtype=np.float64).copy()
    targets = []
    for velocity in velocities:
        target = measured + velocity[:7] * scale
        targets.append(target.copy())
        measured = measured + (target - measured) * tracking
    return np.stack(targets)


# --------------------------------------------------------------------------------------
# The contract that actually reaches the robot
# --------------------------------------------------------------------------------------


def test_round_trip_recovers_the_absolute_targets_exactly():
    """Convert to velocities, let the robot rebuild targets from them, get the input back.

    This is the deployment contract in one assertion. Everything else in this file is a way of
    failing it more specifically.
    """
    rng = np.random.default_rng(0)
    state = np.concatenate([rng.uniform(-0.5, 0.5, 7), [0.3]])
    # Small per-step motion, matching what Cosmos actually produces: the measured median joint
    # delta over 5 simulated runs was 0.0074 rad/step (0.037 in velocity units).
    steps = rng.uniform(-0.01, 0.01, size=(HORIZON, 7))
    targets = state[:7] + np.cumsum(steps, axis=0)

    result = cosmos_droid_policy.cosmos_velocity_output_chain()(
        {"state": state, **_cosmos_response(targets)}
    )

    assert result["actions"].shape == (HORIZON, 8)
    rebuilt = _droid_client_reconstruct(state, result["actions"])
    np.testing.assert_allclose(rebuilt, targets, atol=1e-6)


def test_round_trip_drifts_when_the_arm_lags():
    """Documents the known limitation: imperfect tracking is not corrected inside a chunk.

    Not a defect of this code -- it is why the deployed procedure re-queries rather than running
    one long chunk blind, and why the sim verification closes the loop instead of trusting the
    algebra alone.
    """
    state = np.concatenate([np.zeros(7), [0.0]])
    targets = np.cumsum(np.full((HORIZON, 7), 0.01), axis=0)

    result = cosmos_droid_policy.cosmos_velocity_output_chain()(
        {"state": state, **_cosmos_response(targets)}
    )
    perfect = _droid_client_reconstruct(state, result["actions"], tracking=1.0)
    lagging = _droid_client_reconstruct(state, result["actions"], tracking=0.9)

    np.testing.assert_allclose(perfect, targets, atol=1e-6)
    assert np.abs(lagging - targets).max() > 1e-3, "a lagging arm should visibly drift"


def test_first_row_is_measured_from_state_and_the_rest_from_the_previous_target():
    state = np.zeros(8)
    targets = np.concatenate(
        [np.full((1, 7), 0.10), np.full((1, 7), 0.14), np.full((HORIZON - 2, 7), 0.18)]
    )

    result = cosmos_droid_policy.cosmos_velocity_output_chain()(
        {"state": state, **_cosmos_response(targets)}
    )

    # float32 tolerances: the chunk arrives as float32 and dividing by 0.2 scales the
    # representation error by five. The physical quantity is rad/s, where 1e-6 is nothing.
    np.testing.assert_allclose(result["actions"][0, :7], 0.10 / SCALE, rtol=1e-5)
    np.testing.assert_allclose(result["actions"][1, :7], 0.04 / SCALE, rtol=1e-5)
    np.testing.assert_allclose(result["actions"][2, :7], 0.04 / SCALE, rtol=1e-5)
    np.testing.assert_allclose(result["actions"][3, :7], 0.0, atol=1e-6)


def test_stationary_chunk_produces_zero_velocity():
    state = np.concatenate([np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7]), [0.25]])
    targets = np.repeat(state[np.newaxis, :7], HORIZON, axis=0)

    result = cosmos_droid_policy.cosmos_velocity_output_chain()(
        {"state": state, **_cosmos_response(targets)}
    )

    np.testing.assert_allclose(result["actions"][:, :7], 0.0, atol=1e-6)


def test_gripper_passes_through_untouched():
    """Cosmos already un-inverts the gripper server-side; a second flip is invisible until the
    success rate is zero."""
    rng = np.random.default_rng(1)
    state = np.concatenate([np.zeros(7), [0.0]])
    gripper = (rng.random(HORIZON) > 0.5).astype(np.float32)
    response = _cosmos_response(np.zeros((HORIZON, 7)), gripper=gripper)

    result = cosmos_droid_policy.cosmos_velocity_output_chain()({"state": state, **response})

    np.testing.assert_array_equal(result["actions"][:, 7], gripper)


# --------------------------------------------------------------------------------------
# The difference from pi05 that is most likely to be copied by mistake
# --------------------------------------------------------------------------------------


def test_adding_absolute_actions_would_double_the_targets():
    """Negative control for the one stage this path must NOT reuse from pi05.

    pi05 predicts deltas and needs ``AbsoluteActions``; Cosmos predicts absolute angles. Keeping
    that stage adds the state a second time. Nothing raises -- the arm simply goes somewhere
    else -- so the failure is pinned here instead.
    """
    # Deliberately small: at 0.5 rad the doubled step is 2.5 in velocity units and the +-0.5
    # clip caps it, which *partially hides* the bug -- the arm creeps instead of jumping. At
    # 0.05 rad the doubled step stays inside the clip and the error shows at full size.
    state = np.concatenate([np.full(7, 0.05), [0.0]])
    targets = np.full((HORIZON, 7), 0.05)  # "stay where you are"

    correct = cosmos_droid_policy.cosmos_velocity_output_chain()(
        {"state": state, **_cosmos_response(targets)}
    )
    np.testing.assert_allclose(correct["actions"][:, :7], 0.0, atol=1e-6)

    wrong_chain = transforms.compose(
        [
            cosmos_droid_policy.CosmosActionsToJointPositions(),
            transforms.AbsoluteActions(transforms.make_bool_mask(7, -1)),
            droid_policy.JointPositionToDroidVelocity(expected_horizon=HORIZON),
            droid_policy.DroidOutputs(),
        ]
    )
    wrong = wrong_chain({"state": state, **_cosmos_response(targets)})

    rebuilt = _droid_client_reconstruct(state, wrong["actions"])
    # "stay at 0.05" becomes "go to 0.10": the state was added twice.
    np.testing.assert_allclose(rebuilt[0], np.full(7, 0.10), atol=1e-5)

    # And the same mistake at a larger pose, where the clip caps the command. The target is
    # still wrong, just less obviously -- worth pinning so nobody reads a clipped log line as
    # evidence that the chain is fine.
    big_state = np.concatenate([np.full(7, 0.5), [0.0]])
    big = wrong_chain({"state": big_state, **_cosmos_response(np.full((HORIZON, 7), 0.5))})
    np.testing.assert_allclose(big["actions"][0, :7], 0.5)  # saturated
    big_rebuilt = _droid_client_reconstruct(big_state, big["actions"])
    assert np.abs(big_rebuilt[0] - 0.5).max() > 0.05, "clipped, but still not holding position"


# --------------------------------------------------------------------------------------
# Clipping
# --------------------------------------------------------------------------------------


def test_clip_does_not_engage_at_the_measured_cosmos_speed():
    """0.0074 rad/step is the median joint delta measured over 5 simulated Cosmos runs
    (24 environments x 300 steps each); the measured saturation rate was 0.013-0.041%."""
    state = np.zeros(8)
    targets = np.cumsum(np.full((HORIZON, 7), 0.0074), axis=0)

    result = cosmos_droid_policy.cosmos_velocity_output_chain()(
        {"state": state, **_cosmos_response(targets)}
    )

    assert np.abs(result["actions"][:, :7]).max() < 0.5


def test_clip_engages_and_is_logged_when_the_step_is_too_large(caplog: pytest.LogCaptureFixture):
    import logging

    state = np.zeros(8)
    targets = np.cumsum(np.full((HORIZON, 7), 0.4), axis=0)  # 2.0 in velocity units

    with caplog.at_level(logging.INFO, logger=droid_policy.__name__):
        result = cosmos_droid_policy.cosmos_velocity_output_chain()(
            {"state": state, **_cosmos_response(targets)}
        )

    np.testing.assert_allclose(result["actions"][:, :7], 0.5)
    assert "saturation_fraction=1.000000" in caplog.text


# --------------------------------------------------------------------------------------
# Malformed input
# --------------------------------------------------------------------------------------


def test_rejects_a_response_that_was_already_converted():
    with pytest.raises(ValueError, match="already been converted"):
        cosmos_droid_policy.CosmosActionsToJointPositions()(
            {"state": np.zeros(8), "actions": np.zeros((HORIZON, 8))}
        )


@pytest.mark.parametrize(
    ("actions", "message"),
    [
        (np.zeros(8), "actions must have shape"),
        (np.zeros((15, 8)), "expected action horizon"),
        (np.zeros((HORIZON, 7)), "7 joints and 1 gripper"),
        (np.full((HORIZON, 8), np.nan), "NaN or Inf"),
    ],
)
def test_rejects_malformed_chunks(actions, message):
    with pytest.raises(ValueError, match=message):
        cosmos_droid_policy.CosmosActionsToJointPositions()(
            {cosmos_droid_policy.COSMOS_ACTION_KEY: actions}
        )


def test_dtype_is_normalised_to_float32():
    result = cosmos_droid_policy.CosmosActionsToJointPositions()(
        {cosmos_droid_policy.COSMOS_ACTION_KEY: np.zeros((HORIZON, 8), dtype=np.float64)}
    )
    assert result["actions"].dtype == np.float32


# --------------------------------------------------------------------------------------
# Request geometry
# --------------------------------------------------------------------------------------


def test_composite_frame_matches_the_cosmos_layout():
    rng = np.random.default_rng(2)
    left, right, wrist = (rng.integers(1, 256, size=(720, 1280, 3), dtype=np.uint8) for _ in range(3))

    frame = cosmos_droid_policy.make_cosmos_observation_image(left, right, wrist)

    assert frame.shape == (540, 640, 3)
    assert frame.dtype == np.uint8
    # Distinct exterior views land in distinct quadrants.
    assert not np.array_equal(frame[360:, :320], frame[360:, 320:])


def test_composite_frame_rejects_a_non_rgb_view():
    good = np.zeros((720, 1280, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match=r"wrist_image must be \[H, W, 3\]"):
        cosmos_droid_policy.make_cosmos_observation_image(good, good, np.zeros((720, 1280)))


def test_make_state_concatenates_joints_and_gripper():
    state = cosmos_droid_policy.make_state(np.arange(7, dtype=np.float32), np.float32(0.25))
    assert state.shape == (8,)
    np.testing.assert_allclose(state[:7], np.arange(7))
    np.testing.assert_allclose(state[7], 0.25)


def test_composite_frame_is_byte_identical_to_the_reference_cosmos_client():
    """Our composition must match ``Cosmos3Client._pack_request`` exactly.

    That client is what every measured Cosmos number in simulation was produced with, so a
    different interpolation here would hand the policy different pixels while every log line
    still looked healthy. The reference arithmetic is reproduced inline rather than imported:
    the RoboLab tree is a separate checkout with its own environment.
    """
    import torch
    import torch.nn.functional as F  # noqa: N812
    from openpi_client import image_tools

    rng = np.random.default_rng(7)
    left, right, wrist = (rng.integers(1, 256, size=(720, 1280, 3), dtype=np.uint8) for _ in range(3))

    # -- reference: RoboLab/policies/cosmos3/client.py, _extract_observation_serial + _pack_request
    h, w = 360, 640
    ref_wrist = image_tools.resize_with_pad(wrist, h, w)
    ref_left = image_tools.resize_with_pad(left, h, w)
    ref_right = image_tools.resize_with_pad(right, h, w)

    def _shrink(x):
        t = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float()
        t = F.interpolate(t, size=(h // 2, w // 2), mode="bilinear")
        return t.squeeze(0).permute(1, 2, 0).numpy().astype(ref_wrist.dtype)

    reference = np.concatenate(
        (ref_wrist, np.concatenate((_shrink(ref_left), _shrink(ref_right)), axis=1))
    )

    ours = cosmos_droid_policy.make_cosmos_observation_image(left, right, wrist)

    assert ours.shape == reference.shape
    np.testing.assert_array_equal(ours, reference)


def test_composite_accepts_views_the_robot_already_downscaled():
    """The DROID client resizes on the robot laptop to cut latency. Sending views already at
    360x640 must give the same frame as sending the raw camera images, because
    ``resize_with_pad`` returns early when the shape already matches."""
    from openpi_client import image_tools

    rng = np.random.default_rng(8)
    left, right, wrist = (rng.integers(1, 256, size=(720, 1280, 3), dtype=np.uint8) for _ in range(3))

    from_raw = cosmos_droid_policy.make_cosmos_observation_image(left, right, wrist)
    from_resized = cosmos_droid_policy.make_cosmos_observation_image(
        *(image_tools.resize_with_pad(v, 360, 640) for v in (left, right, wrist))
    )

    np.testing.assert_array_equal(from_raw, from_resized)
