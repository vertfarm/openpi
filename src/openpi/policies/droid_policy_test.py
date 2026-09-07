import logging

import numpy as np
import pytest

from openpi import transforms
from openpi.policies import droid_policy
from openpi.training import config as _config


def _make_actions(joint_targets: np.ndarray, gripper: np.ndarray | None = None, *, action_dim: int = 32) -> np.ndarray:
    horizon = joint_targets.shape[0]
    actions = np.zeros((horizon, action_dim), dtype=np.float64)
    actions[:, :7] = joint_targets
    actions[:, 7] = np.arange(horizon) / max(horizon - 1, 1) if gripper is None else gripper
    return actions


def test_joint_position_to_droid_velocity_stationary_and_gripper_preserved():
    state = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, 0.25])
    joints = np.repeat(state[np.newaxis, :7], repeats=3, axis=0)
    actions = _make_actions(joints)

    result = droid_policy.JointPositionToDroidVelocity(expected_horizon=3)({"state": state, "actions": actions})

    np.testing.assert_allclose(result["actions"][:, :7], 0.0)
    np.testing.assert_array_equal(result["actions"][:, 7], actions[:, 7])
    np.testing.assert_array_equal(result["actions"][:, 8:], actions[:, 8:])


def test_joint_position_to_droid_velocity_uses_observation_then_previous_target():
    state = np.zeros(8)
    joints = np.stack(
        [
            np.full(7, 0.10),
            np.full(7, 0.14),
            np.full(7, 0.18),
        ]
    )
    actions = _make_actions(joints, gripper=np.array([0.1, 0.4, 0.9]))

    result = droid_policy.JointPositionToDroidVelocity(expected_horizon=3)({"state": state, "actions": actions})

    np.testing.assert_allclose(result["actions"][:, :7], np.array([[0.5] * 7, [0.2] * 7, [0.2] * 7]))
    np.testing.assert_array_equal(result["actions"][:, 7], np.array([0.1, 0.4, 0.9]))


def test_jointpos_output_chain_reconstructs_absolute_targets_before_velocity_conversion():
    state = np.array([1.0, -1.0, 0.5, -0.5, 0.2, -0.2, 0.0, 0.25])
    absolute_targets = np.stack([state[:7] + 0.1, state[:7] + 0.14, state[:7] + 0.18])
    delta_actions = _make_actions(absolute_targets - state[:7], gripper=np.array([0.2, 0.4, 0.8]))
    output_chain = transforms.compose(
        [
            transforms.AbsoluteActions(transforms.make_bool_mask(7, -1)),
            droid_policy.JointPositionToDroidVelocity(expected_horizon=3),
            droid_policy.DroidOutputs(),
        ]
    )

    result = output_chain({"state": state, "actions": delta_actions})

    assert result["actions"].shape == (3, 8)
    np.testing.assert_allclose(result["actions"][:, :7], np.array([[0.5] * 7, [0.2] * 7, [0.2] * 7]))
    np.testing.assert_array_equal(result["actions"][:, 7], np.array([0.2, 0.4, 0.8]))


def test_joint_position_to_droid_velocity_clips_and_logs_saturation(caplog: pytest.LogCaptureFixture):
    state = np.zeros(8)
    actions = _make_actions(np.full((2, 7), 0.4))

    with caplog.at_level(logging.INFO, logger=droid_policy.__name__):
        result = droid_policy.JointPositionToDroidVelocity(expected_horizon=2)(
            {
                "state": state,
                "actions": actions,
            }
        )

    np.testing.assert_allclose(result["actions"][0, :7], 0.5)
    np.testing.assert_allclose(result["actions"][1, :7], 0.0)
    assert "raw_max=2.000000" in caplog.text
    assert "saturation_fraction=0.500000" in caplog.text


@pytest.mark.parametrize(
    ("state", "actions", "message"),
    [
        (np.zeros(8), np.zeros(8), "actions must have shape"),
        (np.zeros(8), np.zeros((2, 8)), "expected action horizon"),
        (np.zeros(8), np.zeros((3, 7)), "7 joints and 1 gripper"),
        (np.zeros(6), np.zeros((3, 8)), "state must contain"),
        (np.zeros(8), np.full((3, 8), np.nan), "actions contain NaN or Inf"),
        (np.full(8, np.inf), np.zeros((3, 8)), "joint state contains NaN or Inf"),
    ],
)
def test_joint_position_to_droid_velocity_rejects_invalid_inputs(state, actions, message):
    transform = droid_policy.JointPositionToDroidVelocity(expected_horizon=3)

    with pytest.raises(ValueError, match=message):
        transform({"state": state, "actions": actions})


@pytest.mark.parametrize("missing_key", ["state", "actions"])
def test_joint_position_to_droid_velocity_requires_state_and_actions(missing_key):
    data = {"state": np.zeros(8), "actions": np.zeros((3, 8))}
    del data[missing_key]

    with pytest.raises(ValueError, match=f"{missing_key} .* required"):
        droid_policy.JointPositionToDroidVelocity(expected_horizon=3)(data)


def test_pi05_droid_jointpos_velocity_config_contract():
    config = _config.get_config("pi05_droid_jointpos_velocity")
    data_config = config.data.create(config.assets_dirs, config.model)

    assert config.model.action_horizon == 15
    assert config.model.action_dim == 32
    assert config.policy_metadata["checkpoint"] == "gs://openpi-assets/checkpoints/pi05_droid_jointpos"
    assert config.policy_metadata["output_action_space"] == "joint_velocity"
    assert [type(transform) for transform in data_config.data_transforms.outputs] == [
        transforms.AbsoluteActions,
        droid_policy.JointPositionToDroidVelocity,
        droid_policy.DroidOutputs,
    ]
