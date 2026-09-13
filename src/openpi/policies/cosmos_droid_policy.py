"""Serve a Cosmos3 DROID policy through the deployed KETI joint-velocity contract.

This is the Cosmos counterpart of the ``pi05_droid_jointpos_velocity`` path. The robot side
is unchanged: the 3090 workstation still runs ``RobotEnv(action_space="joint_velocity")`` and
the G15 controller is untouched. What changes is where the joint targets come from.

Why this file exists instead of another entry in ``training/config.py``
----------------------------------------------------------------------
Cosmos3 is not an openpi model. It is served by its own websocket server
(``cosmos_framework.scripts.action_policy_server_robolab``) which has no notion of openpi's
transform chain, so the chain has to run in a proxy that sits in front of it. Only the output
half of the chain is ours; the input half is the Cosmos request format.

How the chain differs from pi05
-------------------------------
pi05's output chain is::

    Unnormalize -> AbsoluteActions -> JointPositionToDroidVelocity -> DroidOutputs

Cosmos's is::

    CosmosActionsToJointPositions -> JointPositionToDroidVelocity -> DroidOutputs

Two stages are gone and one is new:

``Unnormalize``
    The Cosmos server unnormalizes inside its own process; what comes back over the wire is
    already in joint-angle units.

``AbsoluteActions``  **-- removing this is the single most dangerous difference**
    pi05 predicts joint-position *deltas*, so the current state has to be added back.
    **Cosmos predicts absolute joint angles directly.** Measured on a live server: feeding the
    robot's current pose ``q`` and reading the first row ``a0`` of the returned chunk gives
    ``|a0 - q| = 0.023 rad`` against ``|a0| = 0.721 rad`` -- ``a0`` sits on ``q``, not on zero.
    Confirmed again in simulation, where the first executed action matched RoboLab's Franka
    reset pose to within 1e-3 rad per joint. Keeping ``AbsoluteActions`` here would add the
    state a second time and roughly double every joint target, with no error raised anywhere.

``CosmosActionsToJointPositions`` (new)
    Renames the Cosmos response key (``action``) to openpi's (``actions``), normalizes dtype,
    and rejects malformed chunks. It deliberately performs no arithmetic on the values.

``JointPositionToDroidVelocity`` and ``DroidOutputs`` are reused from ``droid_policy`` verbatim,
so the conversion the robot sees is bit-for-bit the one the deployed pi05 path uses. Only
``expected_horizon`` differs (see below).

Action horizon
--------------
Cosmos's chunk is 32 steps, set by the checkpoint; pi05's velocity config declares 15 to match
the deployed contract. Truncating Cosmos to 15 is **not** free: in a paired simulation A/B over
24 environments, success went from 46/72 at chunk 32 to 17/48 at chunk 15, with no overlap
between the per-run ranges (32: 17, 17, 12; 15: 9, 8). Cosmos is trained to emit a complete
32-step motion and restarting it early leaves the arm repeating the approach.

Latency points the same way: Cosmos inference is ~860 ms on an idle GPU. At 15 Hz a 32-step
chunk covers 2.13 s, an 8-step chunk 0.53 s -- so the DROID client's default
``open_loop_horizon=8`` cannot be served in real time regardless of success rate.

Both arguments give the same answer: serve 32 and execute 32. This requires the 3090 client to
raise ``action_horizon`` and ``open_loop_horizon`` to 32; see ``examples/droid/main_cosmos.py``.

Gripper
-------
Untouched, in both directions. The Cosmos server already inverts the gripper on the way in and
inverts it back on the way out, so the value on the wire follows the same convention pi05 uses
(0 = open, 1 = closed). ``JointPositionToDroidVelocity`` passes column 7 through unchanged.
Inverting it here would make the arm open when it should grasp, and nothing would log an error.
"""

from __future__ import annotations

import dataclasses
import logging

import numpy as np

from openpi import transforms
from openpi.policies import droid_policy

logger = logging.getLogger(__name__)

# Cosmos request geometry, from the reference RoboLab client
# (``RoboLab/policies/cosmos3/client.py::Cosmos3Client``). The server validates the frame shape
# on arrival, so these are a contract rather than a preference.
COSMOS_VIEW_HEIGHT = 360
COSMOS_VIEW_WIDTH = 640
COSMOS_IMAGE_HEIGHT = 540  # wrist (360) stacked on the half-height exterior pair (180)
COSMOS_IMAGE_WIDTH = 640

# The Cosmos checkpoint's action horizon. Not a tunable: it is inferred from the checkpoint and
# feeds the conditioning tensor shape.
COSMOS_ACTION_HORIZON = 32

# Key the Cosmos server returns its chunk under. openpi uses the plural form everywhere.
COSMOS_ACTION_KEY = "action"


def make_cosmos_observation_image(
    left_image: np.ndarray,
    right_image: np.ndarray,
    wrist_image: np.ndarray,
) -> np.ndarray:
    """Compose the three DROID camera views into the single frame Cosmos expects.

    Layout, matching ``Cosmos3Client._pack_request``: the wrist view on top at 360x640, and the
    two exterior views side by side beneath it at 180x320 each, giving 540x640.

    The arithmetic is copied from that client rather than reimplemented: ``resize_with_pad`` to
    360x640 first, then a bilinear ``F.interpolate`` down to half size for the exterior pair.
    Using a different interpolation here would change the pixels the policy sees, which is
    exactly the kind of silent difference that makes a sim A/B stop being comparable.

    Args:
        left_image: Left exterior camera, HxWx3 uint8.
        right_image: Right exterior camera, HxWx3 uint8.
        wrist_image: Wrist camera, HxWx3 uint8.

    Returns:
        A 540x640x3 uint8 frame.
    """
    import torch
    import torch.nn.functional as F  # noqa: N812
    from openpi_client import image_tools

    for name, image in (("left", left_image), ("right", right_image), ("wrist", wrist_image)):
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"{name}_image must be [H, W, 3], got {image.shape}")

    wrist = image_tools.resize_with_pad(np.asarray(wrist_image), COSMOS_VIEW_HEIGHT, COSMOS_VIEW_WIDTH)

    def _half(image: np.ndarray) -> np.ndarray:
        full = image_tools.resize_with_pad(np.asarray(image), COSMOS_VIEW_HEIGHT, COSMOS_VIEW_WIDTH)
        tensor = torch.from_numpy(full).permute(2, 0, 1).unsqueeze(0).float()
        tensor = F.interpolate(
            tensor, size=(COSMOS_VIEW_HEIGHT // 2, COSMOS_VIEW_WIDTH // 2), mode="bilinear"
        )
        return tensor.squeeze(0).permute(1, 2, 0).numpy().astype(wrist.dtype)

    exterior = np.concatenate((_half(left_image), _half(right_image)), axis=1)
    return np.concatenate((wrist, exterior), axis=0)


@dataclasses.dataclass(frozen=True)
class CosmosActionsToJointPositions(transforms.DataTransformFn):
    """Adopt the Cosmos server's chunk as openpi's ``actions``, without touching the values.

    Cosmos returns ``{"action": [horizon, 8]}`` holding **absolute** joint angles plus a gripper
    command. This renames the key, normalizes dtype to float32, and validates the shape. It adds
    nothing to the values -- see the module docstring on why ``AbsoluteActions`` must not run
    before ``JointPositionToDroidVelocity`` on this path.

    ``expected_horizon`` is checked rather than inferred, so a server started with a different
    ``--action-chunk-size`` fails here instead of producing a chunk the robot client will reject
    a layer later.
    """

    expected_horizon: int = COSMOS_ACTION_HORIZON
    source_key: str = COSMOS_ACTION_KEY

    def __post_init__(self) -> None:
        if self.expected_horizon <= 0:
            raise ValueError("expected_horizon must be positive")

    def __call__(self, data: dict) -> dict:
        if self.source_key not in data:
            raise ValueError(
                f"Cosmos response is missing {self.source_key!r}; got keys {sorted(data)}. "
                "A response carrying 'actions' has already been converted."
            )

        actions = np.asarray(data[self.source_key])
        if actions.ndim != 2:
            raise ValueError(f"actions must have shape [horizon, action_dim], got {actions.shape}")
        if actions.shape[0] != self.expected_horizon:
            raise ValueError(f"expected action horizon {self.expected_horizon}, got {actions.shape[0]}")
        if actions.shape[1] < 8:
            raise ValueError(f"actions must contain 7 joints and 1 gripper, got action_dim={actions.shape[1]}")
        if not np.all(np.isfinite(actions)):
            raise ValueError("actions contain NaN or Inf")

        converted = {key: value for key, value in data.items() if key != self.source_key}
        converted["actions"] = np.ascontiguousarray(actions, dtype=np.float32)
        return converted


def cosmos_velocity_output_chain(
    *,
    expected_horizon: int = COSMOS_ACTION_HORIZON,
    joint_delta_scale: float = 0.2,
    max_abs_velocity: float = 0.5,
) -> transforms.DataTransformFn:
    """The full output chain: Cosmos response in, DROID joint-velocity commands out.

    The last two stages are ``droid_policy``'s own, unmodified, so the command the G15 receives
    is produced by the same code path the deployed pi05 server uses.
    """
    return transforms.compose(
        [
            CosmosActionsToJointPositions(expected_horizon=expected_horizon),
            droid_policy.JointPositionToDroidVelocity(
                expected_horizon=expected_horizon,
                joint_delta_scale=joint_delta_scale,
                max_abs_velocity=max_abs_velocity,
            ),
            droid_policy.DroidOutputs(),
        ]
    )


def make_state(joint_position: np.ndarray, gripper_position: np.ndarray) -> np.ndarray:
    """Build the ``state`` vector ``JointPositionToDroidVelocity`` reads.

    ``DroidInputs`` does this for the openpi path; the Cosmos proxy has no such stage, so the
    proxy assembles it from the same two request fields.
    """
    gripper = np.asarray(gripper_position)
    if gripper.ndim == 0:
        gripper = gripper[np.newaxis]
    return np.concatenate([np.asarray(joint_position).reshape(-1), gripper.reshape(-1)]).astype(np.float64)
