import dataclasses
import logging

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

logger = logging.getLogger(__name__)


def make_droid_example() -> dict:
    """Creates a random input example for the Droid policy."""
    return {
        "observation/exterior_image_1_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.random.rand(7),
        "observation/gripper_position": np.random.rand(1),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class DroidInputs(transforms.DataTransformFn):
    # Determines which model will be used.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        gripper_pos = np.asarray(data["observation/gripper_position"])
        if gripper_pos.ndim == 0:
            # Ensure gripper position is a 1D array, not a scalar, so we can concatenate with joint positions
            gripper_pos = gripper_pos[np.newaxis]
        state = np.concatenate([data["observation/joint_position"], gripper_pos])

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference
        base_image = _parse_image(data["observation/exterior_image_1_left"])
        wrist_image = _parse_image(data["observation/wrist_image_left"])

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (base_image, wrist_image, np.zeros_like(base_image))
                image_masks = (np.True_, np.True_, np.False_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                # We don't mask out padding images for FAST models.
                images = (base_image, np.zeros_like(base_image), wrist_image)
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class DroidOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Only return the first 8 dims.
        return {"actions": np.asarray(data["actions"][:, :8])}


@dataclasses.dataclass(frozen=True)
class JointPositionToDroidVelocity(transforms.DataTransformFn):
    """Convert an absolute joint-position chunk to DROID normalized joint velocity commands.

    DROID's ``joint_velocity`` action space interprets each joint command as a normalized
    per-control-step delta and multiplies it by ``joint_delta_scale`` before sending an
    absolute target to the robot. The first target is measured from the current observation;
    later targets are measured from the previous target in the same open-loop chunk.

    Gripper and padded model dimensions are preserved. ``DroidOutputs`` is expected to run
    after this transform and retain the first seven joints plus the gripper dimension.
    """

    expected_horizon: int = 15
    joint_delta_scale: float = 0.2
    max_abs_velocity: float = 0.5

    def __post_init__(self) -> None:
        if self.expected_horizon <= 0:
            raise ValueError("expected_horizon must be positive")
        if self.joint_delta_scale <= 0:
            raise ValueError("joint_delta_scale must be positive")
        if self.max_abs_velocity <= 0:
            raise ValueError("max_abs_velocity must be positive")

    def __call__(self, data: dict) -> dict:
        if "actions" not in data:
            raise ValueError("actions are required for joint-position conversion")
        if "state" not in data:
            raise ValueError("state is required for joint-position conversion")

        actions = np.asarray(data["actions"])
        state = np.asarray(data["state"])
        if actions.ndim != 2:
            raise ValueError(f"actions must have shape [horizon, action_dim], got {actions.shape}")
        if actions.shape[0] != self.expected_horizon:
            raise ValueError(f"expected action horizon {self.expected_horizon}, got {actions.shape[0]}")
        if actions.shape[1] < 8:
            raise ValueError(f"actions must contain 7 joints and 1 gripper, got action_dim={actions.shape[1]}")
        if state.ndim != 1 or state.shape[0] < 7:
            raise ValueError(f"state must contain at least 7 joint positions, got {state.shape}")
        if not np.all(np.isfinite(actions)):
            raise ValueError("actions contain NaN or Inf")
        if not np.all(np.isfinite(state[:7])):
            raise ValueError("joint state contains NaN or Inf")

        joint_targets = actions[:, :7]
        previous_targets = np.concatenate([state[np.newaxis, :7], joint_targets[:-1]], axis=0)
        raw_joint_velocity = (joint_targets - previous_targets) / self.joint_delta_scale
        saturation_mask = np.abs(raw_joint_velocity) > self.max_abs_velocity
        saturation_fraction = float(np.mean(saturation_mask))

        converted_actions = np.array(actions, copy=True)
        converted_actions[:, :7] = np.clip(
            raw_joint_velocity,
            -self.max_abs_velocity,
            self.max_abs_velocity,
        )

        logger.info(
            "DROID joint-position adapter: raw_min=%.6f raw_max=%.6f saturation_fraction=%.6f limit=%.3f horizon=%d",
            float(np.min(raw_joint_velocity)),
            float(np.max(raw_joint_velocity)),
            saturation_fraction,
            self.max_abs_velocity,
            actions.shape[0],
        )
        return {**data, "actions": converted_actions}
