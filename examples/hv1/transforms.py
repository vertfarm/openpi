"""HV1-only transforms. No Aloha sign flips or DROID velocity conversion."""

from dataclasses import dataclass

import numpy as np

from .artifacts import ContractError
from .workflow import validate_profile

CAMERA_KEYS = {"head": "base_0_rgb", "hand_l": "left_wrist_0_rgb", "hand_r": "right_wrist_0_rgb"}
AUGMENTED_INTERFACE = "hv1_augmented_v1"
AUGMENTED_TASKS = {
    "cylinder_table_to_tray",
    "cylinder_tray_to_table",
    "cylinder_language_zone_sort",
    "cylinder_axis_alignment",
}
AUGMENTED_PHASES = {"reach", "pre_grasp", "grasp", "lift", "transport", "place", "release"}
AUGMENTED_SOURCES = {"real", "sim_physics", "sim_kinematic"}
AUGMENTED_STYLES = {"real", "rtx", "3dgs", "cosmos"}
KINEMATIC_PHASES = {"reach", "pre_grasp", "transport"}


def validate_augmented_sample(data, *, allow_synthetic=False):
    """Validate the common real/sim wrapper before any OpenPI transform runs."""
    try:
        if data["schema_version"] != AUGMENTED_INTERFACE:
            raise ContractError("unsupported augmented interface")
        state = np.asarray(data["state"], dtype=np.float32)
        actions = np.asarray(data["actions"], dtype=np.float32)
        if state.shape != (15,) or actions.shape != (15, 8):
            raise ContractError("augmented sample must carry state[15] and actions[15,8]")
        if not np.isfinite(state).all() or not np.isfinite(actions).all():
            raise ContractError("augmented state/action contains NaN or Inf")
        if set(data["images"]) != set(CAMERA_KEYS):
            raise ContractError("augmented sample requires all three cameras")
        if data["task_id"] not in AUGMENTED_TASKS or data["phase"] not in AUGMENTED_PHASES:
            raise ContractError("unknown augmented task or phase")
        source, style = data["source_domain"], data["render_style"]
        if source not in AUGMENTED_SOURCES or style not in AUGMENTED_STYLES:
            raise ContractError("unknown augmented source or render style")
        synthetic = data["synthetic"]
        if type(synthetic) is not bool:
            raise ContractError("synthetic must be boolean")
        if synthetic != (source != "real"):
            raise ContractError("synthetic marker disagrees with source_domain")
        if synthetic and not allow_synthetic:
            raise ContractError("synthetic data requires explicit --allow-synthetic")
        if (source == "real") != (style == "real"):
            raise ContractError("real source/render identity mismatch")
        if source == "sim_kinematic" and data["phase"] not in KINEMATIC_PHASES:
            raise ContractError("sim_kinematic contact phase is forbidden")
        if not isinstance(data["language"], str) or not data["language"].strip():
            raise ContractError("augmented language prompt is empty")
        if type(data["sampleable"]) is not bool or not np.isfinite(float(data["quality_weight"])):
            raise ContractError("invalid sampleability or quality weight")
        if data["quality_weight"] < 0:
            raise ContractError("quality weight cannot be negative")
        if not data["sampleable"]:
            raise ContractError("non-sampleable record reached the SFT wrapper")
        if data["success"] is not True and data.get("corrected_recovery") is not True:
            raise ContractError("SFT accepts only success or corrected recovery")
        for key in ("seeds", "hashes", "camera_calibration", "randomization_vector", "timestamps"):
            if not isinstance(data[key], dict):
                raise ContractError(f"augmented {key} must be an object")
        if set(data["camera_calibration"]) != set(CAMERA_KEYS):
            raise ContractError("camera calibration identity is incomplete")
        hashes = data["hashes"]
        if any(
            not isinstance(hashes.get(key), str)
            or len(hashes[key]) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in hashes[key])
            for key in ("robot", "scene", "asset", "policy_checkpoint")
        ):
            raise ContractError("augmented provenance SHA-256 is incomplete")
        timestamps = data["timestamps"]
        anchor_ns = timestamps["anchor_ns"]
        image_ns = [timestamps["image_ns"][camera] for camera in CAMERA_KEYS]
        actions_ns = timestamps["actions_ns"]
        if (
            type(anchor_ns) is not int
            or len(actions_ns) != 15
            or any(type(value) is not int for value in [*image_ns, *actions_ns])
            or actions_ns != sorted(actions_ns)
            or actions_ns[0] < anchor_ns
            or any(value > anchor_ns or anchor_ns - value > 150_000_000 for value in image_ns)
            or max(image_ns) - min(image_ns) > 100_000_000
        ):
            raise ContractError("augmented timestamps violate causality, age, or skew")
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, ContractError):
            raise
        raise ContractError(f"invalid augmented sample: {exc}") from exc
    return data


@dataclass(frozen=True)
class HV1AugmentedInputs:
    """Map ``hv1_augmented_v1`` anchors through the unchanged HV1 policy contract."""

    profile: dict
    allow_synthetic: bool = False

    def __call__(self, data):
        sample = validate_augmented_sample(data, allow_synthetic=self.allow_synthetic)
        return HV1Inputs(self.profile)(
            {
                "state": sample["state"],
                "images": sample["images"],
                "actions": sample["actions"],
                "prompt": sample["language"],
            }
        )


@dataclass(frozen=True)
class HV1Inputs:
    profile: dict

    def __call__(self, data):
        p = validate_profile(self.profile)
        state = np.asarray(data["state"], dtype=np.float32)
        if state.shape != (len(p["state"]["names"]),) or not np.isfinite(state).all():
            raise ContractError("HV1 state shape/value mismatch")
        images, masks = {}, {}
        for source, target in CAMERA_KEYS.items():
            if source in p["images"]:
                image = np.asarray(data["images"][source])
                # LeRobot's loader returns CHW float [0,1]; live/replay input uses RGB HWC uint8.
                expected = tuple(p["images"][source]["shape"])
                if image.shape == (3, expected[0], expected[1]):
                    image = image.transpose(1, 2, 0)
                if image.shape != expected:
                    raise ContractError(f"{source}: shape mismatch")
                if image.dtype != np.uint8:
                    if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
                        raise ContractError("float image must be finite [0,1]")
                    image = np.rint(image * 255).astype(np.uint8)
                images[target], masks[target] = image, np.True_
            else:
                images[target], masks[target] = np.zeros((224, 224, 3), np.uint8), np.False_
        result = {"image": images, "image_mask": masks, "state": state}
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32).copy()
            if actions.ndim != 2 or actions.shape[-1] != len(p["action"]["names"]) or not np.isfinite(actions).all():
                raise ContractError("HV1 action shape/value mismatch")
            for action_index, state_index in enumerate(p["action"]["delta_state_indices"]):
                if state_index >= 0:
                    actions[:, action_index] -= state[state_index]
            result["actions"] = actions
        if "prompt" in data:
            result["prompt"] = data["prompt"]
        return result


@dataclass(frozen=True)
class HV1Outputs:
    profile: dict

    def __call__(self, data):
        p = self.profile
        actions = np.asarray(data["actions"])
        if actions.ndim != 2 or actions.shape[1] < len(p["action"]["names"]):
            raise ContractError("policy action shape mismatch")
        actions = actions[:, : len(p["action"]["names"])].copy()
        for action_index, state_index in enumerate(p["action"]["delta_state_indices"]):
            if state_index >= 0:
                actions[:, action_index] += np.asarray(data["state"])[state_index]
        if not np.isfinite(actions).all():
            raise ContractError("nonfinite policy output")
        return {"actions": actions}
