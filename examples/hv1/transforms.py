"""HV1-only transforms. No Aloha sign flips or DROID velocity conversion."""

from dataclasses import dataclass

import numpy as np

from .workflow import ContractError
from .workflow import validate_profile

CAMERA_KEYS = {"head": "base_0_rgb", "hand_l": "left_wrist_0_rgb", "hand_r": "right_wrist_0_rgb"}


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
