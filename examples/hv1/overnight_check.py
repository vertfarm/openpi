"""Official loader and model transforms, checked at every episode boundary."""

import argparse
from pathlib import Path

import numpy as np

from .artifacts import ContractError
from .artifacts import read_json
from .artifacts import write_new_json
from .native import PROMPT
from .native import read_numeric
from .overnight_common import recipe
from .overnight_train import configure
from .transforms import HV1Outputs


def check(campaign, name):
    config, export = configure(campaign, recipe(name, 50))
    from openpi import transforms
    from openpi.training import data_loader

    data = config.data.create(config.assets_dirs, config.model)
    ds = data_loader.create_torch_dataset(data, config.model.action_horizon, config.model)
    provenance = read_json(Path(export["splits"]["train"]["root"]) / "hv1_provenance.json")
    offset = 0
    checked = 0
    for e in provenance:
        state, action, _ = read_numeric(Path(e["path"]) / "data.hdf5")
        for local in sorted({0, e["grasp_frame"], e["release_frame"], len(state) - 1}):
            row = ds[offset + local]
            expected = action[np.minimum(np.arange(config.model.action_horizon) + local, len(state) - 1)]
            np.testing.assert_allclose(row["action"], expected, atol=1e-6)
            value = row
            for t in (*data.repack_transforms.inputs, *data.data_transforms.inputs):
                value = t(value)
            if value["prompt"] != PROMPT or not all(value["image_mask"].values()):
                raise ContractError("prompt/camera contract")
            value = transforms.Normalize(data.norm_stats, use_quantiles=data.use_quantile_norm)(value)
            for t in data.model_transforms.inputs:
                value = t(value)
            assert value["actions"].shape == (config.model.action_horizon, 32)
            assert np.asarray(value["tokenized_prompt_mask"]).any()
            restored = transforms.Unnormalize(data.norm_stats, use_quantiles=data.use_quantile_norm)(
                {"state": value["state"], "actions": value["actions"]}
            )
            actual = HV1Outputs(config.policy_metadata["profile"])(restored)["actions"]
            np.testing.assert_allclose(actual, expected, atol=1e-5)
            checked += 1
        offset += len(state)
    assert offset == len(ds)
    report = dict(
        experiment=name,
        frames=len(ds),
        boundary_and_gripper_checks=checked,
        all_three_cameras=True,
        prompt=PROMPT,
        normalization_roundtrip=True,
        episode_crossing=False,
    )
    write_new_json(Path(campaign) / f"contract_check_{name}.json", report)
    print(report)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--campaign", required=True)
    p.add_argument("--experiment", choices=["A", "B", "D"], required=True)
    a = p.parse_args()
    check(a.campaign, a.experiment)
