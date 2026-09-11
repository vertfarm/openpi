"""The official openpi TrainConfig for HV1, built from an export manifest.

A library, not a runnable step. Its data/stats/train/serve entry points belonged
to the 2026-09-09 generation and had no callers left, so they went on
2026-09-11; `pipeline_config.configure` is what builds on `make_config` now.
No robot commands.
"""

import os
from pathlib import Path
import re

from .artifacts import ContractError
from .artifacts import digest
from .artifacts import read_json
from .transforms import HV1Inputs
from .transforms import HV1Outputs
from .workflow import validate_profile


def make_config(export_path, runtime, experiment, *, steps=20):
    export = read_json(export_path)
    p = validate_profile(export["profile"])
    if export.get("complete") is not True or digest(p) != export["profile_sha256"]:
        raise ContractError("incomplete export or profile integrity mismatch")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", experiment):
        raise ContractError("experiment must be a simple new name")
    runtime = Path(runtime).resolve()
    os.environ["HF_LEROBOT_HOME"] = export["hf_lerobot_home"]
    os.environ["HF_HOME"] = str(runtime / "cache" / "huggingface")
    os.environ["OPENPI_DATA_HOME"] = str(runtime / "cache" / "openpi")
    os.environ["WANDB_MODE"] = "disabled"
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    from openpi import transforms
    from openpi.models.pi0_config import Pi0Config
    from openpi.training.config import AssetsConfig
    from openpi.training.config import DataConfig
    from openpi.training.config import SimpleDataConfig
    from openpi.training.config import TrainConfig
    from openpi.training.weight_loaders import CheckpointWeightLoader

    train_repo = export["splits"]["train"]["repo_id"]
    config = TrainConfig(
        name="pi05_hv1",
        exp_name=experiment,
        project_name="hv1",
        seed=42,
        model=Pi0Config(pi05=True, action_dim=32, action_horizon=p["action_horizon"]),
        data=SimpleDataConfig(
            repo_id=train_repo,
            assets=AssetsConfig(asset_id=train_repo),
            base_config=DataConfig(
                action_sequence_keys=("action",),
                prompt_from_task=True,
                repack_transforms=transforms.Group(
                    inputs=[
                        transforms.RepackTransform(
                            {
                                "images": {slot: f"observation.images.{slot}" for slot in p["images"]},
                                "state": "observation.state",
                                "actions": "action",
                                "prompt": "prompt",
                            }
                        )
                    ]
                ),
            ),
            data_transforms=lambda model: transforms.Group(inputs=[HV1Inputs(p)], outputs=[HV1Outputs(p)]),
        ),
        weight_loader=CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        assets_base_dir=str(runtime / "assets"),
        checkpoint_base_dir=str(runtime / "checkpoints"),
        batch_size=1,
        num_workers=0,
        num_train_steps=steps,
        log_interval=1,
        save_interval=max(1, steps),
        keep_period=None,
        wandb_enabled=False,
        overwrite=False,
        resume=False,
        policy_metadata={
            "model": "pi05_hv1",
            "profile_sha256": digest(p),
            "manifest_sha256": export["manifest_sha256"],
            "action_horizon": p["action_horizon"],
            "action_names": p["action"]["names"],
            "action_units": p["action"]["units"],
            "output_action_space": "commanded_target",
            "synthetic": p["status"] == "synthetic",
        },
    )
    return config, export
