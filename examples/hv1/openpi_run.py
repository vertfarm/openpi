"""Explicit HV1 JAX data/stats/train/serve entrypoints; no robot commands."""

import argparse
import os
from pathlib import Path
import re
import shutil

from .artifacts import ContractError
from .artifacts import digest
from .artifacts import read_json
from .artifacts import write_new_json
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


def check_data(config):
    """Exercise the official loader/action chunks without loading model weights."""
    import numpy as np

    from openpi import transforms
    from openpi.training import data_loader

    data = config.data.create(config.assets_dirs, config.model)
    ds = data_loader.create_torch_dataset(data, config.model.action_horizon, config.model)
    sample = ds[0]
    expected_actions = np.asarray(sample["action"]).copy()
    for transform in (*data.repack_transforms.inputs, *data.data_transforms.inputs):
        sample = transform(sample)
    if sample["actions"].shape != (config.model.action_horizon, len(config.policy_metadata["action_names"])):
        raise ContractError("official loader action chunk shape mismatch")
    if not np.isfinite(sample["actions"]).all():
        raise ContractError("invalid loader actions")
    if not str(sample.get("prompt", "")).strip():
        raise ContractError("task prompt lost during training transforms")
    end_checks = 0
    for i in range(len(ds)):
        row = ds[i]
        if i + 1 == len(ds) or int(row["episode_index"]) != int(ds[i + 1]["episode_index"]):
            chunk = np.asarray(row["action"])
            np.testing.assert_allclose(chunk, np.repeat(chunk[:1], len(chunk), axis=0), atol=1e-6)
            end_checks += 1
    model_input = dict(sample)
    if data.norm_stats is not None:
        model_input = transforms.Normalize(data.norm_stats, use_quantiles=data.use_quantile_norm)(model_input)
    for transform in data.model_transforms.inputs:
        model_input = transform(model_input)
    if model_input["actions"].shape != (config.model.action_horizon, 32):
        raise ContractError("model padding shape mismatch")
    if not np.asarray(model_input["tokenized_prompt_mask"]).any():
        raise ContractError("empty language tokens")
    if data.norm_stats is not None:
        restored = transforms.Unnormalize(data.norm_stats, use_quantiles=data.use_quantile_norm)(
            {"state": model_input["state"], "actions": model_input["actions"]}
        )
        restored = HV1Outputs(config.data.data_transforms(config.model).inputs[0].profile)(restored)
        np.testing.assert_allclose(restored["actions"], expected_actions, atol=1e-5)
    return {
        "frames": len(ds),
        "actions_shape": list(sample["actions"].shape),
        "state_shape": list(sample["state"].shape),
        "image_slots": list(sample["image"]),
        "episode_end_chunk_checks": end_checks,
        "prompt_tokens_checked": True,
        "normalization_roundtrip_checked": data.norm_stats is not None,
        "model_loaded": False,
    }


def compute_stats(config):
    import numpy as np

    from openpi.shared import normalize
    from openpi.training import data_loader

    data = config.data.create(config.assets_dirs, config.model)
    # Train repo is physically separate from validation. Uses official chunk sampling.
    ds = data_loader.create_torch_dataset(data, config.model.action_horizon, config.model)
    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    for index in range(len(ds)):
        sample = ds[index]
        for transform in (*data.repack_transforms.inputs, *data.data_transforms.inputs):
            sample = transform(sample)
        for key, running in stats.items():
            array = np.asarray(sample[key])
            running.update(array.reshape(-1, array.shape[-1]))
    output = config.assets_dirs / data.repo_id
    if (output / "norm_stats.json").exists():
        raise ContractError("normalization stats exist; use a new dataset manifest")
    normalize.save(output, {key: value.get_statistics() for key, value in stats.items()})
    return {"stats_path": str(output), "training_frames": len(ds), "validation_used": False}


def main():
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check-data", "stats", "train", "serve"))
    parser.add_argument("--export", required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--experiment", default="smoke")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--allow-synthetic", action="store_true")
    parser.add_argument("--allow-gpu-run", action="store_true")
    parser.add_argument("--checkpoint")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not 1 <= args.steps <= 100000:
        parser.error("steps must be 1..100000")
    config, export = make_config(args.export, args.runtime, args.experiment, steps=args.steps)
    if export["profile"]["status"] == "synthetic" and not args.allow_synthetic:
        parser.error("synthetic fixture requires --allow-synthetic")
    if args.command == "check-data":
        print(json.dumps(check_data(config)))
        return
    if args.command == "stats":
        print(json.dumps(compute_stats(config)))
        return
    if not args.allow_gpu_run:
        parser.error("train/serve requires explicit --allow-gpu-run; this does NOT authorize robot motion")
    runtime = Path(args.runtime).resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(runtime).free < 50 * 1024**3:
        raise ContractError("less than 50 GiB disk headroom; storage review required (no automatic cleanup)")
    from filelock import FileLock

    with FileLock(str(runtime / "hv1-ml-gpu.lock"), timeout=0):
        if args.command == "train":
            from scripts.train import main as train

            if config.data.create(config.assets_dirs, config.model).norm_stats is None:
                raise ContractError("compute train-only normalization stats first")
            run_manifest = runtime / "runs" / f"{args.experiment}.json"
            write_new_json(run_manifest, config.policy_metadata)
            train(config)
        else:
            from openpi.policies.policy_config import create_trained_policy
            from openpi.serving.websocket_policy_server import WebsocketPolicyServer

            if not args.checkpoint:
                parser.error("--checkpoint required for serve")
            checkpoint = Path(args.checkpoint).resolve()
            if not checkpoint.is_relative_to(config.checkpoint_dir.resolve()) or not checkpoint.is_dir():
                raise ContractError("checkpoint must belong to the selected HV1 experiment")
            if read_json(runtime / "runs" / f"{args.experiment}.json") != config.policy_metadata:
                raise ContractError("run metadata/profile mismatch")
            policy = create_trained_policy(config, checkpoint)
            WebsocketPolicyServer(policy, host="127.0.0.1", port=args.port, metadata=policy.metadata).serve_forever()


if __name__ == "__main__":
    main()
