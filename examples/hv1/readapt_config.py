"""Reviewed N/M configuration and local dataset loading, without a trainer import."""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

from . import readapt
from .artifacts import ContractError
from .artifacts import digest
from .artifacts import file_hash
from .artifacts import read_json
from .native import PROMPT
from .openpi_run import make_config


def configure(campaign, r):
    campaign = Path(campaign).resolve()
    manifest = readapt.verify_campaign(campaign)
    if r != readapt.recipe(r["name"], manifest["sha256"]):
        raise ContractError("recipe differs from approved N/M plan")
    export = read_json(campaign / "export/export.json")
    export_scan = read_json(campaign / "export_scan.json")
    if (
        export["manifest_sha256"] != export_scan["manifest_sha256"]
        or export_scan["manifest_sha256"] != digest({k: v for k, v in export_scan.items() if k != "manifest_sha256"})
        or export_scan["episodes"] != manifest["episodes"]
    ):
        raise ContractError("export/manifest identity mismatch")
    for split in ("train", "validation"):
        wanted = [e["id"] for e in manifest["episodes"] if e["validation"] == (split == "validation")]
        if export["splits"][split]["episode_ids"] != wanted:
            raise ContractError("export episode order/split mismatch")
        item = export["splits"][split]
        if Path(item["root"]).resolve() != (campaign / "export" / item["repo_id"]).resolve():
            raise ContractError("export root does not match this campaign")
    os.environ["HF_LEROBOT_HOME"] = export["hf_lerobot_home"]
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    config, _ = make_config(campaign / "export/export.json", campaign.parent, r["name"], steps=r["steps"])
    from openpi.training.config import AssetsConfig
    from openpi.training.optimizer import CosineDecaySchedule
    from openpi.training.weight_loaders import CheckpointWeightLoader

    parent = manifest["parent"]
    selected = [e for e in manifest["episodes"] if not e["validation"] and (r["name"] == "M" or e["cohort"] == "new")]
    metadata = dict(
        config.policy_metadata,
        campaign_schema=readapt.SCHEMA,
        profile=manifest["profile"],
        recipe=r,
        recipe_sha256=digest(r),
        manifest_sha256=manifest["sha256"],
        parent=parent,
        prompt=PROMPT,
        state_names=manifest["profile"]["state"]["names"],
        camera_dropout=False,
        gripper=manifest["profile"]["gripper"],
        train_episode_ids=[e["id"] for e in selected],
        validation_episode_ids={
            c: [e["id"] for e in manifest["episodes"] if e["validation"] and e["cohort"] == c] for c in ("new", "old")
        },
        norm_stats_sha256=parent["norm_stats_sha256"],
        validation_scope="separate_new_and_old_sessions",
        initialization="BF16_C5000_weights_FP32_training_new_optimizer",
        robot_motion_authorized=False,
        export_roots={s: export["splits"][s]["root"] for s in ("train", "validation")},
        implementation_sha256={
            name: file_hash(Path(__file__).with_name(name))
            for name in (
                "artifacts.py",
                "checkpoints.py",
                "readapt.py",
                "readapt_config.py",
                "readapt_train.py",
                "readapt_eval.py",
                "readapt_run.py",
                "openpi_run.py",
                "native.py",
                "transforms.py",
                "workflow.py",
            )
        },
    )
    config = dataclasses.replace(
        config,
        data=dataclasses.replace(
            config.data,
            assets=AssetsConfig(
                assets_dir=str(Path(parent["norm_stats"]).parent.parent), asset_id="hv1_common_clean19"
            ),
        ),
        weight_loader=CheckpointWeightLoader(str(Path(parent["snapshot"]) / "params")),
        batch_size=r["batch_size"],
        seed=r["seed"],
        num_workers=0,
        ema_decay=r["ema_decay"],
        checkpoint_base_dir=str(campaign / "restarts"),
        keep_period=None,
        lr_schedule=CosineDecaySchedule(
            warmup_steps=r["warmup_steps"], peak_lr=r["peak_lr"], decay_steps=r["decay_steps"], decay_lr=r["decay_lr"]
        ),
        policy_metadata=metadata,
    )
    return config, export


def local_dataset(data, model, root):
    """Official LeRobot/chunk/prompt semantics with an explicit local root.

    Avoid LeRobot's import-time HF_LEROBOT_HOME constant and any Hub fallback.
    """
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

    from openpi import transforms
    from openpi.training import data_loader

    root = Path(root).resolve()
    if not (root / "meta/info.json").is_file():
        raise ContractError("local exported dataset missing; do not query the Hub")
    meta = LeRobotDatasetMetadata(data.repo_id, root=root)
    dataset = LeRobotDataset(
        data.repo_id,
        root=root,
        delta_timestamps={k: [t / meta.fps for t in range(model.action_horizon)] for k in data.action_sequence_keys},
    )
    if data.prompt_from_task:
        dataset = data_loader.TransformedDataset(dataset, [transforms.PromptFromLeRobotTask(meta.tasks)])
    return dataset
