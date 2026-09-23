"""Official pi05_base configuration for the HV1 TODAY30/ALL59 campaign."""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

from . import augmented
from . import pipeline
from .artifacts import ContractError
from .artifacts import digest
from .artifacts import file_hash
from .artifacts import read_json
from .checkpoints import snapshot_identity
from .native import PROMPT
from .openpi_run import make_config


def local_dataset(data, model, root):
    """Use the explicit local LeRobot export and never fall back to the Hub."""
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

    from openpi import transforms
    from openpi.training import data_loader

    root = Path(root).resolve()
    if not (root / "meta/info.json").is_file():
        raise ContractError("local common export is missing")
    metadata = LeRobotDatasetMetadata(data.repo_id, root=root)
    dataset = LeRobotDataset(
        data.repo_id,
        root=root,
        delta_timestamps={
            key: [t / metadata.fps for t in range(model.action_horizon)] for key in data.action_sequence_keys
        },
    )
    if data.prompt_from_task:
        dataset = data_loader.TransformedDataset(dataset, [transforms.PromptFromLeRobotTask(metadata.tasks)])
    return dataset


def configure(campaign, recipe):
    campaign = Path(campaign).resolve()
    manifest = pipeline.verify_campaign(campaign)
    expected = pipeline.recipe(recipe["name"], manifest["sha256"], recipe["steps"])
    if recipe != expected:
        raise ContractError("recipe differs from the approved two-track plan")
    export = read_json(campaign / "export/export.json")
    if (
        export.get("complete") is not True
        or export.get("manifest_sha256") != manifest["sha256"]
        or export["splits"]["train"]["episode_ids"] != [episode["id"] for episode in manifest["episodes"]]
    ):
        raise ContractError("common export does not match the campaign manifest")
    os.environ["HF_LEROBOT_HOME"] = export["hf_lerobot_home"]
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    config, _ = make_config(campaign / "export/export.json", campaign.parent, recipe["name"], steps=recipe["steps"])
    from openpi.training.config import AssetsConfig
    from openpi.training.optimizer import CosineDecaySchedule
    from openpi.training.weight_loaders import CheckpointWeightLoader

    track = recipe["track"]
    asset_id = f"hv1_{track.lower()}_{manifest['sha256'][:8]}"
    stats_path = campaign / "assets" / asset_id / "norm_stats.json"
    provenance = read_json(campaign / "assets" / asset_id / "provenance.json")
    expected_ids = sorted(manifest["tracks"][track]["episode_ids"])
    if (
        provenance.get("manifest_sha256") != manifest["sha256"]
        or provenance.get("track") != track
        or provenance.get("episode_ids") != expected_ids
        or provenance.get("norm_stats_sha256") != file_hash(stats_path)
    ):
        raise ContractError("track normalization identity mismatch")
    parent_record = None
    if recipe.get("parent"):
        parent = (campaign / recipe["parent"]["snapshot"]).resolve()
        if not parent.is_relative_to(campaign / "snapshots"):
            raise ContractError("filter fine-tune parent escapes campaign snapshots")
        parent_record = snapshot_identity(parent)
        parent_recipe = parent_record.get("recipe", {})
        if (
            parent_recipe.get("name") != recipe["parent"]["experiment"]
            or parent_record.get("step") != recipe["parent"]["step"]
            or parent_record.get("manifest_sha256") != manifest["sha256"]
            or parent_record.get("norm_stats_sha256") != provenance["norm_stats_sha256"]
        ):
            raise ContractError("filter fine-tune parent lineage/statistics mismatch")
        weight_loader = CheckpointWeightLoader(str(parent / "params"))
    else:
        weight_loader = CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params")
    metadata = dict(
        config.policy_metadata,
        campaign_schema=pipeline.SCHEMA,
        profile=manifest["profile"],
        recipe=recipe,
        recipe_sha256=digest(recipe),
        manifest_sha256=manifest["sha256"],
        prompt=PROMPT,
        state_names=manifest["profile"]["state"]["names"],
        camera_dropout=False,
        gripper=manifest["profile"]["gripper"],
        train_episode_ids=expected_ids,
        diagnostic_episode_ids=manifest["diagnostics"],
        norm_stats_sha256=file_hash(stats_path),
        normalization_asset_id=asset_id,
        validation_scope="training_overlap_diagnostics_only",
        initialization=recipe["initialization"],
        parent_snapshot=None if parent_record is None else recipe["parent"]["snapshot"],
        parent_snapshot_sha256=None if parent_record is None else file_hash(parent / "snapshot.json"),
        parent_updates=0 if parent_record is None else parent_record["step"],
        robot_motion_authorized=False,
        export_roots={"train": export["splits"]["train"]["root"]},
        implementation_sha256={
            name: file_hash(Path(__file__).with_name(name))
            for name in (
                "artifacts.py",
                "checkpoints.py",
                "metrics.py",
                "native.py",
                "transforms.py",
                "workflow.py",
                "openpi_run.py",
                "pipeline.py",
                "pipeline_config.py",
                "pipeline_train.py",
                "pipeline_eval.py",
                "pipeline_run.py",
            )
        },
    )
    config = dataclasses.replace(
        config,
        data=dataclasses.replace(
            config.data,
            assets=AssetsConfig(assets_dir=str(campaign / "assets"), asset_id=asset_id),
        ),
        weight_loader=weight_loader,
        freeze_filter=config.model.get_freeze_filter(),
        batch_size=recipe["batch_size"],
        seed=recipe["seed"],
        num_workers=0,
        ema_decay=recipe["ema_decay"],
        checkpoint_base_dir=str(campaign / "restarts"),
        keep_period=None,
        lr_schedule=CosineDecaySchedule(
            warmup_steps=recipe["warmup_steps"],
            peak_lr=recipe["peak_lr"],
            decay_steps=recipe["decay_steps"],
            decay_lr=recipe["decay_lr"],
        ),
        policy_metadata=metadata,
    )
    return config, export


def configure_augmented(root, recipe):
    """Build the same π0.5 policy with a hash-bound real+sim data contract."""
    root = Path(root).resolve()
    index_path = root / "index.json"
    index = augmented.load_augmented_index(index_path, allow_synthetic=True)
    profile = augmented.load_campaign_profile(index["sim_campaign_root"])
    expected_recipe = augmented.augmented_recipe(
        recipe["model"], recipe["stage"], recipe["seed"], profile
    )
    if recipe != expected_recipe:
        raise ContractError("recipe differs from the approved augmented plan")
    schedule_path = root / "schedules" / f"{recipe['name']}.json"
    schedule = pipeline.checked(schedule_path)
    expected_schedule = augmented.augmented_schedule(
        index_path,
        recipe["model"],
        recipe["steps"] * recipe["batch_size"],
        recipe["seed"],
        allow_synthetic=True,
    )
    expected_schedule = pipeline.sealed(
        {key: value for key, value in expected_schedule.items() if key != "sha256"}
        | {"recipe": recipe, "index_sha256": index["sha256"]}
    )
    if schedule != expected_schedule:
        raise ContractError("augmented schedule differs from the approved recipe/index")

    real_export_path = Path(index["real_export"]["export_manifest_path"])
    export = read_json(real_export_path)
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    config, _ = make_config(real_export_path, root.parent, recipe["name"], steps=recipe["steps"])

    from openpi.training.config import AssetsConfig
    from openpi.training.optimizer import CosineDecaySchedule
    from openpi.training.weight_loaders import CheckpointWeightLoader

    asset_id = f"hv1_augmented_{recipe['name'].lower()}_{schedule['sha256'][:8]}"
    asset_root = root / "assets" / asset_id
    stats_path = asset_root / "norm_stats.json"
    provenance = pipeline.checked(asset_root / "provenance.json")
    if (
        provenance.get("schema") != augmented.AUGMENTED_SCHEMA
        or provenance.get("index_sha256") != index["sha256"]
        or provenance.get("schedule_sha256") != schedule["sha256"]
        or provenance.get("recipe") != recipe
        or provenance.get("norm_stats_sha256") != file_hash(stats_path)
    ):
        raise ContractError("augmented normalization identity mismatch")
    metadata = dict(
        config.policy_metadata,
        campaign_schema=augmented.AUGMENTED_SCHEMA,
        profile=index["profile"],
        recipe=recipe,
        recipe_sha256=digest(recipe),
        augmented_index_sha256=index["sha256"],
        augmented_schedule_sha256=schedule["sha256"],
        norm_stats_sha256=file_hash(stats_path),
        normalization_asset_id=asset_id,
        source_mix=recipe["mix"]["source"],
        render_mix=recipe["mix"]["render"],
        pair_fraction=recipe["mix"]["pair_fraction"],
        initialization=recipe["initialization"],
        synthetic=True,
        robot_motion_authorized=False,
        export_roots={"real": index["real_export"]["root"], "sim": index["sim_campaign_root"]},
        implementation_sha256={
            name: file_hash(Path(__file__).with_name(name))
            for name in (
                "artifacts.py",
                "checkpoints.py",
                "native.py",
                "transforms.py",
                "openpi_run.py",
                "pipeline.py",
                "pipeline_config.py",
                "pipeline_train.py",
            )
        },
    )
    config = dataclasses.replace(
        config,
        data=dataclasses.replace(
            config.data,
            assets=AssetsConfig(assets_dir=str(root / "assets"), asset_id=asset_id),
        ),
        weight_loader=CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        freeze_filter=config.model.get_freeze_filter(),
        batch_size=recipe["batch_size"],
        seed=recipe["seed"],
        num_workers=0,
        ema_decay=None,
        checkpoint_base_dir=str(root / "restarts"),
        keep_period=None,
        lr_schedule=CosineDecaySchedule(
            warmup_steps=recipe["warmup_steps"],
            peak_lr=recipe["peak_lr"],
            decay_steps=recipe["decay_steps"],
            decay_lr=recipe["decay_lr"],
        ),
        policy_metadata=metadata,
    )
    return config, index, schedule
