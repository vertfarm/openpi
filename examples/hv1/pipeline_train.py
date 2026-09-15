"""Inference-snapshot-first training for the official-base HV1 two-track campaign."""

from __future__ import annotations

import argparse
import dataclasses
from datetime import datetime
import functools
import io
import json
from pathlib import Path
import signal
import time
import zlib

import numpy as np

from . import pipeline
from . import native
from .artifacts import GIB
from .artifacts import ContractError
from .artifacts import atomic_json
from .artifacts import digest
from .artifacts import file_hash
from .artifacts import read_json
from .artifacts import sealed
from .artifacts import storage_gate
from .artifacts import write_new_json
from .checkpoints import save_snapshot
from .checkpoints import snapshot_identity
from .pipeline_config import configure
from .pipeline_config import configure_augmented
from .pipeline_config import local_dataset
from .transforms import validate_augmented_sample


@dataclasses.dataclass(frozen=True)
class StateNoise:
    """Corrupt the state input during training so the images have to carry the task.

    The policy reads state as a phase clock. Actions are deltas on it, and with
    every recorded approach mutually cosine 0.89-0.99, arm pose alone says where
    in the episode it is - so "the mean delta for this phase" collects most of
    the loss and the cameras are never needed. This degrades that clock.

    It runs *before* `HV1Inputs` on purpose. `HV1Inputs` computes the action
    delta against the state, so noising afterwards would anchor the input and
    the target differently and turn the perturbation into irreducible label
    noise. Applied here, the target moves with the input and stays recoverable.

    The draw is a deterministic function of the sample, so a run reproduces and
    a resume does not silently train on a different dataset.

    Training only. `make_loader` builds it into a copy of the data config;
    `pipeline_config.configure` returns the config deployment shares, and that
    one never carries this. Putting it there would noise live inference.
    """

    sigma: tuple[float, ...]
    seed: int

    def __call__(self, data):
        state = np.asarray(data["state"], dtype=np.float32)
        scale = np.asarray(self.sigma, dtype=np.float32)
        if state.shape != scale.shape or not np.isfinite(state).all():
            raise ContractError("state noise scale/state shape mismatch")
        rng = np.random.default_rng([self.seed, zlib.crc32(state.tobytes())])
        noise = rng.normal(size=state.shape).astype(np.float32) * scale
        return {**data, "state": state + noise}


def _noisy_data_config(data, recipe):
    """Prepend `StateNoise` when the recipe asks for it, else hand back `data`."""
    sigma = float(recipe.get("state_noise_sigma", 0.0) or 0.0)
    if sigma <= 0:
        return data, None
    stats = (data.norm_stats or {}).get("state")
    if stats is None or stats.std is None:
        raise ContractError("state noise needs the track's normalization statistics")
    # Scaled per channel by the training spread, so one sigma means the same
    # thing on a joint that moves a radian and one that moves six milliradians.
    scale = tuple(float(sigma * value) for value in np.asarray(stats.std, dtype=np.float64))
    noise = StateNoise(sigma=scale, seed=recipe["seed"])
    group = data.data_transforms
    return dataclasses.replace(data, data_transforms=dataclasses.replace(group, inputs=(noise, *group.inputs))), noise


class FixedSampler:
    def __init__(self, schedule, consumed=0):
        if consumed < 0 or consumed > len(schedule["records"]):
            raise ContractError("sampler cursor outside the approved schedule")
        self.indices = [row["index"] for row in schedule["records"][consumed:]]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class AugmentedDataset:
    """Read an unchanged real LeRobot export plus local, verified anchor tar shards."""

    def __init__(self, index_path, model, *, allow_synthetic=False, real_dataset=None):
        if not allow_synthetic:
            raise ContractError("augmented dataset requires explicit --allow-synthetic")
        self.index = pipeline.load_augmented_index(index_path, allow_synthetic=True)
        self.sim_root = Path(self.index["sim_campaign_root"]).resolve()
        if real_dataset is None:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
            from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

            real = self.index["real_export"]
            metadata = LeRobotDatasetMetadata(real["repo_id"], root=real["root"])
            real_dataset = LeRobotDataset(
                real["repo_id"],
                root=real["root"],
                delta_timestamps={
                    "action": [step / metadata.fps for step in range(model.action_horizon)]
                },
            )
        self.real_dataset = real_dataset

    def __len__(self):
        return len(self.index["records"])

    @staticmethod
    def _real_sample(raw, metadata):
        return {
            **metadata,
            "state": raw["observation.state"],
            "images": {
                camera: raw[f"observation.images.{camera}"] for camera in native.CAMERAS
            },
            "actions": raw["action"],
            "language": metadata["language"],
        }

    def _sim_sample(self, data_ref):
        shard = (self.sim_root / data_ref["shard_relative_path"]).resolve()
        if not shard.is_relative_to(self.sim_root):
            raise ContractError("sim shard path escapes campaign root")
        from PIL import Image

        members = data_ref["members"]
        shard_size = shard.stat().st_size

        def read_member(stream, name):
            descriptor = members[name]
            offset, size = int(descriptor["offset"]), int(descriptor["size"])
            if offset < 0 or size < 0 or offset + size > shard_size:
                raise ContractError("sim anchor tar member range is invalid")
            stream.seek(offset)
            payload = stream.read(size)
            if len(payload) != size:
                raise ContractError("sim anchor tar member is truncated")
            return payload

        with shard.open("rb") as stream:
            sample = json.loads(read_member(stream, "metadata"))
            sample["images"] = {}
            for camera in native.CAMERAS:
                with Image.open(io.BytesIO(read_member(stream, camera))) as image:
                    sample["images"][camera] = np.asarray(image.convert("RGB"), dtype=np.uint8)
        return sample

    def __getitem__(self, index):
        record = self.index["records"][int(index)]
        ref = record["data_ref"]
        if ref["kind"] == "real_lerobot":
            sample = self._real_sample(self.real_dataset[ref["index"]], record["metadata"])
        elif ref["kind"] == "anchor_tar":
            sample = self._sim_sample(ref)
        else:
            raise ContractError("unknown augmented data reference")
        validate_augmented_sample(sample, allow_synthetic=True)
        return {
            "observation.state": sample["state"],
            **{
                f"observation.images.{camera}": sample["images"][camera]
                for camera in native.CAMERAS
            },
            "action": sample["actions"],
            "prompt": sample["language"],
        }


class AugmentedScheduleSampler:
    def __init__(self, schedule, index_sha256, consumed=0):
        if schedule.get("schema") != pipeline.AUGMENTED_SCHEMA:
            raise ContractError("unsupported augmented schedule")
        if schedule.get("index_sha256") != index_sha256:
            raise ContractError("schedule/index identity mismatch")
        if consumed < 0 or consumed > len(schedule["records"]):
            raise ContractError("augmented sampler cursor outside schedule")
        self.indices = [row["record_index"] for row in schedule["records"][consumed:]]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def make_augmented_loader(config, index_path, schedule, sharding, *, consumed=0, allow_synthetic=False):
    """Build a local-only OpenPI loader; NAS paths are never read by the trainer."""
    if not allow_synthetic:
        raise ContractError("augmented training requires explicit --allow-synthetic")
    from openpi.training import data_loader

    data = config.data.create(config.assets_dirs, config.model)
    dataset = AugmentedDataset(index_path, config.model, allow_synthetic=True)
    sampler = AugmentedScheduleSampler(schedule, dataset.index["sha256"], consumed)
    dataset = data_loader.transform_dataset(dataset, data)
    loader = data_loader.TorchDataLoader(
        dataset,
        local_batch_size=config.batch_size,
        sharding=sharding,
        shuffle=False,
        sampler=sampler,
        num_batches=len(sampler) // config.batch_size,
        num_workers=0,
        seed=config.seed,
    )
    return data_loader.DataLoaderImpl(data, loader)


def compute_augmented_statistics(root, model, stage, seed, *, allow_synthetic=False):
    """Compute state/action statistics from the exact scheduled real+sim mixture."""
    if not allow_synthetic:
        raise ContractError("augmented statistics require explicit --allow-synthetic")
    root = Path(root).resolve()
    recipe = pipeline.augmented_recipe(model, stage, seed)
    index_path = root / "index.json"
    index = pipeline.load_augmented_index(index_path, allow_synthetic=True)
    schedule = pipeline.checked(root / "schedules" / f"{recipe['name']}.json")
    if schedule.get("recipe") != recipe or schedule.get("index_sha256") != index["sha256"]:
        raise ContractError("augmented statistics schedule lineage mismatch")
    dataset = AugmentedDataset(
        index_path,
        type("ModelContract", (), {"action_horizon": 15})(),
        allow_synthetic=True,
    )
    from openpi.shared import normalize

    running = {key: normalize.RunningStats() for key in ("state", "actions")}
    delta_indices = index["profile"]["action"]["delta_state_indices"]
    for row in schedule["records"]:
        raw = dataset[row["record_index"]]
        state = np.asarray(raw["observation.state"], dtype=np.float32)
        actions = np.asarray(raw["action"], dtype=np.float32).copy()
        for action_index, state_index in enumerate(delta_indices):
            if state_index >= 0:
                actions[:, action_index] -= state[state_index]
        running["state"].update(state[None, :])
        running["actions"].update(actions)
    asset_id = f"hv1_augmented_{recipe['name'].lower()}_{schedule['sha256'][:8]}"
    asset_root = root / "assets" / asset_id
    if asset_root.exists():
        raise ContractError("augmented normalization asset already exists")
    normalize.save(asset_root, {key: value.get_statistics() for key, value in running.items()})
    provenance = sealed(
        {
            "schema": pipeline.AUGMENTED_SCHEMA,
            "index_sha256": index["sha256"],
            "schedule_sha256": schedule["sha256"],
            "recipe": recipe,
            "scheduled_samples": len(schedule["records"]),
            "coverage": pipeline.augmented_coverage(schedule, len(schedule["records"])),
            "normalization_scope": "exact_scheduled_training_mixture",
            "norm_stats_sha256": file_hash(asset_root / "norm_stats.json"),
            "robot_motion_authorized": False,
        }
    )
    write_new_json(asset_root / "provenance.json", provenance)
    return provenance


def make_loader(config, schedule, sharding, consumed=0):
    from openpi.training import data_loader

    data = config.data.create(config.assets_dirs, config.model)
    dataset = local_dataset(data, config.model, config.policy_metadata["export_roots"]["train"])
    if max(row["index"] for row in schedule["records"]) >= len(dataset):
        raise ContractError("sampler index is outside the common export")
    data, noise = _noisy_data_config(data, config.policy_metadata["recipe"])
    if noise is not None:
        print(json.dumps(dict(state_noise_sigma=config.policy_metadata["recipe"]["state_noise_sigma"])), flush=True)
    dataset = data_loader.transform_dataset(dataset, data)
    loader = data_loader.TorchDataLoader(
        dataset,
        local_batch_size=config.batch_size,
        sharding=sharding,
        shuffle=False,
        sampler=FixedSampler(schedule, consumed),
        num_batches=(len(schedule["records"]) - consumed) // config.batch_size,
        num_workers=0,
        seed=config.seed,
    )
    return data_loader.DataLoaderImpl(data, loader)


def train(campaign, name, deadline, *, resume=False):
    campaign = Path(campaign).resolve()
    manifest = pipeline.verify_campaign(campaign, raw=True)
    steps = 50 if name == "SMOKE" else 1000 if name in pipeline.FILTER_FINETUNES else pipeline.TARGET_STEPS
    recipe = pipeline.recipe(name, manifest["sha256"], steps)
    schedule = pipeline.checked(campaign / f"sampler_{name}.json")
    expected = pipeline.sample_schedule(manifest, name, recipe["steps"] * recipe["batch_size"])
    if schedule != expected:
        raise ContractError("sampler schedule changed")
    stop_at = datetime.fromisoformat(deadline)
    if stop_at.tzinfo is None or stop_at.timestamp() <= time.time():
        raise ContractError("future timezone-aware deadline required")
    config, _ = configure(campaign, recipe)
    run = campaign / "runs" / name
    contract = run / "recipe.json"
    if contract.exists():
        if not resume or read_json(contract) != recipe:
            raise ContractError("existing run requires an explicit compatible resume")
    else:
        if resume:
            raise ContractError("cannot resume a run that has never started")
        write_new_json(contract, recipe)
    identity = dict(
        recipe_sha256=digest(recipe),
        manifest_sha256=manifest["sha256"],
        sampler_sha256=schedule["sha256"],
    )
    progress_path = run / "progress.json"
    if resume and progress_path.is_file():
        progress = read_json(progress_path)
        snapshots = [campaign / "snapshots" / name / f"step_{step:06d}" for step in recipe["snapshots"]]
        records = [snapshot_identity(snapshot) for snapshot in snapshots if snapshot.is_dir()]
        if (
            progress.get("step") == recipe["steps"]
            and len(records) == len(snapshots)
            and [record["step"] for record in records] == recipe["snapshots"]
            and all(record.get("recipe") == recipe for record in records)
        ):
            result = dict(
                name=name,
                track=recipe["track"],
                step=recipe["steps"],
                target=recipe["steps"],
                complete=True,
                reason="target_reached_restart_skipped_storage",
                elapsed_seconds=None,
                seconds_per_update=progress["seconds_per_update"],
                snapshots=[str(snapshot) for snapshot in snapshots],
                coverage=progress["coverage"],
                identity=identity,
                restart_path=None,
                exact_optimizer_resume_available=False,
                recovery_evidence="target progress and every planned inference snapshot hash verified",
                robot_commands_sent=0,
            )
            atomic_json(run / "result.json", result)
            return result
    storage_gate(campaign, (16 if name in pipeline.FILTER_FINETUNES else 40) * GIB)
    from filelock import FileLock
    import jax

    from openpi.training import checkpoints
    from openpi.training import sharding
    from scripts.train import init_train_state
    from scripts.train import train_step

    stopped = []
    signal.signal(signal.SIGINT, lambda *_: stopped.append("SIGINT"))
    signal.signal(signal.SIGTERM, lambda *_: stopped.append("SIGTERM"))
    durations = []
    metrics = {}
    started = time.monotonic()
    jax.config.update("jax_compilation_cache_dir", str(campaign.parent / "cache/jax"))
    with FileLock(str(campaign.parent / "hv1-ml-gpu.lock"), timeout=0):
        mesh = sharding.make_mesh(config.fsdp_devices)
        batch_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
        loader = make_loader(config, schedule, batch_sharding)
        manager, resuming = checkpoints.initialize_checkpoint_dir(
            config.checkpoint_dir,
            keep_period=None,
            overwrite=False,
            resume=resume,
        )
        try:
            train_rng, init_rng = jax.random.split(jax.random.key(config.seed))
            state, state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
            if resuming:
                state = checkpoints.restore_state(manager, state, loader)
            jax.block_until_ready(state)
            initial_step = int(state.step)
            if resuming:
                cursor = read_json(run / f"sampler_cursor_{initial_step:06d}.json")
                if cursor != dict(identity, consumed_samples=initial_step * config.batch_size):
                    raise ContractError("restart and sampler cursor differ")
                loader = make_loader(config, schedule, batch_sharding, initial_step * config.batch_size)
            iterator = iter(loader)
            replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
            compiled = jax.jit(
                functools.partial(train_step, config),
                in_shardings=(replicated, state_sharding, batch_sharding),
                out_shardings=(state_sharding, replicated),
                donate_argnums=(1,),
            )
            reason = "target_reached"
            while int(state.step) < recipe["steps"]:
                if stopped or time.time() >= stop_at.timestamp():
                    reason = stopped[0] if stopped else "deadline"
                    break
                storage_gate(campaign)
                before = time.monotonic()
                batch = next(iterator)
                with sharding.set_mesh(mesh):
                    state, info = compiled(train_rng, state, batch)
                info = jax.device_get(info)
                if not all(np.isfinite(value).all() for value in jax.tree.leaves(info)):
                    raise ContractError("nonfinite training metrics")
                durations.append(time.monotonic() - before)
                step = int(state.step)
                metrics = {key: float(value) for key, value in info.items()}
                if step == 1 or step % 50 == 0:
                    progress = dict(
                        step=step,
                        target=recipe["steps"],
                        metrics=metrics,
                        coverage=pipeline.coverage(schedule, step * config.batch_size),
                        seconds_per_update=float(np.median(durations[-30:])),
                        initialization=recipe["initialization"],
                        robot_commands_sent=0,
                    )
                    atomic_json(run / "progress.json", progress)
                    print(json.dumps(progress), flush=True)
                if step in recipe["snapshots"]:
                    save_snapshot(campaign, config, state, loader, step, metrics)
            step = int(state.step)
            restart_path = None
            restart_skip_reason = None
            if step > initial_step and name not in pipeline.FILTER_FINETUNES:
                try:
                    storage_gate(campaign, 34 * GIB)
                except ContractError as error:
                    if step != recipe["steps"]:
                        raise
                    restart_skip_reason = str(error)
                else:
                    cursor_value = dict(identity, consumed_samples=step * config.batch_size)
                    atomic_json(run / f"sampler_cursor_{step:06d}.json", cursor_value)
                    checkpoints.save_state(manager, state, loader, step)
                    manager.wait_until_finished()
                    restart_path = str(config.checkpoint_dir / str(step))
            elif step > initial_step:
                restart_skip_reason = "inference-only filter fine-tune; optimizer state intentionally not retained"
            if restart_skip_reason and step == recipe["steps"]:
                reason = "target_reached_restart_skipped_storage"
            elif restart_skip_reason:
                reason = f"{reason}_without_optimizer_resume"
            result = dict(
                name=name,
                track=recipe["track"],
                step=step,
                target=recipe["steps"],
                complete=step == recipe["steps"],
                reason=reason,
                elapsed_seconds=time.monotonic() - started,
                seconds_per_update=None if not durations else float(np.median(durations[-30:])),
                snapshots=[
                    str(campaign / "snapshots" / name / f"step_{snapshot:06d}")
                    for snapshot in recipe["snapshots"]
                    if snapshot <= step
                ],
                coverage=pipeline.coverage(schedule, step * config.batch_size),
                identity=identity,
                restart_path=restart_path,
                exact_optimizer_resume_available=restart_path is not None,
                restart_skip_reason=restart_skip_reason,
                robot_commands_sent=0,
            )
            atomic_json(run / "result.json", result)
            return result
        finally:
            manager.wait_until_finished()
            manager.close()


def train_augmented(root, model, stage, seed, deadline, *, resume=False, allow_synthetic=False):
    """Train one hash-bound M0--M4 run without reading shards from NAS."""
    if not allow_synthetic:
        raise ContractError("augmented training requires explicit --allow-synthetic")
    root = Path(root).resolve()
    recipe = pipeline.augmented_recipe(model, stage, seed)
    index = pipeline.load_augmented_index(root / "index.json", allow_synthetic=True)
    schedule = pipeline.checked(root / "schedules" / f"{recipe['name']}.json")
    if schedule.get("recipe") != recipe or schedule.get("index_sha256") != index["sha256"]:
        raise ContractError("augmented run schedule lineage mismatch")
    stop_at = datetime.fromisoformat(deadline)
    if stop_at.tzinfo is None or stop_at.timestamp() <= time.time():
        raise ContractError("future timezone-aware deadline required")
    config, _, configured_schedule = configure_augmented(root, recipe)
    if configured_schedule != schedule:
        raise ContractError("configured augmented schedule changed")
    run = root / "runs" / recipe["name"]
    contract = run / "recipe.json"
    if contract.exists():
        if not resume or read_json(contract) != recipe:
            raise ContractError("existing augmented run requires an explicit compatible resume")
    else:
        if resume:
            raise ContractError("cannot resume an augmented run that has never started")
        write_new_json(contract, recipe)
    identity = {
        "recipe_sha256": digest(recipe),
        "index_sha256": index["sha256"],
        "sampler_sha256": schedule["sha256"],
    }
    progress_path = run / "progress.json"
    if resume and progress_path.is_file():
        progress = read_json(progress_path)
        snapshots = [root / "snapshots" / recipe["name"] / f"step_{step:06d}" for step in recipe["snapshots"]]
        records = [snapshot_identity(snapshot) for snapshot in snapshots if snapshot.is_dir()]
        if (
            progress.get("step") == recipe["steps"]
            and len(records) == len(snapshots)
            and [record["step"] for record in records] == recipe["snapshots"]
            and all(record.get("recipe") == recipe for record in records)
        ):
            result = {
                "name": recipe["name"],
                "model": model,
                "stage": stage,
                "seed": seed,
                "step": recipe["steps"],
                "target": recipe["steps"],
                "complete": True,
                "reason": "target_reached_restart_skipped_storage",
                "elapsed_seconds": None,
                "seconds_per_update": progress["seconds_per_update"],
                "snapshots": [str(snapshot) for snapshot in snapshots],
                "coverage": progress["coverage"],
                "identity": identity,
                "restart_path": None,
                "exact_optimizer_resume_available": False,
                "robot_commands_sent": 0,
            }
            atomic_json(run / "result.json", result)
            return result
    storage_gate(root, 40 * GIB)
    from filelock import FileLock
    import jax

    from openpi.training import checkpoints
    from openpi.training import sharding
    from scripts.train import init_train_state
    from scripts.train import train_step

    stopped = []
    signal.signal(signal.SIGINT, lambda *_: stopped.append("SIGINT"))
    signal.signal(signal.SIGTERM, lambda *_: stopped.append("SIGTERM"))
    durations = []
    metrics = {}
    started = time.monotonic()
    jax.config.update("jax_compilation_cache_dir", str(root.parent / "cache/jax"))
    with FileLock(str(root.parent / "hv1-ml-gpu.lock"), timeout=0):
        mesh = sharding.make_mesh(config.fsdp_devices)
        batch_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
        loader = make_augmented_loader(
            config,
            root / "index.json",
            schedule,
            batch_sharding,
            allow_synthetic=True,
        )
        manager, resuming = checkpoints.initialize_checkpoint_dir(
            config.checkpoint_dir,
            keep_period=None,
            overwrite=False,
            resume=resume,
        )
        try:
            train_rng, init_rng = jax.random.split(jax.random.key(config.seed))
            state, state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
            if resuming:
                state = checkpoints.restore_state(manager, state, loader)
            jax.block_until_ready(state)
            initial_step = int(state.step)
            if resuming:
                cursor = read_json(run / f"sampler_cursor_{initial_step:06d}.json")
                if cursor != dict(identity, consumed_samples=initial_step * config.batch_size):
                    raise ContractError("augmented restart and sampler cursor differ")
                loader = make_augmented_loader(
                    config,
                    root / "index.json",
                    schedule,
                    batch_sharding,
                    consumed=initial_step * config.batch_size,
                    allow_synthetic=True,
                )
            iterator = iter(loader)
            replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
            compiled = jax.jit(
                functools.partial(train_step, config),
                in_shardings=(replicated, state_sharding, batch_sharding),
                out_shardings=(state_sharding, replicated),
                donate_argnums=(1,),
            )
            reason = "target_reached"
            while int(state.step) < recipe["steps"]:
                if stopped or time.time() >= stop_at.timestamp():
                    reason = stopped[0] if stopped else "deadline"
                    break
                storage_gate(root)
                before = time.monotonic()
                batch = next(iterator)
                with sharding.set_mesh(mesh):
                    state, info = compiled(train_rng, state, batch)
                info = jax.device_get(info)
                if not all(np.isfinite(value).all() for value in jax.tree.leaves(info)):
                    raise ContractError("nonfinite augmented training metrics")
                durations.append(time.monotonic() - before)
                step = int(state.step)
                metrics = {key: float(value) for key, value in info.items()}
                if step == 1 or step % 50 == 0:
                    progress = {
                        "step": step,
                        "target": recipe["steps"],
                        "metrics": metrics,
                        "coverage": pipeline.augmented_coverage(schedule, step * config.batch_size),
                        "seconds_per_update": float(np.median(durations[-30:])),
                        "initialization": recipe["initialization"],
                        "robot_commands_sent": 0,
                    }
                    atomic_json(progress_path, progress)
                    print(json.dumps(progress), flush=True)
                if step in recipe["snapshots"]:
                    save_snapshot(root, config, state, loader, step, metrics)
            step = int(state.step)
            restart_path = None
            restart_skip_reason = None
            if step > initial_step:
                try:
                    storage_gate(root, 34 * GIB)
                except ContractError as error:
                    if step != recipe["steps"]:
                        raise
                    restart_skip_reason = str(error)
                else:
                    cursor_value = dict(identity, consumed_samples=step * config.batch_size)
                    atomic_json(run / f"sampler_cursor_{step:06d}.json", cursor_value)
                    checkpoints.save_state(manager, state, loader, step)
                    manager.wait_until_finished()
                    restart_path = str(config.checkpoint_dir / str(step))
            if restart_skip_reason and step == recipe["steps"]:
                reason = "target_reached_restart_skipped_storage"
            elif restart_skip_reason:
                reason = f"{reason}_without_optimizer_resume"
            result = {
                "name": recipe["name"],
                "model": model,
                "stage": stage,
                "seed": seed,
                "step": step,
                "target": recipe["steps"],
                "complete": step == recipe["steps"],
                "reason": reason,
                "elapsed_seconds": time.monotonic() - started,
                "seconds_per_update": None if not durations else float(np.median(durations[-30:])),
                "snapshots": [
                    str(root / "snapshots" / recipe["name"] / f"step_{snapshot:06d}")
                    for snapshot in recipe["snapshots"]
                    if snapshot <= step
                ],
                "coverage": pipeline.augmented_coverage(schedule, step * config.batch_size),
                "identity": identity,
                "restart_path": restart_path,
                "exact_optimizer_resume_available": restart_path is not None,
                "restart_skip_reason": restart_skip_reason,
                "robot_commands_sent": 0,
            }
            atomic_json(run / "result.json", result)
            return result
        finally:
            manager.wait_until_finished()
            manager.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--campaign")
    target.add_argument("--augmented-root")
    parser.add_argument(
        "--experiment",
        choices=[
            "SMOKE",
            *pipeline.TRACKS,
            *pipeline.FILTER_FINETUNES,
            *pipeline.ABLATIONS,
            *pipeline.AUGMENTED_MODELS,
        ],
        required=True,
    )
    parser.add_argument("--stage", choices=("screen", "confirm"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deadline")
    parser.add_argument("--stats-only", action="store_true")
    parser.add_argument("--allow-synthetic", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-gpu-run", action="store_true")
    args = parser.parse_args()
    if args.augmented_root:
        if args.experiment not in pipeline.AUGMENTED_MODELS or args.stage is None:
            parser.error("--augmented-root requires M0--M4 --experiment and --stage")
        if args.stats_only:
            result = compute_augmented_statistics(
                args.augmented_root,
                args.experiment,
                args.stage,
                args.seed,
                allow_synthetic=args.allow_synthetic,
            )
        else:
            if not args.allow_gpu_run or args.deadline is None:
                parser.error("augmented training requires --deadline and explicit --allow-gpu-run")
            result = train_augmented(
                args.augmented_root,
                args.experiment,
                args.stage,
                args.seed,
                args.deadline,
                resume=args.resume,
                allow_synthetic=args.allow_synthetic,
            )
    else:
        if args.experiment in pipeline.AUGMENTED_MODELS or args.stage or args.stats_only or args.allow_synthetic:
            parser.error("augmented options require --augmented-root")
        if not args.allow_gpu_run or args.deadline is None:
            parser.error("training requires --deadline and explicit --allow-gpu-run")
        result = train(args.campaign, args.experiment, args.deadline, resume=args.resume)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
