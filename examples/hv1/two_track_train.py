"""Inference-snapshot-first training for the official-base HV1 two-track campaign."""

from __future__ import annotations

import argparse
from datetime import datetime
import functools
import json
from pathlib import Path
import signal
import time

import numpy as np

from . import two_track
from .artifacts import GIB
from .artifacts import ContractError
from .artifacts import atomic_json
from .artifacts import digest
from .artifacts import read_json
from .artifacts import storage_gate
from .artifacts import write_new_json
from .checkpoints import save_snapshot
from .checkpoints import snapshot_identity
from .two_track_config import configure
from .two_track_config import local_dataset


class FixedSampler:
    def __init__(self, schedule, consumed=0):
        if consumed < 0 or consumed > len(schedule["records"]):
            raise ContractError("sampler cursor outside the approved schedule")
        self.indices = [row["index"] for row in schedule["records"][consumed:]]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def make_loader(config, schedule, sharding, consumed=0):
    from openpi.training import data_loader

    data = config.data.create(config.assets_dirs, config.model)
    dataset = local_dataset(data, config.model, config.policy_metadata["export_roots"]["train"])
    if max(row["index"] for row in schedule["records"]) >= len(dataset):
        raise ContractError("sampler index is outside the common export")
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
    manifest = two_track.verify_campaign(campaign, raw=True)
    steps = (
        50
        if name == "SMOKE"
        else 1000
        if name in two_track.FILTER_FINETUNES
        else two_track.TARGET_STEPS
    )
    recipe = two_track.recipe(name, manifest["sha256"], steps)
    schedule = two_track.checked(campaign / f"sampler_{name}.json")
    expected = two_track.sample_schedule(manifest, name, recipe["steps"] * recipe["batch_size"])
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
    storage_gate(campaign, (16 if name in two_track.FILTER_FINETUNES else 40) * GIB)
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
                        coverage=two_track.coverage(schedule, step * config.batch_size),
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
            if step > initial_step and name not in two_track.FILTER_FINETUNES:
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
                coverage=two_track.coverage(schedule, step * config.batch_size),
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--experiment", choices=["SMOKE", *two_track.TRACKS, *two_track.FILTER_FINETUNES], required=True)
    parser.add_argument("--deadline", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-gpu-run", action="store_true")
    args = parser.parse_args()
    if not args.allow_gpu_run:
        parser.error("explicit --allow-gpu-run required; never authorizes robot motion")
    print(json.dumps(train(args.campaign, args.experiment, args.deadline, resume=args.resume)))


if __name__ == "__main__":
    main()
