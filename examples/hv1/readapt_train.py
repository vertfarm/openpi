"""C-5000 warm starts with fixed statistics and resumable sample schedules."""

from __future__ import annotations

import argparse
from datetime import datetime
import functools
import json
from pathlib import Path
import signal
import time

import numpy as np

from . import readapt
from .artifacts import GIB
from .artifacts import ContractError
from .artifacts import atomic_json
from .artifacts import digest
from .artifacts import read_json
from .artifacts import storage_gate
from .artifacts import tree_bytes
from .artifacts import write_new_json
from .checkpoints import save_snapshot
from .readapt_config import configure as configure
from .readapt_config import local_dataset as local_dataset


class FixedSampler:
    def __init__(self, schedule, consumed=0):
        if consumed < 0 or consumed > len(schedule["records"]):
            raise ContractError("sampler cursor outside planned schedule")
        self.indices = [r["index"] for r in schedule["records"][consumed:]]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def make_loader(config, schedule, sharding, consumed=0):
    from openpi.training import data_loader

    data = config.data.create(config.assets_dirs, config.model)
    dataset = local_dataset(data, config.model, config.policy_metadata["export_roots"]["train"])
    if max(x["index"] for x in schedule["records"]) >= len(dataset):
        raise ContractError("sampler index outside export")
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
    manifest = readapt.verify_campaign(campaign, raw=True)
    readapt.require_normalization_review(campaign)
    r = readapt.recipe(name, manifest["sha256"])
    # Recheck all parent weight files at each independent warm start.
    readapt.parent_record(manifest["parent"]["campaign"])
    schedule = readapt.checked(campaign / f"sampler_{name}.json")
    expected = readapt.sample_schedule(manifest["episodes"], name, r["steps"] * r["batch_size"])
    if schedule != expected:
        raise ContractError("sample schedule changed")
    stop_at = datetime.fromisoformat(deadline)
    if stop_at.tzinfo is None or time.time() >= stop_at.timestamp():
        raise ContractError("explicit future timezone-aware deadline required")
    config, _ = configure(campaign, r)
    resident = tree_bytes(config.checkpoint_dir) if config.checkpoint_dir.exists() else 0
    storage_gate(campaign, max(0, 72 * GIB - resident))  # count existing resume state only once.
    from filelock import FileLock
    import jax

    from openpi.training import checkpoints
    from openpi.training import sharding
    from scripts.train import init_train_state
    from scripts.train import train_step

    run = campaign / "runs" / name
    if (run / "recipe.json").exists():
        if not resume or read_json(run / "recipe.json") != r:
            raise ContractError("existing run requires explicit compatible resume")
    else:
        if resume:
            raise ContractError("cannot resume a run that has never started")
        write_new_json(run / "recipe.json", r)
    identity = dict(recipe_sha256=digest(r), manifest_sha256=manifest["sha256"], sampler_sha256=schedule["sha256"])
    stopped = []
    signal.signal(signal.SIGINT, lambda *_: stopped.append("SIGINT"))
    signal.signal(signal.SIGTERM, lambda *_: stopped.append("SIGTERM"))
    durations, snapshots, metrics = [], [], {}
    started = time.monotonic()
    jax.config.update("jax_compilation_cache_dir", str(campaign.parent / "cache/jax"))
    with FileLock(str(campaign.parent / "hv1-ml-gpu.lock"), timeout=0):
        mesh = sharding.make_mesh(config.fsdp_devices)
        bs = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
        loader = make_loader(config, schedule, bs)
        manager, resuming = checkpoints.initialize_checkpoint_dir(
            config.checkpoint_dir, keep_period=None, overwrite=False, resume=resume
        )
        try:
            train_rng, init_rng = jax.random.split(jax.random.key(config.seed))
            state, ss = init_train_state(config, init_rng, mesh, resume=resuming)
            if resuming:
                state = checkpoints.restore_state(manager, state, loader)
            jax.block_until_ready(state)
            initial_step = int(state.step)
            if resuming:
                cursor = read_json(run / f"sampler_cursor_{initial_step:06d}.json")
                if cursor != dict(identity, consumed_samples=initial_step * config.batch_size):
                    raise ContractError("restart sampler cursor does not match restored optimizer update")
                loader = make_loader(config, schedule, bs, initial_step * config.batch_size)
            iterator = iter(loader)
            replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
            compiled = jax.jit(
                functools.partial(train_step, config),
                in_shardings=(replicated, ss, bs),
                out_shardings=(ss, replicated),
                donate_argnums=(1,),
            )

            def save_restart(step):
                storage_gate(campaign, 34 * GIB)
                # Cursor first: a crash cannot expose a committed optimizer checkpoint without its cursor.
                atomic_json(
                    run / f"sampler_cursor_{step:06d}.json", dict(identity, consumed_samples=step * config.batch_size)
                )
                checkpoints.save_state(manager, state, loader, step)
                manager.wait_until_finished()

            reason = "target_reached"
            while int(state.step) < r["steps"]:
                if stopped or time.time() >= stop_at.timestamp():
                    reason = stopped[0] if stopped else "deadline"
                    break
                storage_gate(campaign)
                before = time.monotonic()
                batch = next(iterator)
                with sharding.set_mesh(mesh):
                    state, info = compiled(train_rng, state, batch)
                info = jax.device_get(info)
                if not all(np.isfinite(x).all() for x in jax.tree.leaves(info)):
                    raise ContractError("nonfinite training state metrics")
                durations.append(time.monotonic() - before)
                step = int(state.step)
                metrics = {k: float(v) for k, v in info.items()}
                if step % 50 == 0 or step == 1:
                    progress = dict(
                        step=step,
                        target=r["steps"],
                        metrics=metrics,
                        coverage=readapt.coverage(schedule, step * config.batch_size),
                        seconds_per_update=float(np.median(durations[-30:])),
                        parent_updates=manifest["parent"]["prior_updates"],
                        additional_updates=step,
                        robot_commands_sent=0,
                    )
                    atomic_json(run / "progress.json", progress)
                    print(json.dumps(progress), flush=True)
                if step % 1000 == 0 or step in r["snapshots"]:
                    save_restart(step)
                    if step in r["snapshots"]:
                        snapshots.append(str(save_snapshot(campaign, config, state, loader, step, metrics)))
            step = int(state.step)
            if step > initial_step and manager.latest_step() != step:
                save_restart(step)
            result = dict(
                name=name,
                step=step,
                target=r["steps"],
                complete=step == r["steps"],
                reason=reason,
                elapsed_seconds=time.monotonic() - started,
                snapshots=snapshots,
                coverage=readapt.coverage(schedule, step * config.batch_size),
                identity=identity,
                restart_path=str(config.checkpoint_dir / str(step)),
                robot_commands_sent=0,
            )
            atomic_json(run / "result.json", result)
            return result
        finally:
            manager.wait_until_finished()
            manager.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign", required=True)
    p.add_argument("--experiment", required=True, choices=["N", "M", "smoke"])
    p.add_argument("--deadline", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--allow-gpu-run", action="store_true")
    a = p.parse_args()
    if not a.allow_gpu_run:
        p.error("explicit --allow-gpu-run required; never authorizes robot motion")
    print(json.dumps(train(a.campaign, a.experiment, a.deadline, resume=a.resume)))


if __name__ == "__main__":
    main()
