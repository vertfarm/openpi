"""Deadline-aware HV1 training with separate BF16 inference snapshots. No ROS."""

from __future__ import annotations

import argparse
import copy
import dataclasses
from datetime import datetime
import functools
import json
import logging
import os
from pathlib import Path
import signal
import time

import numpy as np

from .artifacts import GIB
from .artifacts import ContractError
from .artifacts import atomic_json
from .artifacts import digest
from .artifacts import file_hash
from .artifacts import read_json
from .artifacts import storage_gate
from .artifacts import write_new_json
from .checkpoints import save_snapshot as save_snapshot
from .native import PROMPT
from .native import read_numeric
from .openpi_run import make_config
from .overnight_common import verify_campaign
from .transforms import HV1Inputs
from .transforms import HV1Outputs


def configure(campaign, r):
    campaign = Path(campaign).resolve()
    export = read_json(campaign / f"export_{r['cohort']}/export.json")
    # LeRobot reads these constants at import time, not on each dataset load.
    os.environ["HF_LEROBOT_HOME"] = export["hf_lerobot_home"]
    os.environ["HF_HOME"] = str(campaign.parent / "cache/huggingface")
    os.environ["OPENPI_DATA_HOME"] = str(campaign.parent / "cache/openpi")
    from openpi import transforms
    from openpi.models.pi0_config import Pi0Config
    from openpi.training.config import AssetsConfig
    from openpi.training.optimizer import CosineDecaySchedule

    config, export = make_config(
        campaign / f"export_{r['cohort']}/export.json", campaign.parent, r["name"], steps=r["steps"]
    )
    p = copy.deepcopy(export["profile"])
    p["action_horizon"] = r["action_horizon"]
    model = Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=r["action_horizon"],
        **(dict(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora") if r["lora"] else {}),
    )
    data = dataclasses.replace(
        config.data,
        assets=AssetsConfig(assets_dir=str(campaign / "shared_assets"), asset_id="hv1_common_clean19"),
        data_transforms=lambda _: transforms.Group(inputs=[HV1Inputs(p)], outputs=[HV1Outputs(p)]),
    )
    metadata = dict(
        config.policy_metadata,
        profile=p,
        profile_sha256=digest(p),
        recipe=r,
        recipe_sha256=digest(r),
        prompt=PROMPT,
        action_horizon=r["action_horizon"],
        state_names=p["state"]["names"],
        camera_dropout=False,
        gripper=p["gripper"],
        train_episode_ids=export["splits"]["train"]["episode_ids"],
        validation_episode_ids=export["splits"]["validation"]["episode_ids"],
        validation_scope="within_session",
        robot_motion_authorized=False,
        norm_stats_sha256=file_hash(campaign / "shared_assets/hv1_common_clean19/norm_stats.json"),
        implementation_sha256={
            n: file_hash(Path(__file__).with_name(n))
            for n in (
                "native.py",
                "workflow.py",
                "artifacts.py",
                "checkpoints.py",
                "transforms.py",
                "overnight_common.py",
                "overnight_train.py",
                "overnight_eval.py",
                "overnight.py",
            )
        },
    )
    config = dataclasses.replace(
        config,
        model=model,
        data=data,
        seed=r["seed"],
        batch_size=r["batch_size"],
        ema_decay=None,
        freeze_filter=model.get_freeze_filter(),
        num_workers=0,
        checkpoint_base_dir=str(campaign / "restarts"),
        keep_period=None,
        lr_schedule=CosineDecaySchedule(
            warmup_steps=r["warmup_steps"], peak_lr=r["peak_lr"], decay_steps=r["decay_steps"], decay_lr=r["decay_lr"]
        ),
        policy_metadata=metadata,
    )
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    return config, export


def stats(campaign):
    from openpi.shared import normalize

    scan = verify_campaign(campaign)
    root = Path(campaign) / "shared_assets/hv1_common_clean19"
    if root.exists():
        raise ContractError("shared stats already exist; do not overwrite")
    running = {k: normalize.RunningStats() for k in ("state", "actions")}
    ids = []
    for e in scan["episodes"]:
        if e["validation"] or e["suspect"]:
            continue
        directory = Path(e["path"])
        if file_hash(directory / "data.hdf5") != e["source_hashes"]["data.hdf5"]:
            raise ContractError("raw changed")
        state, action, _ = read_numeric(directory / "data.hdf5")
        indices = np.minimum(np.arange(len(state))[:, None] + np.arange(15), len(state) - 1)
        chunks = action[indices].copy()
        chunks[:, :, :7] -= state[:, None, :7]
        running["state"].update(state)
        running["actions"].update(chunks.reshape(-1, 8))
        ids.append(e["id"])
    normalize.save(root, {k: v.get_statistics() for k, v in running.items()})
    write_new_json(
        root / "provenance.json",
        dict(
            episode_ids=ids,
            action_horizon_basis=15,
            validation_used=False,
            manifest_sha256=scan["manifest_sha256"],
            sha256=file_hash(root / "norm_stats.json"),
        ),
    )


def train(campaign, r, deadline, *, resume=False):
    campaign = Path(campaign).resolve()
    config, export = configure(campaign, r)
    from filelock import FileLock
    import jax

    from openpi.training import checkpoints
    from openpi.training import data_loader
    from openpi.training import sharding
    from scripts.train import init_train_state
    from scripts.train import train_step

    stop_at = datetime.fromisoformat(deadline).timestamp()
    if time.time() >= stop_at:
        raise ContractError("training deadline already passed")
    verify_campaign(campaign)
    storage_gate(campaign, 80 * GIB)  # includes overlap of latest and next full save.
    stop_requested = []
    signal.signal(signal.SIGTERM, lambda *_: stop_requested.append("SIGTERM"))
    signal.signal(signal.SIGINT, lambda *_: stop_requested.append("SIGINT"))
    jax.config.update("jax_compilation_cache_dir", str(campaign.parent / "cache/jax"))
    run_dir = campaign / "runs" / r["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    contract = run_dir / "recipe.json"
    if contract.exists():
        if not resume or read_json(contract) != r:
            raise ContractError("recipe exists/mismatched; explicit identical resume required")
    else:
        write_new_json(contract, r)
    started = time.monotonic()
    durations = []
    snapshots = []
    last_metrics = {}
    reason = "target_reached"
    with FileLock(str(campaign.parent / "hv1-ml-gpu.lock"), timeout=0):
        mesh = sharding.make_mesh(config.fsdp_devices)
        batch_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
        loader = data_loader.create_data_loader(config, sharding=batch_sharding, shuffle=True)
        iterator = iter(loader)
        rng = jax.random.key(config.seed)
        train_rng, init_rng = jax.random.split(rng)
        manager, resuming = checkpoints.initialize_checkpoint_dir(
            config.checkpoint_dir, keep_period=None, overwrite=False, resume=resume
        )
        try:
            state, state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
            if resuming:
                state = checkpoints.restore_state(manager, state, loader)
            jax.block_until_ready(state)
            initial_step = int(state.step)
            # With workers=0, replay the deterministic sampler to the consumed batch cursor.
            for _ in range(initial_step):
                next(iterator)
            replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
            compiled = jax.jit(
                functools.partial(train_step, config),
                in_shardings=(replicated, state_sharding, batch_sharding),
                out_shardings=(state_sharding, replicated),
                donate_argnums=(1,),
            )
            while int(state.step) < r["steps"]:
                if stop_requested or time.time() >= stop_at:
                    reason = "deadline" if not stop_requested else stop_requested[0]
                    break
                # Keep 50 GiB free even at peak checkpoint overlap; no source cleanup.
                storage_gate(campaign, 48 * GIB)
                before = time.monotonic()
                batch = next(iterator)
                with sharding.set_mesh(mesh):
                    state, info = compiled(train_rng, state, batch)
                info = jax.device_get(info)
                elapsed = time.monotonic() - before
                step = int(state.step)
                if not all(np.isfinite(x).all() for x in jax.tree.leaves(info)):
                    raise ContractError("nonfinite loss/gradient; no rollout snapshot from invalid state")
                durations.append(elapsed)
                last_metrics = {k: float(v) for k, v in info.items()}
                if step % 10 == 0 or step == 1:
                    progress = dict(
                        step=step,
                        target=r["steps"],
                        metrics=last_metrics,
                        step_seconds=elapsed,
                        elapsed_seconds=time.monotonic() - started,
                        seen_samples=step * r["batch_size"],
                        estimated_epochs=step * r["batch_size"] / export["splits"]["train"]["frames"],
                        robot_motion_authorized=False,
                    )
                    atomic_json(run_dir / "progress.json", progress)
                    print(json.dumps(progress), flush=True)
                if step % 250 == 0 or step in r["snapshots"]:
                    storage_gate(campaign, 48 * GIB)
                    checkpoints.save_state(manager, state, loader, step)
                    manager.wait_until_finished()
                    if step in r["snapshots"]:
                        snapshots.append(str(save_snapshot(campaign, config, state, loader, step, last_metrics)))
            step = int(state.step)
            if step > initial_step:
                if manager.latest_step() != step:
                    storage_gate(campaign, 48 * GIB)
                    checkpoints.save_state(manager, state, loader, step)
                    manager.wait_until_finished()
                final = campaign / "snapshots" / r["name"] / f"step_{step:06d}"
                if not final.exists():
                    snapshots.append(str(save_snapshot(campaign, config, state, loader, step, last_metrics)))
            result = dict(
                name=r["name"],
                step=step,
                target=r["steps"],
                reason=reason,
                recipe=r,
                elapsed_seconds=time.monotonic() - started,
                steady_step_seconds=float(np.median(durations[-30:])) if durations else None,
                first_step_seconds=durations[0] if durations else None,
                snapshots=snapshots,
                restart_path=str(config.checkpoint_dir / str(step)),
                last_metrics=last_metrics,
                complete=step == r["steps"],
                robot_motion_authorized=False,
            )
            atomic_json(run_dir / "result.json", result)
            return result
        finally:
            manager.wait_until_finished()
            manager.close()


def main():
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["stats", "train"])
    p.add_argument("--campaign", required=True)
    p.add_argument("--recipe")
    p.add_argument("--deadline", default="2026-09-10T09:30:00+09:00")
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    if a.command == "stats":
        stats(a.campaign)
    else:
        print(json.dumps(train(a.campaign, read_json(a.recipe), a.deadline, resume=a.resume)))


if __name__ == "__main__":
    main()
