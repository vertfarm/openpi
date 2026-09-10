"""Recorded-observation evaluation. Never a closed-loop robot rollout."""

from __future__ import annotations

import argparse
import dataclasses
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np

from .artifacts import ContractError
from .artifacts import atomic_json
from .artifacts import read_json
from .checkpoints import snapshot_identity
from .native import CAMERAS
from .native import PROMPT
from .overnight_train import configure


def metrics(pred, truth, state, elapsed):
    if not np.isfinite(pred).all() or pred.shape != truth.shape:
        raise ContractError("invalid inference action")
    return dict(
        joint_mae_rad=float(np.mean(np.abs(pred[:, :7] - truth[:, :7]))),
        gripper_error_rate=float(np.mean((pred[:, 7] >= 0.5) != (truth[:, 7] >= 0.5))),
        first_target_delta_rad=float(np.max(np.abs(pred[0, :7] - state[:7]))),
        chunk_max_step_rad=float(np.max(np.abs(np.diff(pred[:, :7], axis=0)))),
        gripper_outside_unit_interval=float(np.mean((pred[:, 7] < 0) | (pred[:, 7] > 1))),
        latency_ms=elapsed * 1000,
    )


def evaluate(campaign, snapshot, *, reference=None, quick=False, deadline=None):
    campaign, snapshot = Path(campaign).resolve(), Path(snapshot).resolve()
    if not snapshot.is_relative_to(campaign / "snapshots"):
        raise ContractError("snapshot outside campaign")
    record = snapshot_identity(snapshot)
    config, export = configure(campaign, record["recipe"])
    from filelock import FileLock
    import jax

    from openpi import transforms
    from openpi.models import model as model_lib
    from openpi.policies.policy_config import create_trained_policy
    from openpi.shared import nnx_utils
    from openpi.training import data_loader

    data = config.data.create(config.assets_dirs, config.model)
    valdata = dataclasses.replace(data, repo_id=export["splits"]["validation"]["repo_id"])
    dataset = data_loader.create_torch_dataset(valdata, config.model.action_horizon, config.model)
    provenance = read_json(Path(export["splits"]["validation"]["root"]) / "hv1_provenance.json")
    cases = []
    offset = 0
    for e in provenance:
        n, g, r = e["frames"], e["grasp_frame"], e["release_frame"]
        indices = [0, g - 1, g, g + 1, r - 1, r, r + 1, n - 1] + np.linspace(0, n - 1, 5, dtype=int).tolist()
        for i in sorted({min(n - 1, max(0, int(i))) for i in indices}):
            cases.append((e["id"], i, offset + i))
        offset += n
    if quick:
        cases = cases[:3]
    result = dict(
        snapshot=str(snapshot),
        step=record["step"],
        experiment=record["recipe"]["name"],
        robot_commands_sent=0,
        live_freshness_verified=False,
        closed_loop=False,
        validation_scope="within_session",
        case_count=len(cases),
        sweeps={},
    )
    with FileLock(str(campaign.parent / "hv1-ml-gpu.lock"), timeout=0):
        reference_actions = None
        if reference:
            reference = Path(reference).resolve()
            if not reference.is_relative_to(campaign / "restarts"):
                raise ContractError("reference outside restarts")
            policy = create_trained_policy(config, reference, default_prompt=PROMPT, sample_kwargs={"num_steps": 10})
            row = dataset[cases[0][2]]
            obs = {
                "images": {c: np.asarray(row[f"observation.images.{c}"]) for c in CAMERAS},
                "state": np.asarray(row["observation.state"]),
                "prompt": PROMPT,
            }
            noise = np.random.default_rng(20260909).normal(size=(config.model.action_horizon, 32)).astype(np.float32)
            reference_actions = policy.infer(obs, noise=noise)["actions"]
            del policy
            import gc

            gc.collect()
            jax.clear_caches()
        for denoise in [10] if quick else [5, 10, 20]:
            if deadline and time.time() >= datetime.fromisoformat(deadline).timestamp():
                result["deadline_reached"] = True
                break
            policy = create_trained_policy(
                config, snapshot, default_prompt=PROMPT, sample_kwargs={"num_steps": denoise}
            )
            loss_fn = nnx_utils.module_jit(policy._model.compute_loss, static_argnames=("train",))
            rows = []
            predictions = []
            for case_index, (eid, frame, index) in enumerate(cases):
                if deadline and time.time() >= datetime.fromisoformat(deadline).timestamp():
                    result["deadline_reached"] = True
                    break
                raw = dataset[index]
                obs = {
                    "images": {c: np.asarray(raw[f"observation.images.{c}"]) for c in CAMERAS},
                    "state": np.asarray(raw["observation.state"]),
                    "prompt": PROMPT,
                }
                noise = (
                    np.random.default_rng(20260909 + case_index)
                    .normal(size=(config.model.action_horizon, 32))
                    .astype(np.float32)
                )
                if case_index == 0:
                    policy.infer(obs, noise=noise)  # JIT warm-up is excluded from latency.
                started = time.perf_counter()
                pred = policy.infer(obs, noise=noise)["actions"]
                elapsed = time.perf_counter() - started
                truth = np.asarray(raw["action"])
                m = metrics(pred, truth, obs["state"], elapsed)
                if denoise == 10:
                    value = dict(raw)
                    for transform in (
                        *data.repack_transforms.inputs,
                        *data.data_transforms.inputs,
                        transforms.Normalize(data.norm_stats, use_quantiles=data.use_quantile_norm),
                        *data.model_transforms.inputs,
                    ):
                        value = transform(value)
                    if not all(bool(v) for v in value["image_mask"].values()):
                        raise ContractError("camera mask unexpectedly disabled")
                    batched = jax.tree.map(
                        lambda x: jax.numpy.asarray(x)[None], {k: v for k, v in value.items() if k != "prompt"}
                    )
                    loss = loss_fn(
                        jax.random.key(20260909 + case_index),
                        model_lib.Observation.from_dict(batched),
                        batched["actions"],
                        train=False,
                    )
                    m["validation_loss"] = float(np.asarray(loss).mean())
                    if not np.isfinite(m["validation_loss"]):
                        raise ContractError("nonfinite validation loss")
                if reference_actions is not None and case_index == 0 and denoise == 10:
                    np.testing.assert_allclose(pred, reference_actions, atol=1e-5, rtol=1e-5)
                    result["full_to_bf16_inference_match"] = True
                m.update(episode=eid, frame=frame)
                m["execution_prefix"] = {
                    str(k): dict(
                        latency_budget_ms=k / 30 * 1000,
                        measured_model_latency_fits=elapsed <= k / 30,
                        max_prefix_target_step_rad=float(
                            np.max(np.abs(np.diff(np.vstack((obs["state"][:7], pred[:k, :7])), axis=0)))
                        ),
                    )
                    for k in (1, 3, 5)
                }
                rows.append(m)
                predictions.append(pred)
            keys = ("joint_mae_rad", "gripper_error_rate", "first_target_delta_rad", "chunk_max_step_rad", "latency_ms")
            result["sweeps"][str(denoise)] = dict(
                cases=rows, mean={k: float(np.mean([v[k] for v in rows])) for k in keys} if rows else {}
            )
            if denoise == 10 and rows:
                result["validation_loss"] = float(np.mean([v["validation_loss"] for v in rows]))
            del policy, loss_fn
            import gc

            gc.collect()
            jax.clear_caches()
    result["gpu_reload_pass"] = bool(result["sweeps"]) and all(bool(s["cases"]) for s in result["sweeps"].values())
    result["complete"] = not result.get("deadline_reached", False)
    out = campaign / "evaluations" / f"{record['recipe']['name']}_{record['step']:06d}{'_quick' if quick else ''}.json"
    atomic_json(out, result)
    print(json.dumps(result), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign", required=True)
    p.add_argument("--snapshot", required=True)
    p.add_argument("--reference")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--deadline")
    a = p.parse_args()
    evaluate(a.campaign, a.snapshot, reference=a.reference, quick=a.quick, deadline=a.deadline)


if __name__ == "__main__":
    main()
