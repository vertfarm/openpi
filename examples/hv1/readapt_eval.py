"""Recorded-observation comparisons and hash-bound SHADOW_ONLY model registry."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import time

import numpy as np

from . import readapt
from .artifacts import ContractError
from .artifacts import atomic_json
from .artifacts import file_hash
from .artifacts import read_json
from .artifacts import write_new_json
from .checkpoints import snapshot_identity as snapshot_identity
from .native import CAMERAS
from .native import PROMPT
from .readapt_config import configure
from .readapt_config import local_dataset


def transition_metrics(predicted, truth, grasp, release, fps=30):
    p, t = np.asarray(predicted) >= 0.5, np.asarray(truth) >= 0.5
    if p.shape != t.shape or p.ndim != 1 or not len(p):
        raise ContractError("aligned nonempty intent series required")
    changes = np.flatnonzero(np.diff(np.r_[False, p]) != 0)
    closes = [int(i) for i in changes if p[i]]
    opens = [int(i) for i in changes if not p[i]]
    close = min(closes, key=lambda i: abs(i - grasp)) if closes else None
    opening = min(opens, key=lambda i: abs(i - release)) if opens else None
    around = np.zeros(len(p), bool)
    for i in (grasp, release):
        around[max(0, i - fps) : min(len(p), i + fps + 1)] = True
    return dict(
        error_rate=float(np.mean(p != t)),
        transition_error_rate=float(np.mean((p != t)[around])),
        missed_close=close is None,
        missed_release=opening is None,
        close_time_error_s=None if close is None else (close - grasp) / fps,
        release_time_error_s=None if opening is None else (opening - release) / fps,
        close_edges=closes,
        release_edges=opens,
        extra_toggles=max(0, len(changes) - 2),
        initial_closed=bool(p[0]),
        early_release_frames=int(np.sum(~p[grasp:release])),
        note="Teacher-forced recorded observations; not measured physical grasp success.",
    )


def evaluate(campaign, snapshot, *, reference=None):
    campaign, snapshot = Path(campaign).resolve(), Path(snapshot).resolve()
    m = readapt.verify_campaign(campaign, raw=True)
    baseline = snapshot == Path(m["parent"]["snapshot"])
    if not baseline and not snapshot.is_relative_to(campaign / "snapshots"):
        raise ContractError("snapshot outside approved lineage")
    record = snapshot_identity(snapshot)
    r = readapt.recipe("N", m["sha256"]) if baseline else record["recipe"]
    config, export = configure(campaign, r)
    if not baseline and (record.get("manifest_sha256") != m["sha256"] or record.get("parent") != m["parent"]):
        raise ContractError("snapshot lineage mismatch")
    from filelock import FileLock
    import jax

    from openpi.policies.policy_config import create_trained_policy

    data = config.data.create(config.assets_dirs, config.model)
    valdata = dataclasses.replace(data, repo_id=export["splits"]["validation"]["repo_id"])
    dataset = local_dataset(valdata, config.model, export["splits"]["validation"]["root"])
    episodes = [e for e in m["episodes"] if e["validation"]]
    result = dict(
        schema=readapt.SCHEMA,
        manifest_sha256=m["sha256"],
        snapshot=str(snapshot),
        snapshot_sha256=file_hash(snapshot / "snapshot.json"),
        experiment="C_parent" if baseline else r["name"],
        step=record["step"],
        denoise=10,
        prefix=3,
        groups={"new": [], "old": []},
        robot_commands_sent=0,
        closed_loop=False,
        physical_safety_qualified=False,
        success_rate_measured=False,
        normalization_sha256=m["parent"]["norm_stats_sha256"],
    )
    with FileLock(str(campaign.parent / "hv1-ml-gpu.lock"), timeout=0):
        # Smoke proves restart->BF16 output equality, not just independent reloadability.
        reference_actions = None
        obs0 = dataset[0]
        obs0 = {
            "state": np.asarray(obs0["observation.state"]),
            "prompt": PROMPT,
            "images": {c: np.asarray(obs0[f"observation.images.{c}"]) for c in CAMERAS},
        }
        noise = np.random.default_rng(42).normal(size=(15, 32)).astype(np.float32)
        if reference:
            ref = Path(reference).resolve()
            if baseline or not ref.is_relative_to(campaign / "restarts"):
                raise ContractError("reference must be this campaign's restart")
            p = create_trained_policy(config, ref, default_prompt=PROMPT, sample_kwargs={"num_steps": 10})
            reference_actions = np.asarray(p.infer(obs0, noise=noise)["actions"])
            del p
            import gc

            gc.collect()
            jax.clear_caches()
        policy = create_trained_policy(config, snapshot, default_prompt=PROMPT, sample_kwargs={"num_steps": 10})
        warm = np.asarray(policy.infer(obs0, noise=noise)["actions"])
        if reference_actions is not None:
            np.testing.assert_allclose(warm, reference_actions, atol=1e-5, rtol=1e-5)
            result["full_to_bf16_inference_match"] = True
        offset = 0
        for e in episodes:
            all_pred, all_truth, latencies, first_deltas, chunk_steps = [], [], [], [], []
            # Every third recorded observation; its three-action prefix covers every row.
            for frame in range(0, e["frames"], 3):
                raw = dataset[offset + frame]
                state = np.asarray(raw["observation.state"])
                obs = dict(
                    state=state, prompt=PROMPT, images={c: np.asarray(raw[f"observation.images.{c}"]) for c in CAMERAS}
                )
                seeded = np.random.default_rng(42 + offset + frame).normal(size=(15, 32)).astype(np.float32)
                t = time.perf_counter()
                pred = np.asarray(policy.infer(obs, noise=seeded)["actions"])
                latencies.append((time.perf_counter() - t) * 1000)
                if pred.shape != (15, 8) or not np.isfinite(pred).all():
                    raise ContractError("invalid model prediction")
                count = min(3, e["frames"] - frame)
                all_pred.extend(pred[:count])
                all_truth.extend(np.asarray(raw["action"])[:count])
                first_deltas.append(float(np.max(np.abs(pred[0, :7] - state[:7]))))
                chunk_steps.append(float(np.max(np.abs(np.diff(pred[:, :7], axis=0)))))
            pred, truth = np.array(all_pred), np.array(all_truth)
            row = dict(
                episode=e["id"],
                frames=e["frames"],
                joint_mae_rad=float(np.mean(np.abs(pred[:, :7] - truth[:, :7]))),
                prefix_series_max_step_rad=float(np.max(np.abs(np.diff(pred[:, :7], axis=0)))),
                chunk_max_step_rad=max(chunk_steps),
                first_target_delta_rad=max(first_deltas),
                latency_p95_ms=float(np.percentile(latencies, 95)),
                gripper=transition_metrics(pred[:, 7], truth[:, 7], e["grasp_frame"], e["release_frame"]),
            )
            result["groups"][e["cohort"]].append(row)
            offset += e["frames"]
            print(json.dumps(dict(evaluated=e["id"], experiment=result["experiment"])), flush=True)
    for cohort in ("new", "old"):
        if not result["groups"][cohort]:
            raise ContractError("both new and old validation groups required")
    result.update(gpu_reload_pass=True, complete=True, evaluated_rows=offset)
    path = campaign / "evaluations" / f"{result['experiment']}_{result['step']:06d}.json"
    # Do not silently replace evidence already referenced by a registry.
    write_new_json(path, result)
    return result


def register(campaign, snapshot, reviewer):
    campaign, snapshot = Path(campaign).resolve(), Path(snapshot).resolve()
    m = readapt.verify_campaign(campaign)
    record = snapshot_identity(snapshot)
    if not snapshot.is_relative_to(campaign / "snapshots") or record["recipe"]["name"] not in {"N", "M"}:
        raise ContractError("only this campaign's N/M inference snapshots may be registered")
    if not reviewer.strip() or record.get("manifest_sha256") != m["sha256"]:
        raise ContractError("reviewer and matching snapshot manifest required")
    evaluation_path = campaign / "evaluations" / f"{record['recipe']['name']}_{record['step']:06d}.json"
    evaluation = read_json(evaluation_path)
    snapshot_sha = file_hash(snapshot / "snapshot.json")
    if (
        evaluation.get("snapshot_sha256") != snapshot_sha
        or evaluation.get("manifest_sha256") != m["sha256"]
        or not all(evaluation.get(k) is True for k in ("complete", "gpu_reload_pass"))
    ):
        raise ContractError("matching complete GPU evaluation required")
    for cohort in ("new", "old"):
        expected = {e["id"] for e in m["episodes"] if e["validation"] and e["cohort"] == cohort}
        if {e["episode"] for e in evaluation["groups"][cohort]} != expected:
            raise ContractError("evaluation episode coverage mismatch")
    path = campaign / "checkpoint_registry.json"
    registry = (
        readapt.checked(path) if path.exists() else dict(schema=readapt.SCHEMA, manifest_sha256=m["sha256"], entries={})
    )
    if registry["manifest_sha256"] != m["sha256"]:
        raise ContractError("registry belongs to another campaign")
    key = str(snapshot.relative_to(campaign))
    entry = dict(
        snapshot_sha256=snapshot_sha,
        evaluation=str(evaluation_path.relative_to(campaign)),
        evaluation_sha256=file_hash(evaluation_path),
        norm_stats_sha256=m["parent"]["norm_stats_sha256"],
        status="SHADOW_ONLY",
        reviewer=reviewer,
        robot_motion_authorized=False,
    )
    if key in registry["entries"] and registry["entries"][key] != entry:
        raise ContractError("registered evidence cannot be replaced")
    registry["entries"][key] = entry
    registry.pop("sha256", None)
    atomic_json(path, readapt.sealed(registry))
    return entry


def registered_config(campaign, snapshot, registry_path):
    campaign, snapshot, registry_path = (
        Path(campaign).resolve(),
        Path(snapshot).resolve(),
        Path(registry_path).resolve(),
    )
    if registry_path != campaign / "checkpoint_registry.json" or not snapshot.is_relative_to(campaign / "snapshots"):
        raise ContractError("registry/snapshot outside campaign")
    m = readapt.verify_campaign(campaign)
    registry = readapt.checked(registry_path)
    entry = registry["entries"].get(str(snapshot.relative_to(campaign)))
    if (
        registry["manifest_sha256"] != m["sha256"]
        or not entry
        or entry.get("status") != "SHADOW_ONLY"
        or entry.get("robot_motion_authorized") is not False
    ):
        raise ContractError("snapshot is not registered for shadow")
    if (
        entry["snapshot_sha256"] != file_hash(snapshot / "snapshot.json")
        or entry["norm_stats_sha256"] != m["parent"]["norm_stats_sha256"]
    ):
        raise ContractError("registered snapshot/statistics identity changed")
    ep = (campaign / entry["evaluation"]).resolve()
    if not ep.is_relative_to(campaign / "evaluations") or file_hash(ep) != entry["evaluation_sha256"]:
        raise ContractError("registered evaluation changed")
    record = snapshot_identity(snapshot)
    evidence = read_json(ep)
    if (
        evidence.get("snapshot_sha256") != entry["snapshot_sha256"]
        or evidence.get("manifest_sha256") != m["sha256"]
        or evidence.get("complete") is not True
        or evidence.get("gpu_reload_pass") is not True
    ):
        raise ContractError("invalid evaluation evidence")
    return configure(campaign, record["recipe"])[0], record


def compare(campaign):
    m = readapt.verify_campaign(campaign)
    candidates = []
    for p in sorted((Path(campaign) / "evaluations").glob("*.json")):
        e = read_json(p)
        if e.get("manifest_sha256") != m["sha256"] or e.get("complete") is not True:
            continue
        rows = e["groups"]["new"]
        misses = sum(int(r["gripper"]["missed_close"]) + int(r["gripper"]["missed_release"]) for r in rows)
        extra = sum(r["gripper"]["extra_toggles"] for r in rows)
        transition = float(np.mean([r["gripper"]["transition_error_rate"] for r in rows]))
        mae = float(np.mean([r["joint_mae_rad"] for r in rows]))
        candidates.append(
            dict(
                experiment=e["experiment"],
                step=e["step"],
                evidence=str(p),
                ranking_key=[misses, extra, transition, mae],
                physical_safety_qualified=False,
            )
        )
    result = dict(
        candidates=sorted(candidates, key=lambda x: x["ranking_key"]),
        decision="OFFLINE_RANKING_ONLY_REQUIRES_FIELD_SAFETY_REVIEW",
        robot_motion_authorized=False,
    )
    atomic_json(Path(campaign) / "candidate_comparison.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["evaluate", "register", "compare"])
    p.add_argument("--campaign", required=True)
    p.add_argument("--snapshot")
    p.add_argument("--reference")
    p.add_argument("--reviewer")
    p.add_argument("--allow-gpu-run", action="store_true")
    a = p.parse_args()
    if a.command == "evaluate":
        if not a.allow_gpu_run or not a.snapshot:
            p.error("--snapshot and --allow-gpu-run required")
        result = evaluate(a.campaign, a.snapshot, reference=a.reference)
    elif a.command == "register":
        if not a.snapshot or not a.reviewer:
            p.error("--snapshot and --reviewer required")
        result = register(a.campaign, a.snapshot, a.reviewer)
    else:
        result = compare(a.campaign)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
