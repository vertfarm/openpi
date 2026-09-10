"""Sequential baseline -> smoke -> N -> M. Does not collect data or control a robot."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from . import readapt
from .artifacts import ContractError
from .artifacts import atomic_json
from .artifacts import file_hash
from .artifacts import read_json
from .artifacts import storage_gate
from .artifacts import tree_bytes


def prune_completed(campaign, name, *, smoke_snapshot=False):
    """Only this campaign's completed/reloaded generated states, never parent/raw."""
    campaign = Path(campaign).resolve()
    if name not in {"N", "M", "smoke"} or (smoke_snapshot and name != "smoke"):
        raise ContractError("invalid generated-state cleanup target")
    m = readapt.verify_campaign(campaign)
    result_path = campaign / "runs" / name / "result.json"
    result = read_json(result_path)
    recipe = readapt.recipe(name, m["sha256"])
    final = recipe["steps"]
    if (
        result.get("complete") is not True
        or result.get("step") != final
        or result["identity"]["manifest_sha256"] != m["sha256"]
    ):
        raise ContractError("run is not complete")
    steps = recipe["snapshots"]
    for step in steps:
        e = read_json(campaign / "evaluations" / f"{name}_{step:06d}.json")
        snap = campaign / "snapshots" / name / f"step_{step:06d}"
        if (
            e.get("complete") is not True
            or e.get("gpu_reload_pass") is not True
            or e.get("snapshot_sha256") != file_hash(snap / "snapshot.json")
        ):
            raise ContractError("all planned inference snapshots must reload before cleanup")
        if name == "smoke" and e.get("full_to_bf16_inference_match") is not True:
            raise ContractError("smoke restart/BF16 equivalence missing")
    target = campaign / "restarts/pi05_hv1" / name
    if smoke_snapshot:
        target = campaign / "snapshots/smoke"
    relative = target.relative_to(campaign)
    # Check lexical ancestors BEFORE resolution, as well as every descendant.
    for p in [campaign, *[campaign.joinpath(*relative.parts[:i]) for i in range(1, len(relative.parts) + 1)]]:
        if p.is_symlink():
            raise ContractError("cleanup refuses filesystem links")
    resolved = target.resolve()
    if resolved != target or not resolved.is_relative_to(campaign) or resolved == campaign:
        raise ContractError("cleanup target is not exact generated subtree")
    if target.exists() and any(p.is_symlink() for p in target.rglob("*")):
        raise ContractError("cleanup subtree contains filesystem link")
    journal = campaign / "runs" / name / ("smoke_snapshot_pruned.json" if smoke_snapshot else "restart_pruned.json")
    if not target.exists():
        if journal.exists() and read_json(journal).get("complete") is True:
            return read_json(journal)
        raise ContractError("cleanup target missing without completed audit")
    record = dict(
        path=str(target),
        bytes=tree_bytes(target),
        manifest_sha256=m["sha256"],
        result_sha256=file_hash(result_path),
        reason="completed GPU-reloaded generated state",
        exact_optimizer_resume_available=False,
        complete=False,
    )
    atomic_json(journal, record)
    shutil.rmtree(target)
    record["complete"] = True
    atomic_json(journal, record)
    return record


def execute(campaign, deadline, reviewer):
    campaign = Path(campaign).resolve()
    m = readapt.verify_campaign(campaign, raw=True)
    readapt.require_normalization_review(campaign)
    if not reviewer.strip():
        raise ContractError("reviewer required for SHADOW_ONLY registration")
    stop = datetime.fromisoformat(deadline)
    if stop.tzinfo is None or stop.timestamp() <= time.time():
        raise ContractError("future timezone-aware deadline required")
    from filelock import FileLock

    # Fail promptly while the existing inference server owns the GPU. Never kill it or ROS.
    with FileLock(str(campaign.parent / "hv1-ml-gpu.lock"), timeout=0):
        storage_gate(campaign)
    from .readapt_config import configure

    for name in ("N", "M", "smoke"):
        recipe = readapt.recipe(name, m["sha256"])
        configure(campaign, recipe)
        schedule = readapt.checked(campaign / f"sampler_{name}.json")
        if schedule != readapt.sample_schedule(m["episodes"], name, recipe["steps"] * recipe["batch_size"]):
            raise ContractError("sample schedule mismatch")

    def child(module, args, label):
        if time.time() >= stop.timestamp():
            raise ContractError("deadline reached before next stage")
        atomic_json(campaign / "status.json", dict(stage=label, robot_commands_sent=0))
        log = campaign / "logs" / f"{label}_{time.time_ns()}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("x") as stream:
            proc = subprocess.run(
                [sys.executable, "-B", "-m", module, *map(str, args)], stdout=stream, stderr=subprocess.STDOUT
            )
        if proc.returncode:
            atomic_json(
                campaign / "status.json",
                dict(stage="NEEDS_ATTENTION", failed_stage=label, log=str(log), robot_commands_sent=0),
            )
            raise ContractError(f"{label} failed; see {log}")

    def eval_snapshot(snapshot, name, step, reference=None):
        evidence = campaign / "evaluations" / f"{name}_{step:06d}.json"
        if evidence.exists():
            e = read_json(evidence)
            if e.get("complete") is not True or e.get("snapshot_sha256") != file_hash(snapshot / "snapshot.json"):
                raise ContractError("existing evaluation incomplete or mismatched; review instead of overwriting")
            return
        args = ["evaluate", "--campaign", campaign, "--snapshot", snapshot, "--allow-gpu-run"]
        if reference:
            args += ["--reference", reference]
        child("examples.hv1.readapt_eval", args, f"evaluate_{name}_{step}")

    with FileLock(str(campaign / "supervisor.lock"), timeout=0):
        eval_snapshot(Path(m["parent"]["snapshot"]), "C_parent", 5000)
        for name in ("smoke", "N", "M"):
            run = campaign / "runs" / name
            completed = (run / "result.json").exists() and read_json(run / "result.json").get("complete") is True
            if name == "smoke" and (run / "smoke_snapshot_pruned.json").exists():
                if read_json(run / "smoke_snapshot_pruned.json").get("complete") is True:
                    continue
            if not completed:
                args = ["--campaign", campaign, "--experiment", name, "--deadline", deadline, "--allow-gpu-run"]
                if (run / "recipe.json").exists():
                    args += ["--resume"]
                child("examples.hv1.readapt_train", args, f"train_{name}")
                if read_json(run / "result.json").get("complete") is not True:
                    raise ContractError("training paused at deadline; no automatic next experiment")
            steps = readapt.recipe(name, m["sha256"])["snapshots"]
            for step in steps:
                snapshot = campaign / "snapshots" / name / f"step_{step:06d}"
                reference = read_json(run / "result.json")["restart_path"] if name == "smoke" else None
                eval_snapshot(snapshot, name, step, reference)
                if name != "smoke":
                    child(
                        "examples.hv1.readapt_eval",
                        ["register", "--campaign", campaign, "--snapshot", snapshot, "--reviewer", reviewer],
                        f"register_{name}_{step}",
                    )
            prune_completed(campaign, name)
            if name == "smoke":
                prune_completed(campaign, name, smoke_snapshot=True)
        child("examples.hv1.readapt_eval", ["compare", "--campaign", campaign], "compare")
        result = dict(
            stage="OFFLINE_COMPLETE_LIVE_BLOCKED",
            experiments=["N", "M"],
            main_snapshots=4,
            robot_commands_sent=0,
            note="No physical rollout or collection success claimed.",
        )
        atomic_json(campaign / "status.json", result)
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign", required=True)
    p.add_argument("--deadline", required=True)
    p.add_argument("--reviewer", required=True)
    p.add_argument("--allow-gpu-run", action="store_true")
    a = p.parse_args()
    if not a.allow_gpu_run:
        p.error("--allow-gpu-run required; no robot motion authorization")
    print(json.dumps(execute(a.campaign, a.deadline, a.reviewer)))


if __name__ == "__main__":
    main()
