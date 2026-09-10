"""Persistent, single-GPU overnight campaign. CLI start is explicit authorization to train, never move robots."""

from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .artifacts import ContractError
from .artifacts import atomic_json
from .artifacts import read_json
from .artifacts import storage_gate
from .artifacts import tree_bytes
from .artifacts import write_new_json
from .overnight_common import recipe
from .overnight_common import select_target
from .overnight_common import verify_campaign

TRAIN_END = "2026-09-10T09:30:00+09:00"
END = "2026-09-10T10:00:00+09:00"


def summarize(campaign):
    campaign = Path(campaign)
    records = []
    for path in sorted((campaign / "snapshots").glob("*/step_*/snapshot.json")):
        r = read_json(path)
        name = r["recipe"]["name"]
        step = r["step"]
        full = campaign / "evaluations" / f"{name}_{step:06d}.json"
        quick = campaign / "evaluations" / f"{name}_{step:06d}_quick.json"
        ev = read_json(full if full.exists() else quick) if full.exists() or quick.exists() else {}
        records.append(
            dict(
                experiment=name,
                step=step,
                path=str(path.parent),
                recipe=r["recipe"],
                gpu_reload_pass=ev.get("gpu_reload_pass", False),
                evaluation_complete=ev.get("complete", False),
                validation_loss=ev.get("validation_loss"),
                sweeps={k: v["mean"] for k, v in ev.get("sweeps", {}).items()},
                robot_motion_authorized=False,
            )
        )
    result = dict(
        checkpoints=records,
        count=len(records),
        robot_commands_sent=0,
        complete_gpu_checks=sum(r["gpu_reload_pass"] for r in records),
        generated_at=datetime.now().astimezone().isoformat(),
    )
    atomic_json(campaign / "checkpoint_index.json", result)
    lines = [
        "# HV1 overnight checkpoint index",
        "",
        "Offline evidence only; no robot motion authorization.",
        "",
        "| Experiment | Updates | GPU reload | Validation loss | Path |",
        "|---|---:|---|---:|---|",
    ]
    for r in records:
        lines.append(
            f"| {r['experiment']} | {r['step']} | {r['gpu_reload_pass']} | {r['validation_loss']} | {r['path']} |"
        )
    (campaign / "CHECKPOINTS.md").write_text("\n".join(lines) + "\n")
    return result


def run_child(campaign, tag, args, hard_deadline=None, *, resume_campaign=False):
    campaign = Path(campaign)
    logfile = campaign / "logs" / (tag + ".log")
    logfile.parent.mkdir(exist_ok=True)
    if logfile.exists():
        if not resume_campaign:
            raise ContractError("log already exists: refuse duplicate job")
        # An explicitly restarted supervisor may adopt its still-running child.
        status = read_json(campaign / "status.json")
        pid = status.get("pid") if status.get("phase") == tag else None
        cmdline = Path(f"/proc/{pid}/cmdline") if pid else None
        while cmdline and cmdline.exists():
            live = cmdline.read_bytes().decode().strip("\0").split("\0")
            if not live or live == [""]:
                break
            if live[-len(args) :] != args:
                raise ContractError("refuse to adopt mismatched process")
            atomic_json(
                campaign / "status.json",
                dict(phase=tag, pid=pid, supervisor_pid=os.getpid(), adopted=True, robot_commands_sent=0),
            )
            time.sleep(10)
        # execute() requires the exact recipe/result plus a successful reload below.
        return 0
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED="1", XLA_PYTHON_CLIENT_PREALLOCATE="false", TOKENIZERS_PARALLELISM="false")
    env.pop("JAX_PLATFORMS", None)
    with logfile.open("x") as stream:
        proc = subprocess.Popen([sys.executable, "-u", *args], stdout=stream, stderr=subprocess.STDOUT, env=env)
        while proc.poll() is None:
            atomic_json(
                campaign / "status.json",
                dict(
                    phase=tag,
                    pid=proc.pid,
                    supervisor_pid=os.getpid(),
                    updated_at=datetime.now().astimezone().isoformat(),
                    robot_commands_sent=0,
                ),
            )
            if hard_deadline and time.time() >= hard_deadline:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                break
            time.sleep(10)
    return proc.returncode


def prune_owned_restart(campaign, name):
    """Only this campaign's completed optimizer state, after a successful final GPU reload."""
    campaign = Path(campaign).resolve()
    r = read_json(campaign / "runs" / name / "result.json")
    if not r["complete"]:
        return
    ev = read_json(campaign / "evaluations" / f"{name}_{r['step']:06d}_quick.json")
    if not ev.get("gpu_reload_pass") or not ev.get("complete"):
        raise ContractError("cannot prune without reload verification")
    root = campaign / "restarts" / "pi05_hv1" / name
    expected = root.resolve()
    if root.is_symlink() or expected != root or not expected.is_relative_to(campaign / "restarts"):
        raise ContractError("unsafe restart cleanup target")
    if any(p.is_symlink() for p in root.rglob("*")):
        raise ContractError("linked restart tree")
    if root.exists():
        audit = dict(
            path=str(root),
            bytes=tree_bytes(root),
            reason="completed run; BF16 snapshots retained and final GPU reload verified",
            resume_available=False,
        )
        write_new_json(campaign / "runs" / name / "restart_pruned.json", audit)
        shutil.rmtree(root)


def execute(campaign, name, target, batch, *, smoke=False, resume_campaign=False):
    campaign = Path(campaign)
    r = recipe(name, target, batch)
    recipe_path = campaign / "recipes" / (name + ".json")
    if recipe_path.exists():
        if not resume_campaign or read_json(recipe_path) != r:
            raise ContractError("existing recipe mismatch")
    else:
        write_new_json(recipe_path, r)
    rc = run_child(
        campaign,
        name,
        [
            "-m",
            "examples.hv1.overnight_train",
            "train",
            "--campaign",
            str(campaign),
            "--recipe",
            str(recipe_path),
            "--deadline",
            TRAIN_END,
        ],
        datetime.fromisoformat(END).timestamp() - 300,
        resume_campaign=resume_campaign,
    )
    if rc:
        raise ContractError(f"{name} failed (exit {rc}); inspect logs/{name}.log")
    result = read_json(campaign / "runs" / name / "result.json")
    if result["recipe"] != r:
        raise ContractError("result recipe mismatch")
    if not result["complete"]:
        return result
    snapshot = campaign / "snapshots" / name / f"step_{result['step']:06d}"
    args = [
        "-m",
        "examples.hv1.overnight_eval",
        "--campaign",
        str(campaign),
        "--snapshot",
        str(snapshot),
        "--quick",
        "--deadline",
        END,
    ]
    if smoke:
        args += ["--reference", result["restart_path"]]
    rc = run_child(
        campaign, name + "_reload", args, datetime.fromisoformat(END).timestamp(), resume_campaign=resume_campaign
    )
    if rc:
        raise ContractError(f"{name} reload failed; retain full state, do not start more training")
    if smoke:
        ev = read_json(campaign / "evaluations" / f"{name}_{result['step']:06d}_quick.json")
        if ev.get("full_to_bf16_inference_match") is not True:
            raise ContractError("smoke parity gate failed")
    prune_owned_restart(campaign, name)
    summarize(campaign)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign", required=True)
    p.add_argument("--after-smoke", action="store_true")
    p.add_argument("--resume-campaign", action="store_true")
    a = p.parse_args()
    from filelock import FileLock

    campaign = Path(a.campaign).resolve()
    verify_campaign(campaign)
    stop = datetime.fromisoformat(TRAIN_END).timestamp()
    end = datetime.fromisoformat(END).timestamp()
    with FileLock(str(campaign / "supervisor.lock"), timeout=0):
        try:
            if (campaign / "campaign_result.json").exists():
                raise ContractError("campaign already ended; no implicit rerun")
            if a.after_smoke:
                smoke = read_json(campaign / "runs/smoke/result.json")
                ev = read_json(campaign / "evaluations" / f"smoke_{smoke['step']:06d}_quick.json")
                if not ev.get("full_to_bf16_inference_match"):
                    raise ContractError("smoke not verified")
                prune_owned_restart(campaign, "smoke")
            else:
                try:
                    smoke = execute(campaign, "smoke", 50, 2, smoke=True)
                except ContractError:
                    log = (campaign / "logs/smoke.log").read_text(errors="replace")
                    if not any(k in log.lower() for k in ("out of memory", "resource_exhausted")):
                        raise
                    smoke = execute(campaign, "smoke_b1", 50, 1, smoke=True)
            batch = smoke["recipe"]["batch_size"]
            seconds = smoke["steady_step_seconds"]
            overhead = (
                max(600, smoke["elapsed_seconds"] - 50 * seconds) * 2
            )  # compile + multiple saves + reload reserve.
            allocation = campaign / "allocation.json"
            if a.resume_campaign and allocation.exists():
                target = read_json(allocation)["pair_target"]
            else:
                target = select_target(max(0, stop - time.time()) * 0.5, seconds, overhead, pair=True)
            if target == 0:
                raise ContractError("insufficient time for matched A/B minimum 500 updates")
            if not allocation.exists():
                write_new_json(
                    allocation,
                    dict(
                        pair_target=target,
                        step_seconds=seconds,
                        overhead_seconds=overhead,
                        batch=batch,
                        train_deadline=TRAIN_END,
                        final_deadline=END,
                    ),
                )
            results = []
            skipped = []
            for name in ("A", "B"):
                results.append(execute(campaign, name, target, batch, resume_campaign=a.resume_campaign))
                if not results[-1]["complete"]:
                    break
            if len(results) == 2 and all(r["complete"] for r in results):
                seconds = max(seconds, *[r["steady_step_seconds"] for r in results])
                for i, name in enumerate(("C", "D", "E", "F")):
                    remaining = max(0, stop - time.time())
                    budget = remaining / (4 - i)
                    target = select_target(budget, seconds * (2 if name == "D" else 1), overhead)
                    existing_recipe = campaign / "recipes" / f"{name}.json"
                    if a.resume_campaign and existing_recipe.exists():
                        target = read_json(existing_recipe)["steps"]
                    if target == 0:
                        skipped.append(dict(name=name, reason="time_budget"))
                        continue
                    try:
                        storage_gate(campaign, 80 * 1024**3)
                    except ContractError as exc:
                        skipped.extend(
                            dict(name=n, reason="storage_budget", detail=str(exc)) for n in ("C", "D", "E", "F")[i:]
                        )
                        break
                    results.append(execute(campaign, name, target, batch, resume_campaign=a.resume_campaign))
                    if not results[-1]["complete"]:
                        break
            # Evaluate final A/B first, then matching earlier updates, then optional experiments.
            paths = list((campaign / "snapshots").glob("[A-F]/step_*/snapshot.json"))
            paths.sort(
                key=lambda x: (x.parent.parent.name not in ("A", "B"), -read_json(x)["step"], x.parent.parent.name)
            )
            for path in paths:
                if end - time.time() < 120:
                    break
                record = read_json(path)
                name = record["recipe"]["name"]
                step = record["step"]
                rc = run_child(
                    campaign,
                    f"eval_{name}_{step}",
                    [
                        "-m",
                        "examples.hv1.overnight_eval",
                        "--campaign",
                        str(campaign),
                        "--snapshot",
                        str(path.parent),
                        "--deadline",
                        END,
                    ],
                    end,
                    resume_campaign=a.resume_campaign,
                )
                summarize(campaign)
                if rc:
                    break
            index = summarize(campaign)
            atomic_json(
                campaign / "campaign_result.json",
                dict(
                    status="finished",
                    runs=results,
                    skipped=skipped,
                    checkpoint_count=index["count"],
                    gpu_reload_pass_count=index["complete_gpu_checks"],
                    robot_commands_sent=0,
                ),
            )
            atomic_json(campaign / "status.json", dict(phase="finished", robot_commands_sent=0))
        except Exception as exc:
            summarize(campaign)
            atomic_json(campaign / "status.json", dict(phase="failed", error=str(exc), robot_commands_sent=0))
            raise


if __name__ == "__main__":
    main()
