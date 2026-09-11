"""Sequential smoke -> TODAY30 -> ALL59 execution; never imports ROS or commands a robot."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from . import pipeline
from .artifacts import ContractError
from .artifacts import atomic_json
from .artifacts import file_hash
from .artifacts import read_json
from .artifacts import storage_gate
from .artifacts import tree_bytes


def _run(*args):
    subprocess.run([sys.executable, "-B", *args], check=True)


def _safe_generated_cleanup(campaign, relative, evidence, reason):
    campaign = Path(campaign).resolve()
    target = campaign / relative
    for parent in [campaign, *[campaign.joinpath(*relative.parts[:i]) for i in range(1, len(relative.parts) + 1)]]:
        if parent.is_symlink():
            raise ContractError("cleanup refuses filesystem links")
    resolved = target.resolve()
    if resolved != target or not resolved.is_relative_to(campaign) or resolved == campaign:
        raise ContractError("cleanup target escapes the generated campaign subtree")
    if target.exists() and any(path.is_symlink() for path in target.rglob("*")):
        raise ContractError("cleanup subtree contains a filesystem link")
    journal = campaign / "cleanup" / ("_".join(relative.parts) + ".json")
    if not target.exists():
        if journal.exists() and read_json(journal).get("complete") is True:
            return read_json(journal)
        raise ContractError("cleanup target missing without an audit journal")
    record = dict(
        target=str(relative),
        bytes=tree_bytes(target),
        evidence=str(Path(evidence).relative_to(campaign)),
        evidence_sha256=file_hash(evidence),
        reason=reason,
        recoverable=False,
        complete=False,
    )
    atomic_json(journal, record)
    shutil.rmtree(target)
    record["complete"] = True
    atomic_json(journal, record)
    return record


def _train(campaign, name, deadline):
    args = [
        "-m",
        "examples.hv1.pipeline_train",
        "--campaign",
        str(campaign),
        "--experiment",
        name,
        "--deadline",
        deadline,
        "--allow-gpu-run",
    ]
    if (Path(campaign) / "runs" / name / "recipe.json").exists():
        args.append("--resume")
    _run(*args)


def _evaluate(campaign, name, step, *, reference=None, smoke=False):
    snapshot = campaign / "snapshots" / name / f"step_{step:06d}"
    args = [
        "-m",
        "examples.hv1.pipeline_eval",
        "evaluate",
        "--campaign",
        str(campaign),
        "--snapshot",
        str(snapshot),
        "--allow-gpu-run",
    ]
    if reference:
        args.extend(["--reference", str(reference)])
    if smoke:
        args.append("--smoke")
    _run(*args)


def _cleanup_complete(campaign, relative):
    journal = Path(campaign) / "cleanup" / ("_".join(relative.parts) + ".json")
    return journal.is_file() and read_json(journal).get("complete") is True


def execute(campaign, deadline, reviewer):
    campaign = Path(campaign).resolve()
    manifest = pipeline.verify_campaign(campaign, raw=True)
    if not reviewer.strip():
        raise ContractError("offline registry reviewer identifier is required")
    stop = datetime.fromisoformat(deadline)
    if stop.tzinfo is None or stop.timestamp() <= time.time():
        raise ContractError("future timezone-aware deadline required")
    export = read_json(campaign / "export/export.json")
    if export.get("complete") is not True or export.get("manifest_sha256") != manifest["sha256"]:
        raise ContractError("verified common export for this campaign is required")
    for track in pipeline.TRACKS:
        asset_id = f"hv1_{track.lower()}_{manifest['sha256'][:8]}"
        if not (campaign / "assets" / asset_id / "norm_stats.json").is_file():
            raise ContractError("both track normalization assets are required")
    from filelock import FileLock

    smoke_result_path = campaign / "runs/SMOKE/result.json"
    smoke_complete = smoke_result_path.is_file() and read_json(smoke_result_path).get("complete") is True
    with FileLock(str(campaign.parent / "hv1-ml-gpu.lock"), timeout=0):
        storage_gate(campaign, 0 if smoke_complete else 40 * 1024**3)
    started = time.time()
    smoke_evidence = campaign / "evaluations/SMOKE_000050.json"
    smoke_finished = (
        smoke_complete
        and smoke_evidence.is_file()
        and _cleanup_complete(campaign, Path("restarts/pi05_hv1/SMOKE"))
        and _cleanup_complete(campaign, Path("snapshots/SMOKE"))
    )
    if not smoke_finished:
        if not smoke_complete:
            _train(campaign, "SMOKE", deadline)
        smoke_restart = campaign / "restarts/pi05_hv1/SMOKE/50"
        _evaluate(campaign, "SMOKE", 50, reference=smoke_restart, smoke=True)
        _safe_generated_cleanup(
            campaign,
            Path("restarts/pi05_hv1/SMOKE"),
            smoke_evidence,
            "smoke restart/BF16 equality verified; inference checkpoints have priority",
        )
        _safe_generated_cleanup(
            campaign,
            Path("snapshots/SMOKE"),
            smoke_evidence,
            "diagnostic smoke snapshot verified and is not a deployment candidate",
        )
    completed = []
    for track in pipeline.TRACKS:
        result_path = campaign / "runs" / track / "result.json"
        recipe = pipeline.recipe(track, manifest["sha256"])
        registry = (
            pipeline.checked(campaign / "checkpoint_registry.json")
            if (campaign / "checkpoint_registry.json").is_file()
            else {"entries": {}}
        )
        expected_keys = {
            str((campaign / "snapshots" / track / f"step_{step:06d}").relative_to(campaign))
            for step in recipe["snapshots"]
        }
        track_finished = (
            result_path.is_file()
            and read_json(result_path).get("complete") is True
            and expected_keys <= set(registry["entries"])
            and _cleanup_complete(campaign, Path("restarts/pi05_hv1") / track)
        )
        if track_finished:
            completed.append(track)
            continue
        _train(campaign, track, deadline)
        result = read_json(result_path)
        if result.get("complete") is not True:
            raise ContractError(f"{track} stopped before the common target")
        for step in recipe["snapshots"]:
            _evaluate(campaign, track, step)
            snapshot = campaign / "snapshots" / track / f"step_{step:06d}"
            _run(
                "-m",
                "examples.hv1.pipeline_eval",
                "register",
                "--campaign",
                str(campaign),
                "--snapshot",
                str(snapshot),
                "--reviewer",
                reviewer,
            )
        _safe_generated_cleanup(
            campaign,
            Path("restarts/pi05_hv1") / track,
            result_path,
            "all planned inference snapshots GPU-reloaded and registered SHADOW_ONLY",
        )
        completed.append(track)
    _run("-m", "examples.hv1.pipeline_eval", "compare", "--campaign", str(campaign))
    registry = pipeline.checked(campaign / "checkpoint_registry.json")
    result = dict(
        schema=pipeline.SCHEMA,
        manifest_sha256=manifest["sha256"],
        completed_tracks=completed,
        checkpoint_count=len(registry["entries"]),
        elapsed_seconds=time.time() - started,
        deadline=deadline,
        robot_commands_sent=0,
        deployment_status="SHADOW_ONLY",
        complete=completed == list(pipeline.TRACKS) and len(registry["entries"]) == 8,
    )
    atomic_json(campaign / "campaign_result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--deadline", required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--allow-gpu-run", action="store_true")
    args = parser.parse_args()
    if not args.allow_gpu_run:
        parser.error("explicit --allow-gpu-run required; never authorizes robot motion")
    print(json.dumps(execute(args.campaign, args.deadline, args.reviewer)))


if __name__ == "__main__":
    main()
