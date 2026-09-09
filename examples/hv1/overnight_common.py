"""Small, testable campaign contracts. No model or robot imports."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil

from .workflow import ContractError
from .workflow import read_json

GIB = 1024**3
MIN_FREE = 50 * GIB


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial-{os.getpid()}")
    with temporary.open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def storage_gate(root, write_bytes=0):
    free = shutil.disk_usage(root).free
    if free < MIN_FREE + write_bytes:
        raise ContractError(
            f"disk budget: free={free / GIB:.1f}GiB need={(MIN_FREE + write_bytes) / GIB:.1f}GiB; no raw cleanup"
        )


def tree_bytes(root):
    return sum(p.stat().st_size for p in Path(root).rglob("*") if p.is_file())


def snapshot_steps(name, target):
    candidates = (
        [500, 1000, 2000, target] if name in ("A", "B") else ([target] if name.startswith("smoke") else [250, target])
    )
    return sorted({s for s in candidates if 0 < s <= target})


def select_target(available_seconds, step_seconds, overhead_seconds, *, pair=False):
    if (
        not all(math.isfinite(x) and x >= 0 for x in (available_seconds, step_seconds, overhead_seconds))
        or step_seconds == 0
    ):
        raise ContractError("invalid timing estimate")
    multiplier = 2 if pair else 1
    viable = [
        s
        for s in (500, 1000, 2000, 3000, 5000)
        if multiplier * (overhead_seconds + s * step_seconds * 1.25) <= available_seconds
    ]
    return max(viable, default=0)


def recipe(name, steps, batch=2):
    if name not in ("A", "B", "C", "D", "E", "F", "smoke", "smoke_b1"):
        raise ContractError("unknown experiment")
    if steps < 1 or batch not in (1, 2):
        raise ContractError("invalid steps/batch")
    return dict(
        name=name,
        steps=steps,
        batch_size=batch,
        cohort="inclusive" if name == "B" else "clean",
        seed=7 if name == "E" else 42,
        peak_lr=1e-5 if name == "C" else 2.5e-5,
        decay_lr=1e-6 if name == "C" else 2.5e-6,
        warmup_steps=100,
        decay_steps=5000,
        action_horizon=30 if name == "D" else 15,
        lora=name == "F",
        ema_decay=None,
        snapshots=snapshot_steps(name, steps),
        from_base=True,
        robot_motion_authorized=False,
    )


def verify_campaign(campaign):
    campaign = Path(campaign).resolve()
    scan = read_json(campaign / "scan/scan.json")
    review = read_json(campaign / "validation_review.json")
    if review["scan_sha256"] != scan["manifest_sha256"]:
        raise ContractError("visual review does not match frozen scan")
    ids = {e["id"] for e in scan["episodes"] if e["validation"]}
    if set(review["episodes"]) != ids or any(v["verdict"] != "pass" for v in review["episodes"].values()):
        raise ContractError("all validation episodes must pass visual review")
    for kind, expected in (("clean", 19), ("inclusive", 23)):
        e = read_json(campaign / f"export_{kind}/export.json")
        train = set(e["splits"]["train"]["episode_ids"])
        if len(train) != expected or train & ids or set(e["splits"]["validation"]["episode_ids"]) != ids:
            raise ContractError("cohort/validation split mismatch")
        if e["manifest_sha256"] != scan["manifest_sha256"]:
            raise ContractError("mixed scans")
    return scan
