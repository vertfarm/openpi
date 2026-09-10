"""Recipes and frozen cohort checks for the completed A-F campaign.

Artifact helpers remain re-exported for compatibility; new code uses artifacts.
"""

from __future__ import annotations

import math
from pathlib import Path

from .artifacts import GIB as GIB
from .artifacts import MIN_FREE as MIN_FREE
from .artifacts import ContractError
from .artifacts import atomic_json as atomic_json
from .artifacts import read_json
from .artifacts import storage_gate as storage_gate
from .artifacts import tree_bytes as tree_bytes


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
