"""Shared artifact contracts and disk budgets; standard library only.

Immutable evidence uses write_new_json; mutable progress uses atomic_json.
No dataset, model, ROS or robot imports belong in this module.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil

GIB = 1024**3

# Floor of free disk that must survive a reserved write. Lowered from 50 GiB on
# 2026-09-11 by the supervisor: the field machine's root is 1.9 TB at 97% and
# the old floor left no room to run an experiment at all, since a full training
# run reserves 40 GiB on top of it. The reservation is what protects a given
# run; this is the margin left for everything else on the machine.
MIN_FREE = 15 * GIB


class ContractError(ValueError):
    pass


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_new_json(path, value):
    """Never replace a manifest/profile or source file accidentally."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as target:
        json.dump(value, target, ensure_ascii=False, indent=2, allow_nan=False)
        target.write("\n")


def read_json(path):
    with Path(path).open(encoding="utf-8") as source:
        return json.load(source)


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


def sealed(value):
    return dict(value, sha256=digest(value))


def checked(path):
    value = read_json(path)
    if value.get("sha256") != digest({k: v for k, v in value.items() if k != "sha256"}):
        raise ContractError(f"modified manifest: {path}")
    return value
