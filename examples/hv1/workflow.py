"""Read-only episode QA, sidecar reviews, immutable selections, and causal sampling.

The HDF5 paths are an input contract, NOT a replacement recorder schema.
Only explicit raw roots are scanned; no ROS/controller modules are imported.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import time

import h5py
import numpy as np


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


def validate_profile(p):
    try:
        if p["schema_version"] != 1 or p["status"] not in ("synthetic", "confirmed"):
            raise ContractError("profile must be synthetic or explicitly confirmed, schema_version=1")
        if not isinstance(p["profile_id"], str) or not p["profile_id"].strip():
            raise ContractError("profile_id required")
        for key in ("fps", "max_skew_ms", "max_gap_ms"):
            if not math.isfinite(p[key]) or p[key] <= 0:
                raise ContractError(f"positive finite {key} required")
        if not isinstance(p["action_horizon"], int) or not 1 <= p["action_horizon"] <= 100:
            raise ContractError("action_horizon must be 1..100")
        if p["action"]["semantics"] != "commanded_target":
            raise ContractError("action must identify the commanded target, not measured state")
        if p["action"]["path"] == p["state"]["path"]:
            raise ContractError("state/action cannot use the same dataset")
        for kind in ("state", "action"):
            stream = p[kind]
            names = stream["names"]
            if (
                not all(isinstance(n, str) and n for n in names)
                or not 1 <= len(names) <= 32
                or len(set(names)) != len(names)
            ):
                raise ContractError(f"{kind} requires 1..32 unique names")
            if len(stream["units"]) != len(names) or not all(stream["units"]):
                raise ContractError(f"{kind} requires one unit per dimension")
        indices = p["action"]["delta_state_indices"]
        if len(indices) != len(p["action"]["names"]):
            raise ContractError("delta mapping size mismatch")
        if any(not isinstance(i, int) or i < -1 or i >= len(p["state"]["names"]) for i in indices):
            raise ContractError("invalid delta state index")
        for j, i in enumerate(indices):
            if i >= 0 and p["state"]["units"][i] != p["action"]["units"][j]:
                raise ContractError("delta mapping requires matching units")
        if not p["images"] or set(p["images"]) - {"head", "hand_l", "hand_r"} or "head" not in p["images"]:
            raise ContractError("head required; supported slots: head, hand_l, hand_r")
        for camera in p["images"].values():
            if camera["color_order"] not in ("RGB", "BGR"):
                raise ContractError("explicit RGB/BGR required")
            if len(camera["shape"]) != 3 or camera["shape"][-1] != 3 or min(camera["shape"]) < 1:
                raise ContractError("image shape must be HWC with 3 channels")
        streams = [p["state"], p["action"], *p["images"].values()]
        if not p["clock_domain"] or any(s["clock_domain"] != p["clock_domain"] for s in streams):
            raise ContractError("clock domains differ; require a verified clock conversion before ingestion")
        for s in streams:
            for key in ("path", "timestamps_ns"):
                if not isinstance(s[key], str) or not s[key] or ".." in s[key].split("/"):
                    raise ContractError("invalid HDF5 path")
        for key in ("complete_attribute", "episode_id_attribute", "session_id_attribute", "task_attribute"):
            if not p[key]:
                raise ContractError(f"{key} required")
    except (KeyError, TypeError) as exc:
        raise ContractError(f"missing/invalid profile field: {exc}") from exc
    return p


def dataset(handle, path):
    node = handle
    for part in path.strip("/").split("/"):
        if not isinstance(node, h5py.Group) or part not in node:
            raise ContractError(f"missing dataset {path}")
        if not isinstance(node.get(part, getlink=True), h5py.HardLink):
            raise ContractError(f"linked HDF5 data is unsupported: {path}")
        node = node[part]
    if not isinstance(node, h5py.Dataset):
        raise ContractError(f"expected dataset: {path}")
    return node


def timestamps(handle, stream):
    times = dataset(handle, stream["timestamps_ns"])
    if times.ndim != 1 or times.dtype.kind not in "iu" or len(times) < 2:
        raise ContractError("timestamps must be a nonempty integer-nanosecond vector with >=2 samples")
    value = times[:]
    if np.any(value < 0) or np.any(value > np.iinfo(np.int64).max):
        raise ContractError("timestamp out of int64 range")
    value = value.astype(np.int64)
    if np.any(value[1:] <= value[:-1]):
        raise ContractError("timestamps must strictly increase")
    return value


def causal_indices(source_ns, target_ns, max_skew_ms):
    indices = np.searchsorted(source_ns, target_ns, side="right") - 1
    if np.any(indices < 0):
        raise ContractError("missing past sample at episode start; future samples are not substituted")
    age_ns = target_ns - source_ns[np.maximum(indices, 0)]
    if np.any(age_ns > max_skew_ms * 1e6):
        raise ContractError("source stale / synchronization skew exceeds profile")
    return indices


def sampling_plan(handle, p):
    state_ns = timestamps(handle, p["state"])
    count = int((int(state_ns[-1]) - int(state_ns[0])) * p["fps"] / 1e9) + 1
    if count < 2 or count > 1_000_000:
        raise ContractError("sample count outside supported range 2..1000000")
    grid = int(state_ns[0]) + np.rint(np.arange(count) * (1e9 / p["fps"])).astype(np.int64)
    streams = {"state": p["state"], "action": p["action"], **p["images"]}
    result = {}
    for name, stream in streams.items():
        values = dataset(handle, stream["path"])
        source_ns = timestamps(handle, stream)
        if len(values) != len(source_ns):
            raise ContractError(f"{name}: timestamp/data length mismatch")
        if np.max(np.diff(source_ns)) > p["max_gap_ms"] * 1e6:
            raise ContractError(f"{name}: excessive stream gap")
        if name in ("state", "action"):
            if values.shape != (len(source_ns), len(stream["names"])) or values.dtype.kind not in "fiu":
                raise ContractError(f"{name}: dimension or numeric dtype mismatch")
            for begin in range(0, len(values), 4096):
                if not np.isfinite(values[begin : begin + 4096]).all():
                    raise ContractError(f"{name}: NaN/Inf")
        elif values.shape != (len(source_ns), *stream["shape"]) or values.dtype != np.uint8:
            raise ContractError(f"{name}: require uint8 HWC frames; video sidecars need a separate adapter")
        result[name] = causal_indices(source_ns, grid, p["max_skew_ms"])
    return grid, result


def _attribute(handle, key):
    value = handle.attrs.get(key)
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"missing string attribute {key}")
    return value


def inspect_episode(path, p):
    validate_profile(p)
    path = Path(path).resolve()
    before = file_hash(path)
    report = {
        "path": str(path),
        "sha256": before,
        "profile_sha256": digest(p),
        "profile_id": p["profile_id"],
        "synthetic": p["status"] == "synthetic",
        "errors": [],
    }
    report["id"] = digest({"source": before, "profile": digest(p)})
    try:
        with h5py.File(path, "r") as handle:
            complete = handle.attrs.get(p["complete_attribute"])
            if not isinstance(complete, bool | np.bool_) or not complete:
                raise ContractError("recorder has not explicitly marked this episode complete")
            report.update(
                episode_id=_attribute(handle, p["episode_id_attribute"]),
                session_id=_attribute(handle, p["session_id_attribute"]),
                task=_attribute(handle, p["task_attribute"]),
            )
            grid, indices = sampling_plan(handle, p)
            report.update(
                frames=len(grid),
                duration_s=float(grid[-1] - grid[0]) / 1e9,
                raw_state_frames=len(dataset(handle, p["state"]["path"])),
            )
            # Decode the endpoint frames without loading the entire video into RAM.
            for stream in p["images"].values():
                values = dataset(handle, stream["path"])
                _ = values[0], values[-1]
        if file_hash(path) != before:
            raise ContractError("source changed during validation")
    except (ContractError, OSError, ValueError, KeyError) as exc:
        report["errors"].append(str(exc))
    report["quality_pass"] = not report["errors"]
    return report


def read_frame(path, p, frame, *, verify_sha256=None):
    if verify_sha256 and file_hash(path) != verify_sha256:
        raise ContractError("source hash changed; rescan required")
    with h5py.File(path, "r") as handle:
        grid, indices = sampling_plan(handle, p)
        if not 0 <= frame < len(grid):
            raise ContractError("frame index out of range")
        result = {"timestamp_ns": int(grid[frame]), "images": {}}
        for key in ("state", "action"):
            result[key] = np.asarray(dataset(handle, p[key]["path"])[indices[key][frame]], dtype=np.float32)
        for name, stream in p["images"].items():
            image = dataset(handle, stream["path"])[indices[name][frame]]
            result["images"][name] = image[..., ::-1].copy() if stream["color_order"] == "BGR" else image
        return result


class Catalog:
    def __init__(self, runtime, raw_root, profile):
        self.runtime, self.raw_root = Path(runtime).resolve(), Path(raw_root).resolve()
        if not self.raw_root.is_dir():
            raise ContractError("raw root must be an existing directory")
        if self.runtime == self.raw_root or self.runtime.is_relative_to(self.raw_root):
            raise ContractError("runtime must be outside the immutable raw source root")
        self.profile = validate_profile(profile)
        self.runtime.mkdir(parents=True, exist_ok=True)
        self.db = self.runtime / "catalog.sqlite3"
        with self.connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS episodes (id TEXT PRIMARY KEY, path TEXT, report TEXT, "
                "review TEXT NOT NULL DEFAULT '{}', superseded INTEGER NOT NULL DEFAULT 0)"
            )
            conn.execute("CREATE TABLE IF NOT EXISTS audit (at_ns INTEGER, episode_id TEXT, review TEXT)")

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.db, timeout=10)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def scan(self):
        results = []
        for path in sorted(set(self.raw_root.rglob("*.hdf5")) | set(self.raw_root.rglob("*.h5"))):
            if path.is_symlink() or not path.resolve().is_relative_to(self.raw_root):
                continue
            try:
                report = inspect_episode(path, self.profile)
            except OSError as exc:
                results.append({"path": str(path), "quality_pass": False, "errors": [str(exc)]})
                continue
            with self.connect() as conn:
                conn.execute(
                    "UPDATE episodes SET superseded=1 WHERE path=? AND id<>?", (str(path.resolve()), report["id"])
                )
                conn.execute(
                    "INSERT INTO episodes(id,path,report) VALUES (?,?,?) "
                    "ON CONFLICT(id) DO UPDATE SET report=excluded.report, superseded=0",
                    (report["id"], str(path.resolve()), json.dumps(report)),
                )
            results.append(report)
        return results

    def list(self):
        with self.connect() as conn:
            rows = conn.execute("SELECT report,review,superseded FROM episodes ORDER BY path").fetchall()
        return [
            dict(json.loads(report), review=json.loads(review), superseded=bool(superseded))
            for report, review, superseded in rows
        ]

    def get(self, episode_id):
        for report in self.list():
            if report["id"] == episode_id and report["profile_sha256"] == digest(self.profile):
                if not Path(report["path"]).resolve().is_relative_to(self.raw_root):
                    raise ContractError("catalog source is outside the configured raw root")
                return report
        raise ContractError("unknown episode/profile")

    def review(self, episode_id, outcome, use, reason):
        if outcome not in ("success", "failure", "aborted", "unknown") or type(use) is not bool:
            raise ContractError("invalid review")
        report = self.get(episode_id)
        if use and (not report["quality_pass"] or report["superseded"] or outcome != "success"):
            raise ContractError("baseline selection requires quality-pass, current, successful episode")
        if file_hash(report["path"]) != report["sha256"]:
            raise ContractError("source changed; rescan before review")
        value = {"outcome": outcome, "use": use, "reason": str(reason)[:2000], "updated_ns": time.time_ns()}
        with self.connect() as conn:
            conn.execute("UPDATE episodes SET review=? WHERE id=?", (json.dumps(value), episode_id))
            conn.execute("INSERT INTO audit VALUES (?,?,?)", (time.time_ns(), episode_id, json.dumps(value)))
        return value

    def manifest(self, destination, *, allow_synthetic=False):
        if Path(destination).resolve().is_relative_to(self.raw_root):
            raise ContractError("manifest must not be written into the immutable raw root")
        if self.profile["status"] == "synthetic" and not allow_synthetic:
            raise ContractError("synthetic data requires explicit --allow-synthetic")
        selected = [
            r
            for r in self.list()
            if r["quality_pass"]
            and not r["superseded"]
            and r["profile_sha256"] == digest(self.profile)
            and r["review"].get("use")
            and r["review"].get("outcome") == "success"
        ]
        if not selected:
            raise ContractError("no approved episodes")
        if any(file_hash(r["path"]) != r["sha256"] for r in selected):
            raise ContractError("source changed since review")
        sessions = sorted({r["session_id"] for r in selected}, key=lambda s: digest({"seed": 42, "session": s}))
        validation = set(sessions[-max(1, math.ceil(len(sessions) * 0.2)) :]) if len(sessions) >= 2 else set()
        for report in selected:
            report["split"] = "validation" if report["session_id"] in validation else "train"
        payload = {
            "schema_version": 1,
            "raw_root": str(self.raw_root),
            "profile": self.profile,
            "profile_sha256": digest(self.profile),
            "split_seed": 42,
            "validation_available": bool(validation),
            "episodes": selected,
        }
        payload["manifest_sha256"] = digest(payload)
        write_new_json(destination, payload)
        return payload


def load_manifest(path):
    value = read_json(path)
    expected = value.pop("manifest_sha256")
    if digest(value) != expected or digest(value["profile"]) != value["profile_sha256"]:
        raise ContractError("manifest integrity mismatch")
    validate_profile(value["profile"])
    value["manifest_sha256"] = expected
    if not any(e["split"] == "train" for e in value["episodes"]):
        raise ContractError("empty training split")
    if {e["session_id"] for e in value["episodes"] if e["split"] == "train"} & {
        e["session_id"] for e in value["episodes"] if e["split"] == "validation"
    }:
        raise ContractError("session leakage")
    for episode in value["episodes"]:
        if file_hash(episode["path"]) != episode["sha256"]:
            raise ContractError("manifest source was modified")
    return value
