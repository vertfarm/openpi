"""Versioned, operator-reviewed re-acquisition. No ROS or robot command imports."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
from datetime import timezone
import json
from pathlib import Path
import random
import subprocess

import numpy as np

from . import native
from .artifacts import ContractError
from .artifacts import checked as checked
from .artifacts import digest
from .artifacts import file_hash
from .artifacts import read_json
from .artifacts import sealed as sealed
from .artifacts import write_new_json
from .checkpoints import snapshot_identity

PLACEMENTS = ("center", "front", "back", "left", "right")
SCHEMA = "hv1_readapt_v1"


def external_output(output, *sources):
    out = Path(output).resolve()
    if any(out == Path(s).resolve() or out.is_relative_to(Path(s).resolve()) for s in sources):
        raise ContractError("output must be outside source custody")
    if {p.lower() for p in out.parts} & {"raw", "datasets"}:
        raise ContractError("runtime output cannot be in raw/datasets")
    return out


def uid(session, episode):
    if not session or "::" in session or not episode.startswith("episode_") or "::" in episode:
        raise ContractError("invalid session/episode identity")
    return f"{session}::{episode}"


def source_hashes(root):
    """Include dirty and untracked source/config; exclude build/runtime trees."""
    root = Path(root).resolve()
    paths = [
        p
        for p in (root / "ros2/kh_ws/src").rglob("*")
        if p.is_file()
        and p.suffix in {".py", ".yaml", ".yml", ".urdf", ".xacro", ".srdf", ".xml"}
        and not set(p.parts) & {"__pycache__", "build", "install", "log", ".git"}
    ]
    if not paths:
        raise ContractError("ROS source/config files not found")
    return {str(p.relative_to(root)): file_hash(p) for p in sorted(paths)}


def capture_session(ros_root, output):
    ros_root = Path(ros_root).resolve()
    out = external_output(output, ros_root)
    sources = source_hashes(ros_root)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ros_root, check=True, capture_output=True, text=True
    ).stdout.strip()
    value = sealed(
        dict(
            schema=SCHEMA,
            captured_at=datetime.now(timezone.utc).isoformat(),
            ros_root=str(ros_root),
            git_head=head,
            source_hashes=sources,
            runtime_applied=False,
            note="File capture is NOT proof of loaded parameters.",
        )
    )
    write_new_json(out, value)
    return value


def scan_session(raw, capture, output):
    raw = Path(raw).resolve()
    cap = checked(capture)
    if source_hashes(cap["ros_root"]) != cap["source_hashes"]:
        raise ContractError("controller changed since capture; create a new capture/session")
    output = external_output(output, raw, cap["ros_root"])
    if output.exists():
        raise ContractError("scan output must be new")
    meta = read_json(raw / "metadata.json")
    session = meta["dataset_name"]
    if session == "keti_humanoid_data_260909":
        raise ContractError("today requires a separate recorder session, not the old raw directory")
    if float(meta.get("hz", 0)) != 30 or set(meta.get("cameras", {})) != set(native.CAMERAS):
        raise ContractError("30Hz three-camera metadata required")
    if any(c.get("resolution") != [640, 480] or c.get("rotate") != 0 for c in meta["cameras"].values()):
        raise ContractError("recorder camera geometry changed")
    records = []
    for indexed in meta["episodes"]:
        eid = indexed["id"]
        identity = uid(session, eid)
        directory = (raw / "episodes" / eid).resolve()
        if not directory.is_relative_to(raw / "episodes"):
            raise ContractError("episode path escapes raw")
        record = dict(id=identity, episode_id=eid, session_id=session, path=str(directory), cohort="new")
        try:
            hashes = {p.name: file_hash(p) for p in native.source_files(directory)}
            _, _, info = native.read_numeric(directory / "data.hdf5")
            task = read_json(directory / "tasks.json")
            if (
                indexed["num_frames"] != info["frames"]
                or task["num_frames"] != info["frames"]
                or task["episode_id"] != eid
            ):
                raise ContractError("recorder index/tasks/numeric length mismatch")
            # Decode every source frame. This validates files, not visual success.
            for _ in native.decode_episode(directory, info["frames"]):
                pass
            if hashes != {p.name: file_hash(p) for p in native.source_files(directory)}:
                raise ContractError("source changed while scanning")
            record.update(
                info,
                source_hashes=hashes,
                numeric_pass=True,
                source_task=task["main_prompt"],
                task=native.PROMPT,
                release_tail_s=(info["frames"] - info["release_frame"]) / 30,
            )
        except (OSError, ValueError, KeyError) as exc:
            record.update(numeric_pass=False, error=str(exc))
        records.append(record)
    result = sealed(
        dict(
            schema=SCHEMA,
            raw_root=str(raw),
            session_id=session,
            capture=cap,
            raw_metadata_sha256=file_hash(raw / "metadata.json"),
            episodes=records,
        )
    )
    write_new_json(output / "scan.json", result)
    review = dict(
        scan_sha256=result["sha256"],
        reviewer="",
        configuration_applied_at="",
        collection_started_at="",
        owner_confirmed_unchanged_during_collection=False,
        runtime_parameters_evidence="",
        physical_contract_unchanged=False,
        episodes={
            e["id"]: dict(
                kind="pending",
                placement="",
                visual_success=False,
                no_intervention=False,
                no_jitter_or_collision=False,
                transitions_verified=False,
                notes="",
            )
            for e in records
        },
    )
    write_new_json(output / "review.template.json", review)
    return result


def collection_schedule():
    rng = random.Random(42)
    result = []
    for block in range(8):
        order = list(PLACEMENTS)
        rng.shuffle(order)
        base = len(result)
        result.extend(dict(trial=base + i + 1, block=block + 1, placement=p) for i, p in enumerate(order))
    return result


def review_selection(scan, review):
    if (
        review.get("scan_sha256") != scan["sha256"]
        or not isinstance(review.get("reviewer"), str)
        or not review["reviewer"].strip()
    ):
        raise ContractError("review must identify scan and reviewer")
    if not all(
        review.get(k)
        for k in (
            "owner_confirmed_unchanged_during_collection",
            "physical_contract_unchanged",
            "runtime_parameters_evidence",
        )
    ):
        raise ContractError("owner/runtime/physical contract confirmation missing")
    try:
        applied = datetime.fromisoformat(review["configuration_applied_at"])
        started = datetime.fromisoformat(review["collection_started_at"])
        if applied.tzinfo is None or started.tzinfo is None or applied > started:
            raise ValueError("invalid times")
    except (ValueError, KeyError) as exc:
        raise ContractError("timezone-aware applied <= collection start required") from exc
    accepted, diagnostic = [], 0
    if set(review.get("episodes", {})) != {e["id"] for e in scan["episodes"]}:
        raise ContractError("review episode set mismatch")
    for e in scan["episodes"]:
        decision = review["episodes"][e["id"]]
        kind = decision.get("kind")
        if kind == "diagnostic":
            diagnostic += 1
            continue
        if kind == "exclude":
            continue
        if kind != "success":
            raise ContractError("unreviewed episode; mark success, diagnostic or exclude")
        if not e.get("numeric_pass") or e["release_tail_s"] < 2:
            raise ContractError("success requires numeric QA and >=2s release tail")
        if not all(
            decision.get(k) is True
            for k in ("visual_success", "no_intervention", "no_jitter_or_collision", "transitions_verified")
        ):
            raise ContractError("visual/motion/transition review missing")
        if datetime.fromisoformat(e["created_at"]) < started:
            raise ContractError("episode predates frozen collection")
        accepted.append(dict(e, placement=decision["placement"]))
    if (
        diagnostic < 3
        or len(accepted) != 40
        or Counter(e["placement"] for e in accepted) != Counter({p: 8 for p in PLACEMENTS})
    ):
        raise ContractError("need >=3 diagnostics and 40 reviewed successes: 8 per placement")
    rng = random.Random(42)
    # Eight validation episodes: two in three deterministically chosen strata, one in the other two.
    extra = set(rng.sample(list(PLACEMENTS), 3))
    result = []
    for placement in PLACEMENTS:
        group = sorted((e for e in accepted if e["placement"] == placement), key=lambda e: e["id"])
        rng.shuffle(group)
        for i, e in enumerate(group):
            result.append(dict(e, validation=i < (2 if placement in extra else 1), suspect=False))
    return sorted(result, key=lambda e: e["id"])


def verify_files(record):
    root = Path(record["path"]).resolve()
    expected = {p.name for p in native.source_files(root)}
    if set(record["source_hashes"]) != expected:
        raise ContractError("incomplete source file manifest")
    for name, sha in record["source_hashes"].items():
        if file_hash(root / name) != sha:
            raise ContractError(f"raw changed: {record['id']}/{name}")


def parent_record(parent):
    parent = Path(parent).resolve()
    snapshot = parent / "snapshots/C/step_005000"
    record = snapshot_identity(snapshot)
    if record.get("complete") is not True or record.get("step") != 5000 or record["recipe"]["name"] != "C":
        raise ContractError("complete C-5000 parent required")
    if record.get("profile") != native.profile():
        raise ContractError("parent observation/action contract differs")
    stats = parent / "shared_assets/hv1_common_clean19/norm_stats.json"
    if file_hash(stats) != record["norm_stats_sha256"]:
        raise ContractError("parent normalization changed")
    evaluation = read_json(parent / "evaluations/C_005000.json")
    if evaluation.get("complete") is not True or evaluation.get("gpu_reload_pass") is not True:
        raise ContractError("parent reload evaluation missing")
    return dict(
        campaign=str(parent),
        snapshot=str(snapshot),
        snapshot_sha256=file_hash(snapshot / "snapshot.json"),
        norm_stats=str(stats),
        norm_stats_sha256=file_hash(stats),
        prior_updates=5000,
    )


def old_review_template(parent, output):
    scan = read_json(Path(parent) / "scan/scan.json")
    ids = [e["id"] for e in scan["episodes"] if not e["validation"] and not e["suspect"]]
    value = dict(
        parent_manifest_sha256=scan["manifest_sha256"],
        reviewer="",
        episodes={eid: dict(approved=False, video_and_command_reviewed=False, notes="") for eid in ids},
    )
    write_new_json(output, value)
    return value


def prepare(scan_path, review_path, old_review_path, parent, output):
    scan, review = checked(scan_path), read_json(review_path)
    out = external_output(output, scan["raw_root"], parent, scan["capture"]["ros_root"])
    if out.exists():
        raise ContractError("campaign must be new")
    if source_hashes(scan["capture"]["ros_root"]) != scan["capture"]["source_hashes"]:
        raise ContractError("controller changed during collection; split and re-review session")
    new = review_selection(scan, review)
    initial = parent_record(parent)
    old_scan = read_json(Path(parent) / "scan/scan.json")
    parent_meta = read_json(Path(initial["snapshot"]) / "snapshot.json")
    if (
        old_scan["manifest_sha256"] != digest({k: v for k, v in old_scan.items() if k != "manifest_sha256"})
        or old_scan["manifest_sha256"] != parent_meta["manifest_sha256"]
    ):
        raise ContractError("old source manifest no longer matches C-5000")
    old_review = read_json(old_review_path)
    eligible = {e["id"] for e in old_scan["episodes"] if not e["validation"] and not e["suspect"]}
    if eligible != set(parent_meta["train_episode_ids"]):
        raise ContractError("old clean pool differs from C-5000 training identities")
    if old_review.get("parent_manifest_sha256") != old_scan["manifest_sha256"] or not old_review.get("reviewer"):
        raise ContractError("old data review must identify frozen parent and reviewer")
    if set(old_review["episodes"]) != eligible:
        raise ContractError("old review may select only original clean training pool")
    old = []
    for e in old_scan["episodes"]:
        approved = old_review["episodes"].get(e["id"], {})
        if not e["validation"] and not (
            approved.get("approved") is True and approved.get("video_and_command_reviewed") is True
        ):
            continue
        old.append(dict(e, id=uid(e["session_id"], e["id"]), episode_id=e["id"], cohort="old"))
    if not any(not e["validation"] for e in old):
        raise ContractError("M requires reviewed old training episodes")
    for e in new + old:
        verify_files(e)
    value = sealed(
        dict(
            schema=SCHEMA,
            parent=initial,
            profile=native.profile(),
            prompt=native.PROMPT,
            capture=scan["capture"],
            source_scan_sha256=scan["sha256"],
            review_sha256=digest(review),
            old_review_sha256=digest(old_review),
            episodes=new + old,
            robot_motion_authorized=False,
            statistics_policy="reuse_C5000_unchanged",
            split_seed=42,
        )
    )
    write_new_json(out / "manifest.json", value)
    # Reuse the proven decoder/exporter without its 9/9 scan/ID selection rules.
    export_scan = dict(profile=value["profile"], raw_root=scan["raw_root"], episodes=new + old)
    export_scan["manifest_sha256"] = digest(export_scan)
    write_new_json(out / "export_scan.json", export_scan)
    return value


def verify_campaign(campaign, *, raw=False):
    campaign = Path(campaign).resolve()
    m = checked(campaign / "manifest.json")
    if m["schema"] != SCHEMA or m["profile"] != native.profile():
        raise ContractError("unsupported readaptation contract")
    parent = m["parent"]
    if (
        file_hash(Path(parent["snapshot"]) / "snapshot.json") != parent["snapshot_sha256"]
        or file_hash(parent["norm_stats"]) != parent["norm_stats_sha256"]
    ):
        raise ContractError("parent snapshot/statistics identity changed")
    ids = [e["id"] for e in m["episodes"]]
    if len(ids) != len(set(ids)):
        raise ContractError("duplicate qualified episode identity")
    if raw:
        for e in m["episodes"]:
            verify_files(e)
    return m


def recipe(name, manifest_sha):
    if name not in {"N", "M", "smoke"}:
        raise ContractError("only N/M and diagnostic smoke are supported")
    return dict(
        name=name,
        steps=50 if name == "smoke" else 5000,
        batch_size=2,
        seed=42,
        manifest_sha256=manifest_sha,
        peak_lr=1e-5,
        decay_lr=1e-6,
        warmup_steps=100,
        decay_steps=5000,
        action_horizon=15,
        lora=False,
        ema_decay=None,
        new_fraction=0.8 if name == "M" else 1.0,
        phase_fractions={"uniform": 0.70, "close": 0.15, "release": 0.15},
        snapshots=[50] if name == "smoke" else [2000, 5000],
        initialization="C5000_new_optimizer",
        robot_motion_authorized=False,
    )


def sample_schedule(episodes, name, samples=10000):
    if samples <= 0 or samples % 100:
        raise ContractError("sample count must be a positive multiple of 100")
    r = recipe(name, "schedule")
    train = [e for e in episodes if not e["validation"]]
    offsets, start = {}, 0
    for e in train:
        offsets[e["id"]] = start
        start += e["frames"]
    pools = {c: sorted([e for e in train if e["cohort"] == c], key=lambda e: e["id"]) for c in ("new", "old")}
    if not pools["new"] or (name == "M" and not pools["old"]):
        raise ContractError("empty sampling pool")
    rng = random.Random(r["seed"])
    cycles = {"new": [], "old": []}
    records = []
    for _ in range(samples // 100):
        assignments = []
        for cohort, fraction in (("new", r["new_fraction"]), ("old", 1 - r["new_fraction"])):
            for phase, share in r["phase_fractions"].items():
                assignments += [(cohort, phase)] * round(100 * fraction * share)
        rng.shuffle(assignments)
        for cohort, phase in assignments:
            if not cycles[cohort]:
                cycles[cohort] = list(pools[cohort])
                rng.shuffle(cycles[cohort])
            e = cycles[cohort].pop()
            lo, hi = 0, e["frames"] - 1
            if phase != "uniform":
                center = e["grasp_frame" if phase == "close" else "release_frame"]
                lo, hi = max(0, center - 30), min(hi, center + 30)
            frame = rng.randint(lo, hi)
            records.append(
                dict(index=offsets[e["id"]] + frame, episode=e["id"], frame=frame, cohort=cohort, phase=phase)
            )
    return sealed(
        dict(schema=SCHEMA, name=name, samples=samples, records=records, train_episode_order=[e["id"] for e in train])
    )


def coverage(schedule, consumed):
    rows = schedule["records"][:consumed]
    return dict(
        consumed_samples=len(rows),
        unique_anchors=len({(e["episode"], e["frame"]) for e in rows}),
        cohort_counts=dict(Counter(e["cohort"] for e in rows)),
        phase_counts=dict(Counter(e["phase"] for e in rows)),
        episode_counts=dict(Counter(e["episode"] for e in rows)),
    )


def normalization_report(campaign):
    m = verify_campaign(campaign, raw=True)
    stats = read_json(m["parent"]["norm_stats"])
    stats = stats.get("norm_stats", stats)
    groups = {}
    for cohort in ("new", "old"):
        states, actions = [], []
        for e in m["episodes"]:
            if e["validation"] or e["cohort"] != cohort:
                continue
            s, a, _ = native.read_numeric(Path(e["path"]) / "data.hdf5")
            idx = np.minimum(np.arange(len(s))[:, None] + np.arange(15), len(s) - 1)
            chunks = a[idx].copy()
            chunks[:, :, :7] -= s[:, None, :7]
            states.append(s)
            actions.append(chunks.reshape(-1, 8))
        groups[cohort] = {}
        for key, arrays in (("state", states), ("actions", actions)):
            values = np.concatenate(arrays)
            low, high = np.array(stats[key]["q01"]), np.array(stats[key]["q99"])
            z = (values - low) / (high - low + 1e-6) * 2 - 1
            groups[cohort][key] = dict(
                outside_quantiles_fraction=np.mean(np.abs(z) > 1, axis=0).tolist(),
                normalized_min=z.min(axis=0).tolist(),
                normalized_max=z.max(axis=0).tolist(),
                near_constant_stats_dims=np.flatnonzero(high - low < 1e-6).tolist(),
            )
    value = sealed(
        dict(
            manifest_sha256=m["sha256"],
            norm_stats_sha256=m["parent"]["norm_stats_sha256"],
            groups=groups,
            statistics_changed=False,
            requires_review=True,
        )
    )
    write_new_json(Path(campaign) / "normalization_report.json", value)
    write_new_json(
        Path(campaign) / "normalization_review.template.json",
        dict(
            report_sha256=value["sha256"],
            reviewer="",
            approved=False,
            units_sign_zero_mapping_confirmed=False,
            distribution_shift_reviewed=False,
            notes="",
        ),
    )
    return value


def require_normalization_review(campaign):
    report = checked(Path(campaign) / "normalization_report.json")
    review = read_json(Path(campaign) / "normalization_review.json")
    if (
        review.get("report_sha256") != report["sha256"]
        or not review.get("reviewer")
        or not all(
            review.get(k) is True
            for k in ("approved", "units_sign_zero_mapping_confirmed", "distribution_shift_reviewed")
        )
    ):
        raise ContractError("normalization/units review is not approved")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("capture-session")
    c.add_argument("--ros-root", required=True)
    c.add_argument("--output", required=True)
    c = sub.add_parser("schedule")
    c.add_argument("--output", required=True)
    c = sub.add_parser("scan")
    c.add_argument("--raw", required=True)
    c.add_argument("--capture", required=True)
    c.add_argument("--output", required=True)
    c = sub.add_parser("old-review-template")
    c.add_argument("--parent", required=True)
    c.add_argument("--output", required=True)
    c = sub.add_parser("prepare")
    for k in ("scan", "review", "old-review", "parent", "output"):
        c.add_argument("--" + k, required=True)
    for command in ("export", "normalization", "status"):
        c = sub.add_parser(command)
        c.add_argument("--campaign", required=True)
    a = p.parse_args()
    if a.command == "capture-session":
        result = capture_session(a.ros_root, a.output)
    elif a.command == "schedule":
        result = dict(diagnostics=3, trials=collection_schedule(), prompt=native.PROMPT)
        write_new_json(a.output, result)
    elif a.command == "scan":
        result = scan_session(a.raw, a.capture, a.output)
    elif a.command == "old-review-template":
        result = old_review_template(a.parent, a.output)
    elif a.command == "prepare":
        result = prepare(a.scan, a.review, a.old_review, a.parent, a.output)
    elif a.command == "export":
        verify_campaign(a.campaign, raw=True)
        result = native.export(Path(a.campaign) / "export_scan.json", Path(a.campaign) / "export", "inclusive")
        for name in ("N", "M", "smoke"):
            m = checked(Path(a.campaign) / "manifest.json")
            r = recipe(name, m["sha256"])
            write_new_json(
                Path(a.campaign) / f"sampler_{name}.json",
                sample_schedule(m["episodes"], name, r["steps"] * r["batch_size"]),
            )
    elif a.command == "normalization":
        result = normalization_report(a.campaign)
    else:
        exists = (Path(a.campaign) / "manifest.json").exists()
        started = any((Path(a.campaign) / "runs").glob("*/recipe.json"))
        result = dict(
            state="CHECK_RUN_RESULTS"
            if started
            else "PREPARED_REQUIRES_QA"
            if exists
            else "WAITING_FOR_NEW_REVIEWED_DATA",
            training_started=started,
            note="inspect runs/*/result.json for actual execution; status never starts a job",
        )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
