"""Diagnostic replay, GPU reload proof, and SHADOW_ONLY registry for two-track models."""

from __future__ import annotations

import argparse
import base64
from functools import lru_cache
import json
from pathlib import Path
import time

import numpy as np

from . import two_track
from .artifacts import ContractError
from .artifacts import atomic_json
from .artifacts import file_hash
from .artifacts import read_json
from .artifacts import write_new_json
from .checkpoints import snapshot_identity
from .native import CAMERAS
from .native import PROMPT
from .ros.keti_humanoid_inference.keti_humanoid_inference.core import IntentConditioner
from .two_track_config import configure
from .two_track_config import local_dataset

SNAPSHOT_STEPS = (250, 500, 1000, 2000)


def _edges(values):
    closed = np.asarray(values) >= 0.5
    changes = np.flatnonzero(np.diff(np.r_[False, closed]) != 0)
    return [int(i) for i in changes if closed[i]], [int(i) for i in changes if not closed[i]]


def _ordered_match(predicted, expected, tolerance):
    predicted, expected = tuple(predicted), tuple(expected)

    @lru_cache(maxsize=None)
    def solve(i, j):
        if i == len(predicted) or j == len(expected):
            return 0, 0, ()
        options = [solve(i + 1, j), solve(i, j + 1)]
        if abs(predicted[i] - expected[j]) <= tolerance:
            matched, cost, pairs = solve(i + 1, j + 1)
            options.append((matched + 1, cost + abs(predicted[i] - expected[j]), ((i, j), *pairs)))
        return max(options, key=lambda value: (value[0], -value[1]))

    _, _, pairs = solve(0, 0)
    return [(predicted[i], expected[j]) for i, j in pairs]


def transition_metrics(predicted, truth, close_frames, release_frames, fps=30):
    predicted = np.asarray(predicted) >= 0.5
    truth = np.asarray(truth) >= 0.5
    if predicted.shape != truth.shape or predicted.ndim != 1 or not len(predicted):
        raise ContractError("aligned nonempty intent series required")
    predicted_close, predicted_release = _edges(predicted)
    close_pairs = _ordered_match(predicted_close, close_frames, fps)
    release_pairs = _ordered_match(predicted_release, release_frames, fps)
    around = np.zeros(len(predicted), dtype=bool)
    for frame in [*close_frames, *release_frames]:
        around[max(0, frame - fps) : min(len(predicted), frame + fps + 1)] = True
    return dict(
        error_rate=float(np.mean(predicted != truth)),
        transition_error_rate=float(np.mean((predicted != truth)[around])),
        expected_close_frames=close_frames,
        expected_release_frames=release_frames,
        predicted_close_frames=predicted_close,
        predicted_release_frames=predicted_release,
        matched_close=[[p, t] for p, t in close_pairs],
        matched_release=[[p, t] for p, t in release_pairs],
        close_time_error_s=[(p - t) / fps for p, t in close_pairs],
        release_time_error_s=[(p - t) / fps for p, t in release_pairs],
        missed_close=len(close_frames) - len(close_pairs),
        missed_release=len(release_frames) - len(release_pairs),
        extra_close=len(predicted_close) - len(close_pairs),
        extra_release=len(predicted_release) - len(release_pairs),
        initial_closed=bool(predicted[0]),
        note="Teacher-forced recorded observations; not measured physical grasp success.",
    )


def condition_intents(values, *, times=None, fps=30, close_threshold=0.7, open_threshold=0.3, min_hold_s=0.2):
    """Apply the same deploy-time hysteresis/dwell conditioner to an intent timeline."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ContractError("finite nonempty scalar intent series required")
    times = np.arange(len(values), dtype=np.float64) / fps if times is None else np.asarray(times, dtype=np.float64)
    if times.shape != values.shape or not np.isfinite(times).all() or np.any(np.diff(times) < 0):
        raise ContractError("monotonic intent times required")
    conditioner = IntentConditioner(
        close_threshold=close_threshold,
        open_threshold=open_threshold,
        min_hold_s=min_hold_s,
    )
    filtered, events = [], []
    for value, now in zip(values, times, strict=True):
        event = conditioner.update(value, now)
        if event:
            events.append({"event": event, "time_s": float(now)})
        filtered.append(float(conditioner.closed))
    return np.asarray(filtered, dtype=np.float32), events


def _observation(raw):
    return dict(
        state=np.asarray(raw["observation.state"]),
        prompt=PROMPT,
        images={camera: np.asarray(raw[f"observation.images.{camera}"]) for camera in CAMERAS},
    )


def evaluate(campaign, snapshot, *, reference=None, smoke=False, intent_sidecar=False):
    campaign, snapshot = Path(campaign).resolve(), Path(snapshot).resolve()
    manifest = two_track.verify_campaign(campaign, raw=True)
    if not snapshot.is_relative_to(campaign / "snapshots"):
        raise ContractError("snapshot is outside the two-track campaign")
    record = snapshot_identity(snapshot)
    recipe = record["recipe"]
    if recipe != two_track.recipe(recipe["name"], manifest["sha256"], recipe["steps"]):
        raise ContractError("snapshot recipe/manifest lineage mismatch")
    if smoke != (recipe["name"] == "SMOKE"):
        raise ContractError("smoke evaluation flag/recipe mismatch")
    config, export = configure(campaign, recipe)
    from filelock import FileLock
    import jax

    from openpi.policies.policy_config import create_trained_policy

    data = config.data.create(config.assets_dirs, config.model)
    dataset = local_dataset(data, config.model, export["splits"]["train"]["root"])
    offsets, offset = {}, 0
    for episode in manifest["episodes"]:
        offsets[episode["id"]] = offset
        offset += episode["frames"]
    diagnostics = [episode for episode in manifest["episodes"] if episode["diagnostic_group"]]
    if smoke:
        diagnostics = diagnostics[:1]
    selected = set(manifest["tracks"][recipe["track"]]["episode_ids"])
    result = dict(
        schema=two_track.SCHEMA,
        manifest_sha256=manifest["sha256"],
        snapshot=str(snapshot),
        snapshot_sha256=file_hash(snapshot / "snapshot.json"),
        experiment=recipe["name"],
        track=recipe["track"],
        step=record["step"],
        denoise=10,
        prefix=3,
        groups={"old_fixed6": [], "today_fixed6": []},
        diagnostic_role="training_overlap_is_reported_per_episode",
        independent_common_validation=False,
        robot_commands_sent=0,
        closed_loop=False,
        physical_safety_qualified=False,
        success_rate_measured=False,
        normalization_sha256=record["norm_stats_sha256"],
    )
    raw_intents = []
    with FileLock(str(campaign.parent / "hv1-ml-gpu.lock"), timeout=0):
        first = diagnostics[0]
        first_raw = dataset[offsets[first["id"]]]
        first_observation = _observation(first_raw)
        noise = np.random.default_rng(42).normal(size=(15, 32)).astype(np.float32)
        if reference:
            reference = Path(reference).resolve()
            if not reference.is_relative_to(campaign / "restarts"):
                raise ContractError("reference must be this campaign's optimizer checkpoint")
            policy = create_trained_policy(config, reference, default_prompt=PROMPT, sample_kwargs={"num_steps": 10})
            reference_actions = np.asarray(policy.infer(first_observation, noise=noise)["actions"])
            del policy
            import gc

            gc.collect()
            jax.clear_caches()
        else:
            reference_actions = None
        policy = create_trained_policy(config, snapshot, default_prompt=PROMPT, sample_kwargs={"num_steps": 10})
        warm = np.asarray(policy.infer(first_observation, noise=noise)["actions"])
        if reference_actions is not None:
            np.testing.assert_allclose(warm, reference_actions, atol=1e-5, rtol=1e-5)
            result["full_to_bf16_inference_match"] = True
        for episode in diagnostics:
            predictions, truths, latencies, first_steps, chunk_steps = [], [], [], [], []
            base = offsets[episode["id"]]
            for frame in range(0, episode["frames"], 3):
                raw = dataset[base + frame]
                state = np.asarray(raw["observation.state"])
                seeded = np.random.default_rng(42 + base + frame).normal(size=(15, 32)).astype(np.float32)
                started = time.perf_counter()
                prediction = np.asarray(policy.infer(_observation(raw), noise=seeded)["actions"])
                latencies.append((time.perf_counter() - started) * 1000)
                if prediction.shape != (15, 8) or not np.isfinite(prediction).all():
                    raise ContractError("invalid model prediction")
                count = min(3, episode["frames"] - frame)
                predictions.extend(prediction[:count])
                truths.extend(np.asarray(raw["action"])[:count])
                first_steps.append(float(np.max(np.abs(prediction[0, :7] - state[:7]))))
                chunk_steps.append(float(np.max(np.abs(np.diff(prediction[:, :7], axis=0)))))
            prediction, truth = np.asarray(predictions), np.asarray(truths)
            raw_intents.append(
                {
                    "episode": episode["id"],
                    "frames": episode["frames"],
                    "training_overlap": episode["id"] in selected,
                    "suspect": episode["suspect"],
                    "grasp_frames": episode["grasp_frames"],
                    "release_frames": episode["release_frames"],
                    "predicted_grasp_intent": prediction[:, 7].tolist(),
                    "truth_grasp_intent": truth[:, 7].tolist(),
                }
            )
            result["groups"][episode["diagnostic_group"]].append(
                dict(
                    episode=episode["id"],
                    frames=episode["frames"],
                    training_overlap=episode["id"] in selected,
                    suspect=episode["suspect"],
                    joint_mae_rad=float(np.mean(np.abs(prediction[:, :7] - truth[:, :7]))),
                    prefix_series_max_step_rad=float(np.max(np.abs(np.diff(prediction[:, :7], axis=0)))),
                    chunk_max_step_rad=max(chunk_steps),
                    first_target_delta_rad=max(first_steps),
                    latency_p95_ms=float(np.percentile(latencies, 95)),
                    gripper=transition_metrics(
                        prediction[:, 7],
                        truth[:, 7],
                        episode["grasp_frames"],
                        episode["release_frames"],
                    ),
                )
            )
            print(json.dumps(dict(evaluated=episode["id"], experiment=recipe["name"])), flush=True)
    if not smoke and any(len(result["groups"][group]) != 6 for group in result["groups"]):
        raise ContractError("both fixed diagnostic groups must contain six episodes")
    result.update(gpu_reload_pass=True, complete=True, evaluated_rows=sum(e["frames"] for e in diagnostics))
    if intent_sidecar:
        sidecar = {
            "schema": "hv1_grasp_intent_series_v1",
            "manifest_sha256": manifest["sha256"],
            "snapshot": str(snapshot),
            "snapshot_sha256": result["snapshot_sha256"],
            "experiment": recipe["name"],
            "step": record["step"],
            "denoise": 10,
            "fps": 30,
            "closed_loop": False,
            "robot_commands_sent": 0,
            "episodes": raw_intents,
        }
        directory = campaign / "evaluations" / "intent_series"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{recipe['name']}_{record['step']:06d}.json"
        write_new_json(path, sidecar)
        return {
            "complete": True,
            "intent_sidecar": str(path),
            "snapshot_sha256": result["snapshot_sha256"],
            "episodes": len(raw_intents),
            "robot_commands_sent": 0,
        }
    path = campaign / "evaluations" / f"{recipe['name']}_{record['step']:06d}.json"
    write_new_json(path, result)
    return result


def register(campaign, snapshot, reviewer):
    campaign, snapshot = Path(campaign).resolve(), Path(snapshot).resolve()
    manifest = two_track.verify_campaign(campaign)
    record = snapshot_identity(snapshot)
    if (
        not reviewer.strip()
        or not snapshot.is_relative_to(campaign / "snapshots")
        or record["recipe"]["name"] not in two_track.TRACKS
        or record.get("manifest_sha256") != manifest["sha256"]
    ):
        raise ContractError("reviewer and matching two-track snapshot required")
    evidence_path = campaign / "evaluations" / f"{record['recipe']['name']}_{record['step']:06d}.json"
    evidence = read_json(evidence_path)
    snapshot_sha256 = file_hash(snapshot / "snapshot.json")
    if (
        evidence.get("snapshot_sha256") != snapshot_sha256
        or evidence.get("manifest_sha256") != manifest["sha256"]
        or evidence.get("gpu_reload_pass") is not True
        or any(len(evidence["groups"][group]) != 6 for group in ("old_fixed6", "today_fixed6"))
    ):
        raise ContractError("complete matching diagnostic evaluation required")
    path = campaign / "checkpoint_registry.json"
    registry = (
        two_track.checked(path)
        if path.exists()
        else dict(schema=two_track.SCHEMA, manifest_sha256=manifest["sha256"], entries={})
    )
    key = str(snapshot.relative_to(campaign))
    entry = dict(
        track=record["recipe"]["track"],
        step=record["step"],
        snapshot_sha256=snapshot_sha256,
        evaluation=str(evidence_path.relative_to(campaign)),
        evaluation_sha256=file_hash(evidence_path),
        norm_stats_sha256=record["norm_stats_sha256"],
        status="SHADOW_ONLY",
        reviewer=reviewer,
        robot_motion_authorized=False,
    )
    if key in registry["entries"] and registry["entries"][key] != entry:
        raise ContractError("registered evidence cannot be replaced")
    registry["entries"][key] = entry
    registry.pop("sha256", None)
    atomic_json(path, two_track.sealed(registry))
    return entry


def registered_config(campaign, snapshot, registry_path):
    campaign, snapshot, registry_path = map(Path, (campaign, snapshot, registry_path))
    campaign, snapshot, registry_path = campaign.resolve(), snapshot.resolve(), registry_path.resolve()
    if registry_path != campaign / "checkpoint_registry.json" or not snapshot.is_relative_to(campaign / "snapshots"):
        raise ContractError("registry/snapshot outside campaign")
    manifest = two_track.verify_campaign(campaign)
    registry = two_track.checked(registry_path)
    entry = registry["entries"].get(str(snapshot.relative_to(campaign)))
    if (
        registry.get("manifest_sha256") != manifest["sha256"]
        or not entry
        or entry.get("status") != "SHADOW_ONLY"
        or entry.get("robot_motion_authorized") is not False
    ):
        raise ContractError("snapshot is not registered for shadow")
    record = snapshot_identity(snapshot)
    if (
        entry["snapshot_sha256"] != file_hash(snapshot / "snapshot.json")
        or entry["norm_stats_sha256"] != record["norm_stats_sha256"]
    ):
        raise ContractError("registered snapshot/statistics identity changed")
    evidence_path = (campaign / entry["evaluation"]).resolve()
    if not evidence_path.is_relative_to(campaign / "evaluations") or file_hash(evidence_path) != entry["evaluation_sha256"]:
        raise ContractError("registered evaluation changed")
    evidence = read_json(evidence_path)
    if evidence.get("snapshot_sha256") != entry["snapshot_sha256"] or evidence.get("gpu_reload_pass") is not True:
        raise ContractError("invalid evaluation evidence")
    return configure(campaign, record["recipe"])[0], record


def compare(campaign):
    campaign = Path(campaign).resolve()
    manifest = two_track.verify_campaign(campaign)
    candidates = []
    for path in sorted((campaign / "evaluations").glob("*.json")):
        evidence = read_json(path)
        if (
            evidence.get("manifest_sha256") != manifest["sha256"]
            or evidence.get("complete") is not True
            or evidence.get("experiment") not in two_track.TRACKS
        ):
            continue
        rows = [row for group in evidence["groups"].values() for row in group]
        missed = sum(row["gripper"]["missed_close"] + row["gripper"]["missed_release"] for row in rows)
        extra = sum(row["gripper"]["extra_close"] + row["gripper"]["extra_release"] for row in rows)
        transition = float(np.mean([row["gripper"]["transition_error_rate"] for row in rows]))
        joint_mae = float(np.mean([row["joint_mae_rad"] for row in rows]))
        candidates.append(
            dict(
                experiment=evidence["experiment"],
                step=evidence["step"],
                evidence=str(path),
                ranking_key=[missed, extra, transition, joint_mae],
                independent_common_validation=False,
                physical_safety_qualified=False,
            )
        )
    result = dict(
        manifest_sha256=manifest["sha256"],
        candidates=sorted(candidates, key=lambda value: value["ranking_key"]),
        decision="OFFLINE_DIAGNOSTIC_RANKING_ONLY_REQUIRES_SHADOW_AND_FIELD_REVIEW",
        robot_motion_authorized=False,
    )
    atomic_json(campaign / "candidate_comparison.json", result)
    return result


def _read_events(path):
    events = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            event = json.loads(line)
            if not isinstance(event, dict) or not isinstance(event.get("event"), str):
                raise ContractError(f"invalid shadow event at {path}:{line_number}")
            events.append(event)
    return events


def _correct_hand_shadows(root):
    """Select the valid correct-hand log with most scheduled targets for each checkpoint."""
    selected = {}
    ignored = []
    for path in sorted(Path(root).glob("hv1_shadow_*_correcthand_r*/events.jsonl")):
        events = _read_events(path)
        startup = next((event for event in events if event["event"] == "startup"), None)
        targets = [event for event in events if event["event"] == "target" and len(event.get("action", [])) >= 8]
        metadata = {} if startup is None else startup.get("metadata", {})
        key = (metadata.get("experiment"), metadata.get("step"))
        if key[0] not in two_track.TRACKS or key[1] not in SNAPSHOT_STEPS or not targets:
            ignored.append(str(path))
            continue
        value = {"path": str(path), "events": events, "targets": targets}
        if key not in selected or len(targets) > len(selected[key]["targets"]):
            if key in selected:
                ignored.append(selected[key]["path"])
            selected[key] = value
        else:
            ignored.append(str(path))
    expected = {(track, step) for track in two_track.TRACKS for step in SNAPSHOT_STEPS}
    if set(selected) != expected:
        missing = sorted(expected - set(selected))
        raise ContractError(f"missing correct-hand shadow logs: {missing}")
    return selected, ignored


def evaluate_static_intents(campaign, snapshot, shadow_log):
    """Run a fine-tuned policy on saved correct-hand observations without ROS."""
    campaign, snapshot, shadow_log = map(
        lambda value: Path(value).resolve(), (campaign, snapshot, shadow_log)
    )
    manifest = two_track.verify_campaign(campaign)
    if not snapshot.is_relative_to(campaign / "snapshots") or not shadow_log.is_relative_to(campaign / "shadow"):
        raise ContractError("snapshot/static shadow evidence must stay inside the campaign")
    record = snapshot_identity(snapshot)
    recipe = record["recipe"]
    if (
        recipe.get("name") not in two_track.FILTER_FINETUNES
        or recipe != two_track.recipe(recipe["name"], manifest["sha256"], recipe["steps"])
    ):
        raise ContractError("static replay requires a filter fine-tune snapshot")
    events = _read_events(shadow_log)
    startup = next((event for event in events if event["event"] == "startup"), None)
    targets = [event for event in events if event["event"] == "target"]
    if not startup or not targets:
        raise ContractError("static shadow evidence lacks startup/target events")
    parent = recipe["parent"]
    metadata = startup.get("metadata", {})
    if metadata.get("experiment") != parent["experiment"] or metadata.get("step") != parent["step"]:
        raise ContractError("static shadow log does not match the fine-tune parent")
    predictions = {}
    config, _ = configure(campaign, recipe)
    from filelock import FileLock
    from openpi.policies.policy_config import create_trained_policy
    from .deploy_server import prepare_images

    sequences = sorted({int(event["prediction"]["sequence"]) for event in targets})
    with FileLock(str(campaign.parent / "hv1-ml-gpu.lock"), timeout=0):
        policy = create_trained_policy(config, snapshot, default_prompt=PROMPT, sample_kwargs={"num_steps": 10})
        for sequence in sequences:
            path = shadow_log.parent / f"observation_{sequence:06d}.npz"
            if not path.is_file():
                raise ContractError(f"missing static observation: {path}")
            with np.load(path, allow_pickle=False) as saved:
                state = np.asarray(saved["state"], dtype=np.float32)
                if state.shape != (15,) or not np.isfinite(state).all():
                    raise ContractError("invalid saved static state")
                payload = {
                    "images": {
                        camera: base64.b64encode(np.asarray(saved[camera], dtype=np.uint8).tobytes()).decode("ascii")
                        for camera in CAMERAS
                    },
                    "image_encoding": "ros_compressed",
                }
            images = prepare_images(payload)
            noise = np.random.default_rng(900_000 + sequence).normal(size=(15, 32)).astype(np.float32)
            actions = np.asarray(
                policy.infer(
                    {"state": state, "images": images, "prompt": PROMPT}, noise=noise
                )["actions"]
            )
            if actions.shape != (15, 8) or not np.isfinite(actions).all():
                raise ContractError("invalid static replay prediction")
            predictions[sequence] = actions
    times, intents = [], []
    for target in targets:
        prediction = target["prediction"]
        sequence, model_index = int(prediction["sequence"]), int(prediction["model_index"])
        if sequence not in predictions or not 0 <= model_index < 15:
            raise ContractError("invalid static target provenance")
        times.append(float(target["monotonic"]))
        intents.append(float(predictions[sequence][model_index, 7]))
    times = (np.asarray(times) - times[0]).tolist()
    value = {
        "schema": "hv1_static_grasp_intent_series_v1",
        "manifest_sha256": manifest["sha256"],
        "snapshot": str(snapshot),
        "snapshot_sha256": file_hash(snapshot / "snapshot.json"),
        "experiment": recipe["name"],
        "step": record["step"],
        "parent_shadow_log": str(shadow_log),
        "source_experiment": metadata["experiment"],
        "source_step": metadata["step"],
        "times_s": times,
        "predicted_grasp_intent": intents,
        "inference_sequences": len(sequences),
        "targets": len(targets),
        "robot_commands_sent": 0,
        "closed_loop": False,
    }
    directory = campaign / "evaluations" / "static_intent_series"
    path = directory / f"{recipe['name']}_{record['step']:06d}.json"
    write_new_json(path, value)
    return {
        "complete": True,
        "static_intent_sidecar": str(path),
        "targets": len(targets),
        "robot_commands_sent": 0,
    }


def sweep_filter(campaign, shadow_root, output, *, hold_times=(0.1, 0.2, 0.3, 0.5)):
    """CPU-only filter sweep over saved teacher-forced and scheduled shadow intents."""
    campaign, output = Path(campaign).resolve(), Path(output).resolve()
    manifest = two_track.verify_campaign(campaign)
    shadows, ignored = _correct_hand_shadows(shadow_root)
    rows = []
    for hold in hold_times:
        for track in two_track.TRACKS:
            for step in SNAPSHOT_STEPS:
                sidecar_path = campaign / "evaluations" / "intent_series" / f"{track}_{step:06d}.json"
                sidecar = read_json(sidecar_path)
                if (
                    sidecar.get("schema") != "hv1_grasp_intent_series_v1"
                    or sidecar.get("manifest_sha256") != manifest["sha256"]
                    or sidecar.get("experiment") != track
                    or sidecar.get("step") != step
                    or len(sidecar.get("episodes", [])) != 12
                ):
                    raise ContractError(f"invalid intent sidecar: {sidecar_path}")
                metrics = []
                for episode in sidecar["episodes"]:
                    filtered, _ = condition_intents(
                        episode["predicted_grasp_intent"],
                        fps=sidecar["fps"],
                        min_hold_s=hold,
                    )
                    metrics.append(
                        transition_metrics(
                            filtered,
                            episode["truth_grasp_intent"],
                            episode["grasp_frames"],
                            episode["release_frames"],
                            fps=sidecar["fps"],
                        )
                    )
                shadow = shadows[(track, step)]
                targets = shadow["targets"]
                times = np.asarray([float(event["monotonic"]) for event in targets])
                times -= times[0]
                _, shadow_events = condition_intents(
                    [event["action"][7] for event in targets],
                    times=times,
                    min_hold_s=hold,
                )
                close_errors = [error for metric in metrics for error in metric["close_time_error_s"]]
                missed_close = sum(metric["missed_close"] for metric in metrics)
                extra_close = sum(metric["extra_close"] for metric in metrics)
                missed_release = sum(metric["missed_release"] for metric in metrics)
                extra_release = sum(metric["extra_release"] for metric in metrics)
                shadow_close = sum(event["event"] == "close" for event in shadow_events)
                max_abs_close_error = max(map(abs, close_errors), default=None)
                rows.append(
                    {
                        "min_hold_s": hold,
                        "experiment": track,
                        "step": step,
                        "missed_close": missed_close,
                        "extra_close": extra_close,
                        "missed_release": missed_release,
                        "extra_release": extra_release,
                        "median_close_time_error_s": float(np.median(close_errors)) if close_errors else None,
                        "max_abs_close_time_error_s": max_abs_close_error,
                        "shadow_false_close": shadow_close,
                        "shadow_exposure_s": float(times[-1]) if len(times) else 0.0,
                        "shadow_log": shadow["path"],
                        "pass": missed_close == 0
                        and extra_close == 0
                        and shadow_close == 0
                        and max_abs_close_error is not None
                        and max_abs_close_error <= 0.3,
                    }
                )
    common = [
        hold
        for hold in hold_times
        if all(row["pass"] for row in rows if row["min_hold_s"] == hold)
    ]
    result = {
        "schema": "hv1_grasp_filter_sweep_v1",
        "manifest_sha256": manifest["sha256"],
        "close_threshold": 0.7,
        "open_threshold": 0.3,
        "hold_times_s": list(hold_times),
        "rows": rows,
        "common_passing_hold_times_s": common,
        "decision": "SUPERVISOR_SELECTION_REQUIRED" if common else "CONDITIONAL_TRAINING_CRITERION_MET",
        "ignored_shadow_logs": ignored,
        "robot_commands_sent": 0,
        "closed_loop": False,
    }
    write_new_json(output, result)
    return result


def sweep_filter_finetunes(campaign, output, *, hold_times=(0.1, 0.2, 0.3, 0.5)):
    """CPU-only sweep for the conditional parent-warm-start snapshots."""
    campaign, output = Path(campaign).resolve(), Path(output).resolve()
    manifest = two_track.verify_campaign(campaign)
    rows = []
    for hold in hold_times:
        for experiment in two_track.FILTER_FINETUNES:
            for step in (500, 1000):
                teacher = read_json(
                    campaign
                    / "evaluations"
                    / "intent_series"
                    / f"{experiment}_{step:06d}.json"
                )
                static = read_json(
                    campaign / "evaluations" / "static_intent_series" / f"{experiment}_{step:06d}.json"
                )
                if (
                    teacher.get("schema") != "hv1_grasp_intent_series_v1"
                    or static.get("schema") != "hv1_static_grasp_intent_series_v1"
                    or teacher.get("manifest_sha256") != manifest["sha256"]
                    or static.get("manifest_sha256") != manifest["sha256"]
                    or teacher.get("experiment") != experiment
                    or static.get("experiment") != experiment
                    or teacher.get("step") != step
                    or static.get("step") != step
                ):
                    raise ContractError("filter fine-tune intent evidence identity mismatch")
                metrics = []
                for episode in teacher["episodes"]:
                    filtered, _ = condition_intents(
                        episode["predicted_grasp_intent"], fps=teacher["fps"], min_hold_s=hold
                    )
                    metrics.append(
                        transition_metrics(
                            filtered,
                            episode["truth_grasp_intent"],
                            episode["grasp_frames"],
                            episode["release_frames"],
                            fps=teacher["fps"],
                        )
                    )
                _, static_events = condition_intents(
                    static["predicted_grasp_intent"], times=static["times_s"], min_hold_s=hold
                )
                close_errors = [error for metric in metrics for error in metric["close_time_error_s"]]
                missed_close = sum(metric["missed_close"] for metric in metrics)
                extra_close = sum(metric["extra_close"] for metric in metrics)
                shadow_false_close = sum(event["event"] == "close" for event in static_events)
                max_error = max(map(abs, close_errors), default=None)
                rows.append(
                    {
                        "min_hold_s": hold,
                        "experiment": experiment,
                        "step": step,
                        "missed_close": missed_close,
                        "extra_close": extra_close,
                        "missed_release": sum(metric["missed_release"] for metric in metrics),
                        "extra_release": sum(metric["extra_release"] for metric in metrics),
                        "median_close_time_error_s": float(np.median(close_errors)) if close_errors else None,
                        "max_abs_close_time_error_s": max_error,
                        "shadow_false_close": shadow_false_close,
                        "static_targets": static["targets"],
                        "pass": missed_close == 0
                        and extra_close == 0
                        and shadow_false_close == 0
                        and max_error is not None
                        and max_error <= 0.3,
                    }
                )
    passing = [row for row in rows if row["pass"]]
    result = {
        "schema": "hv1_grasp_filter_finetune_sweep_v1",
        "manifest_sha256": manifest["sha256"],
        "close_threshold": 0.7,
        "open_threshold": 0.3,
        "hold_times_s": list(hold_times),
        "rows": rows,
        "passing": passing,
        "decision": "SUPERVISOR_SELECTION_REQUIRED" if passing else "FILTER_FINETUNE_DID_NOT_MEET_CRITERIA",
        "robot_commands_sent": 0,
        "closed_loop": False,
    }
    write_new_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=[
            "evaluate",
            "evaluate-intents",
            "evaluate-static",
            "register",
            "compare",
            "sweep-filter",
            "sweep-filter-finetunes",
        ],
    )
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--snapshot")
    parser.add_argument("--reference")
    parser.add_argument("--reviewer")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--allow-gpu-run", action="store_true")
    parser.add_argument("--shadow-root")
    parser.add_argument("--output")
    parser.add_argument("--static-log")
    args = parser.parse_args()
    if args.command in ("evaluate", "evaluate-intents"):
        if not args.allow_gpu_run or not args.snapshot:
            parser.error("--snapshot and --allow-gpu-run required")
        result = evaluate(
            args.campaign,
            args.snapshot,
            reference=args.reference,
            smoke=args.smoke,
            intent_sidecar=args.command == "evaluate-intents",
        )
    elif args.command == "evaluate-static":
        if not args.allow_gpu_run or not args.snapshot or not args.static_log:
            parser.error("evaluate-static requires --snapshot, --static-log and --allow-gpu-run")
        result = evaluate_static_intents(args.campaign, args.snapshot, args.static_log)
    elif args.command == "register":
        if not args.snapshot or not args.reviewer:
            parser.error("--snapshot and --reviewer required")
        result = register(args.campaign, args.snapshot, args.reviewer)
    elif args.command == "compare":
        result = compare(args.campaign)
    elif args.command == "sweep-filter":
        if not args.shadow_root or not args.output:
            parser.error("sweep-filter requires --shadow-root and --output")
        result = sweep_filter(args.campaign, args.shadow_root, args.output)
    else:
        if not args.output:
            parser.error("sweep-filter-finetunes requires --output")
        result = sweep_filter_finetunes(args.campaign, args.output)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
