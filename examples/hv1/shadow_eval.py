"""Summarize an HV1 ROS shadow rollout without loading saved camera frames."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any, Iterable


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return {
        "count": len(finite),
        "p50": _percentile(finite, 50),
        "p95": _percentile(finite, 95),
        "max": max(finite) if finite else None,
    }


def read_events(path: Path) -> list[dict[str, Any]]:
    events = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            event = json.loads(line)
            if not isinstance(event, dict) or not isinstance(event.get("event"), str):
                raise ValueError(f"invalid event at line {line_number}")
            events.append(event)
    if not events:
        raise ValueError("event log is empty")
    return events


def summarize(events: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(event["event"] for event in events)
    responses = [event for event in events if event["event"] == "response"]
    requests = [event for event in events if event["event"] == "request"]
    schedules = [event for event in events if event["event"] == "schedule"]
    targets = [event for event in events if event["event"] == "target"]
    waiting = [event for event in events if event["event"] == "waiting"]
    faults = [event for event in events if event["event"] == "fault"]
    shutdowns = [event for event in events if event["event"] == "shutdown"]

    source_names = sorted({name for event in requests for name in event.get("source_ages_s", {})})
    source_ages = {
        name: _distribution(
            event["source_ages_s"][name] * 1000 for event in requests if name in event.get("source_ages_s", {})
        )
        for name in source_names
    }

    arm_actions = [event["action"][:7] for event in targets if len(event.get("action", [])) >= 7]
    jumps = []
    per_joint_jumps: list[list[float]] = [[] for _ in range(7)]
    for previous, current in zip(arm_actions, arm_actions[1:]):
        delta = [abs(float(a) - float(b)) for a, b in zip(current, previous, strict=True)]
        jumps.append(max(delta))
        for index, value in enumerate(delta):
            per_joint_jumps[index].append(value)

    tracking = []
    for event in targets:
        action = event.get("action", [])
        measured = event.get("measured", [])
        if len(action) >= 7 and len(measured) >= 7:
            tracking.append(max(abs(float(a) - float(b)) for a, b in zip(action[:7], measured[:7], strict=True)))

    grip_values = [float(event["action"][7]) for event in targets if len(event.get("action", [])) >= 8]
    grip_closed = [value >= 0.5 for value in grip_values]
    grip_crossings = Counter(
        "open_to_close" if current else "close_to_open"
        for previous, current in zip(grip_closed, grip_closed[1:])
        if previous != current
    )

    monotonic = [float(event["monotonic"]) for event in events if "monotonic" in event]
    start = min(monotonic) if monotonic else 0.0
    sent_targets = sum(bool(event.get("sent")) for event in targets)
    command_counts = [int(event["robot_commands_sent"]) for event in events if "robot_commands_sent" in event]

    first_target = targets[0] if targets else None
    first_action = first_target.get("action", [])[:7] if first_target else []
    first_measured = first_target.get("measured", [])[:7] if first_target else []
    first_delta = (
        [float(a) - float(b) for a, b in zip(first_action, first_measured, strict=True)]
        if len(first_action) == len(first_measured) == 7
        else []
    )

    return {
        "event_counts": dict(sorted(counts.items())),
        "duration_s": max(monotonic) - min(monotonic) if monotonic else None,
        "latency_ms": {
            "client": _distribution(event["client_ms"] for event in responses),
            "server": _distribution(event["server_ms"] for event in responses),
            "preprocess": _distribution(event["preprocess_ms"] for event in responses),
        },
        "source_age_ms": source_ages,
        "schedule": {
            "skipped_model_steps": _distribution(event.get("skipped", 0) for event in schedules),
            "first_model_index": _distribution(event.get("first_model_index", 0) for event in schedules),
        },
        "targets": {
            "count": len(targets),
            "sent_true": sent_targets,
            "lateness_ms": _distribution(event.get("lateness_ms", 0) for event in targets),
            "max_abs_step_rad": _distribution(jumps),
            "per_joint_max_abs_step_rad": [max(values) if values else None for values in per_joint_jumps],
            "max_abs_target_minus_measured_rad": _distribution(tracking),
            "gripper_events": dict(
                sorted(Counter(event["gripper_event"] for event in targets if event.get("gripper_event")).items())
            ),
            "gripper_intent": {
                "initial_closed": grip_closed[0] if grip_closed else None,
                "closed_target_fraction": sum(grip_closed) / len(grip_closed) if grip_closed else None,
                "minimum": min(grip_values, default=None),
                "p50": _percentile(grip_values, 50),
                "p95": _percentile(grip_values, 95),
                "maximum": max(grip_values, default=None),
                "threshold_crossings": {
                    "open_to_close": grip_crossings["open_to_close"],
                    "close_to_open": grip_crossings["close_to_open"],
                },
            },
            "gripper_event_offsets_s": [
                {
                    "event": event["gripper_event"],
                    "offset_s": float(event["monotonic"]) - start,
                }
                for event in targets
                if event.get("gripper_event")
            ],
            "first": {
                "offset_s": float(first_target["monotonic"]) - start if first_target else None,
                "action_rad": first_action,
                "measured_rad": first_measured,
                "delta_rad": first_delta,
                "max_abs_delta_rad": max(map(abs, first_delta), default=None),
            },
        },
        "waiting_reasons": dict(sorted(Counter(str(event.get("reason", "")) for event in waiting).items())),
        "fault_reasons": [str(event.get("reason", "")) for event in faults],
        "shutdown_count": len(shutdowns),
        "final_inferences": int(shutdowns[-1].get("inferences", -1)) if shutdowns else None,
        "max_robot_commands_sent": max(command_counts, default=0),
        "shadow_safe": bool(shutdowns) and not faults and sent_targets == 0 and max(command_counts, default=0) == 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("events", type=Path, help="path to a shadow events.jsonl")
    parser.add_argument("--output", type=Path, help="optional JSON summary path")
    args = parser.parse_args()
    result = summarize(read_events(args.events))
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
