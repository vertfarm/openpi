"""Read-only ROS/URDF evidence audit. Produces a DRAFT, never an executable profile.

Run again after the owner's edits; a Git commit alone does not identify WIP.
No ROS import, service invocation, recorder change, or hardware communication.
"""

import argparse
import ast
from datetime import UTC
from datetime import datetime
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import yaml

from .workflow import ContractError
from .workflow import file_hash
from .workflow import write_new_json

KH = "ros2/kh_ws/src/"
HAND = "ros2/hand_ws/src/kdex_3f/"
MAP = KH + "keti_humanoid_description/config/urdf_joint_map.yaml"
RECORD = KH + "keti_humanoid_data_collection/config/data_collect_config.yaml"


def git_read(root, *args):
    result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=True)
    return result.stdout.strip()


def urdf_joints(path):
    """CAD limits are evidence, NOT qualified hardware limits."""
    root = ET.parse(path).getroot()
    return [
        {
            "name": j.attrib["name"],
            "type": j.attrib["type"],
            "axis": None if j.find("axis") is None else j.find("axis").get("xyz"),
            "limit_unverified": None if j.find("limit") is None else dict(j.find("limit").attrib),
        }
        for j in root.findall("joint")
        if j.attrib["type"] != "fixed"
    ]


def collect_topics(value, prefix=""):
    found = {}
    if isinstance(value, dict):
        if "topic" in value:
            found[prefix] = {key: value[key] for key in ("topic", "type", "key") if key in value}
        for key, child in value.items():
            found.update(collect_topics(child, f"{prefix}.{key}".strip(".")))
    return found


def audit(root):
    root = Path(root).resolve(strict=True)
    paths = {
        MAP,
        RECORD,
        KH + "keti_humanoid_description/urdf/kh_upper_body.urdf",
        KH + "keti_humanoid_controller/keti_humanoid_controller/upper_body_controller.py",
        KH + "keti_humanoid_controller/keti_humanoid_controller/upper_body_mqtt_bridge.py",
        KH + "keti_humanoid_interfaces/srv/RecordCommand.srv",
        KH + "keti_humanoid_interfaces/msg/RecorderStatus.msg",
        KH + "keti_humanoid_bringup/config/hardware.yaml",
        KH + "keti_humanoid_bringup/config/keti_humanoid_config.yaml",
        KH + "keti_humanoid_bringup/launch/upper_body.launch.py",
        HAND + "kdex_3f_ros2/config/kdex_3f.yaml",
        HAND + "kdex_3f_ros2_msgs/msg/SpacemouseAction.msg",
        HAND + "kdex_3f_ros2_msgs/msg/GraspState.msg",
        HAND + "kdex_3f_ros2_msgs/srv/SetOpen.srv",
    }
    hand_urdfs = {side: HAND + f"kdex_3f_description/urdf/ftservo/kdex_3f_{side}.urdf" for side in ("left", "right")}
    paths.update(hand_urdfs.values())
    package = root / (KH + "keti_humanoid_data_collection")
    implementation = []
    for path in package.rglob("*.py"):
        if (
            path.parent.name == "keti_humanoid_data_collection"
            and path.name != "setup.py"
            and any(
                isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
                for n in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
            )
        ):
            implementation.append(path.relative_to(root).as_posix())
        paths.add(path.relative_to(root).as_posix())
    existing = [p for p in sorted(paths) if (root / p).is_file()]
    for relative in existing:
        if not (root / relative).resolve().is_relative_to(root):
            raise ContractError("evidence path escapes repository")
    before = {p: file_hash(root / p) for p in existing}
    head = git_read(root, "rev-parse", "HEAD")
    wip = git_read(root, "status", "--porcelain")
    mapping = yaml.safe_load((root / MAP).read_text())["joint_map"]
    if [r["index"] for r in mapping] != list(range(16)):
        raise ContractError("wire map changed; review before deriving the 14-arm candidate")
    arms = [r for r in mapping if r["urdf"] is not None]
    if len(arms) != 14 or len({r["urdf"] for r in arms}) != 14:
        raise ContractError("expected 14 distinct modeled arm joints; source review required")
    params = yaml.safe_load((root / RECORD).read_text())["keti_humanoid_data_collection"]["ros__parameters"]
    upper = urdf_joints(root / (KH + "keti_humanoid_description/urdf/kh_upper_body.urdf"))
    if not {r["urdf"] for r in arms} <= {r["name"] for r in upper}:
        raise ContractError("wire map names are not present in the URDF")
    hands = {side: urdf_joints(root / path) for side, path in hand_urdfs.items()}
    state_names = [r["urdf"] for r in arms] + [f"{side}/{j['name']}" for side in hands for j in hands[side]]
    action_names = [r["urdf"] for r in arms] + ["left/open", "right/open"]
    result = {
        "schema_version": 1,
        "status": "draft_source_inferred",
        "executable_profile": False,
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "ros_commit": head,
        "ros_wip": wip.splitlines(),
        "hand_commit": git_read(root / HAND, "rev-parse", "HEAD"),
        "hand_wip": git_read(root / HAND, "status", "--porcelain").splitlines(),
        "source_sha256": before,
        "missing_source_files": sorted(paths - set(existing)),
        "recorder_python_implementation_candidates": implementation,
        "configured_topics_not_runtime_verified": collect_topics(params),
        "configured_recording": params.get("recording"),
        "configured_format": params.get("format"),
        "configured_record_interface": params.get("record_interface"),
        "configured_output_dir_not_mount_verified": params.get("output_dir"),
        "wire_joint_map": mapping,
        "urdf_arms": upper,
        "urdf_hands": hands,
        "candidate_not_frozen": {
            "state_names": state_names,
            "state_units": ["rad"] * len(state_names),
            "action_names": action_names,
            "action_units": ["rad"] * len(arms) + ["fraction"] * 2,
            "delta_state_indices": [*list(range(len(arms))), -1, -1],
            "assumptions": [
                "hand joint feedback is actually available in radians",
                "one validated fixed grasp mode per session; commanded open is available",
                "arm actions are solved commanded joint targets, not measured state or input twist",
            ],
            "excluded": ["reserved head slots 0/1", "torque/stop/mode buttons", "velocity/effort control"],
        },
        "required_before_confirmation": [
            "HDF5 groups/datasets/units and finalization marker: unknown until writer exists",
            "HEVC file names, frame PTS and raw capture clock mapping: unknown; no frame-index/fps guess",
            "actual topic namespaces/types/remaps, hand accepted command versus requested intent",
            "MQTT raw receive timestamp/sequence: ROS republish time does not establish freshness",
            "real camera identity/FPS/skew budget; YAML 30 Hz and 1 s are not accepted VLA limits",
            "process task/prompt and action contract; example pick object is not the user's task",
            "real joint sign/zero/limits, elbow workspace, hand/TCP geometry, payload and gain revision",
            "exclusive command ownership and local stop/hold/fault behavior before any actuation",
        ],
    }
    if head != git_read(root, "rev-parse", "HEAD") or before != {p: file_hash(root / p) for p in existing}:
        raise ContractError("source changed during audit; rerun without touching the owner's files")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ros-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if Path(args.output).resolve().is_relative_to(Path(args.ros_root).resolve()):
        parser.error("write evidence outside the ROS repository")
    result = audit(args.ros_root)
    write_new_json(args.output, result)
    print(f"DRAFT source evidence saved: {args.output}; not a training or actuation profile")


if __name__ == "__main__":
    main()
