"""The hold published during a fault must also be loggable.

A ROS message turns an assigned list into an array.array, and the JSON log
cannot serialise that. On 2026-09-11 the first live rollout published its hold
and then died writing the log line, which skipped the rest of the fault path.
"""

import array
import json

from examples.hv1.ros.keti_humanoid_inference.keti_humanoid_inference.core import JsonLog


def test_positions_read_back_from_a_ros_message_are_not_json_serialisable(tmp_path):
    """Why hold_here keeps its own list instead of reading the field back."""
    assigned = array.array("d", [0.1, 0.2, 0.3])
    try:
        json.dumps({"position": assigned})
    except TypeError:
        return
    raise AssertionError("array.array serialised; hold_here no longer needs its own list")


def test_hold_event_survives_the_log(tmp_path):
    log = JsonLog(tmp_path / "events.jsonl")
    log.write("hold", why="target step", position=[float(v) for v in array.array("d", [0.1, 0.2])])
    log.close()
    written = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert written[0]["event"] == "hold"
    assert written[0]["position"] == [0.1, 0.2]
