"""Independent qualification of HV1 VLA proposals. Commands nothing, ever.

The executor refuses to move without a fresh `/hv1_vla/guardian` message whose
flags are all true, so this node is the thing that says "keep going". It holds
no publisher to any robot topic: the only way it can affect the arm is by going
quiet, which stops the executor within one lease.

Every flag it raises is checked against a live observation, except the two the
field cannot evidence in software. Those stay false until an operator passes
them explicitly, because a guardian that reports constants is worse than none:
it converts an unverified condition into a recorded assurance.

Stopping this node (Ctrl-C) stops the executor within 100 ms. That is the
operator's software handle, under the physical E-stop and above nothing.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger

from .core import ARM_STATE
from .core import HAND_STATE
from .core import Rejected
from .core import named_positions
from .core import qualify_proposal
from .core import vector

# Both hands sit in mode 2 at open 0.6 for every recorded episode; the left is
# the right mirrored in the two spread joints. Measured over the 2026-09-10
# session, medians. See STATUS.md, confirmed fact 6.
RIGHT_HAND_START = (1.569, -0.612, 0.831, -0.612, 0.831, 1.552, -0.612, 0.831)
LEFT_HAND_START = (-1.572, -0.612, 0.831, -0.611, 0.831, -1.551, -0.611, 0.831)
HAND_TOLERANCE = 0.08

TWIST_TOPICS = ("/kh/upper_body/action/left/twist", "/kh/upper_body/action/right/twist")
JOINT_COMMAND_TOPIC = "/kh/upper_body/action/joint"


class RawFeedback:
    """Receive-only MQTT observer, independent of the executor's own."""

    def __init__(self, host):
        self.received = None
        self.client = None
        if not host:
            return
        import paho.mqtt.client as mqtt

        self.client = mqtt.Client(client_id=f"hv1-guardian-observe-{time.time_ns()}")
        self.client.on_connect = lambda c, *_: c.subscribe("/humanoid/upper/joint_state", qos=0)
        self.client.on_disconnect = self._disconnect
        self.client.on_message = self._message
        self.client.connect_async(host, 1883, 10)
        self.client.loop_start()

    def _disconnect(self, *_args):
        self.received = None

    def _message(self, _client, _data, msg):
        if not msg.retain and msg.topic == "/humanoid/upper/joint_state":
            self.received = time.monotonic()

    def age(self, now):
        return float("inf") if self.received is None else now - self.received

    def close(self):
        if self.client is not None:
            self.client.loop_stop()
            self.client.disconnect()


class Guardian(Node):
    def __init__(self, args):
        super().__init__("hv1_vla_guardian")
        with open(args.profile, encoding="utf-8") as handle:
            self.profile = json.load(handle)
        for key in ("review_id", "start_q", "start_tolerance", "joint_min", "joint_max", "max_step"):
            if self.profile.get(key) in (None, ""):
                raise Rejected(f"field profile is missing {key}")
        self.start_q = vector(self.profile["start_q"], 7)
        self.start_tolerance = vector(self.profile["start_tolerance"], 7)
        self.joint_min = vector(self.profile["joint_min"], 7)
        self.joint_max = vector(self.profile["joint_max"], 7)
        self.max_step = vector(self.profile["max_step"], 7)

        # False until an operator states the evidence on the command line. The
        # workspace is a room, not a topic, and there is no independent hardware
        # watchdog on this rig — the supervisor's E-stop stands in for one.
        self.workspace_clear = args.workspace_clear
        self.hardware_watchdog = args.hardware_watchdog
        self.release_allowed = args.release_allowed

        self.arm = self.hand_r = self.hand_l = None
        self.approved = []
        self.sequence = 0
        self.raw = RawFeedback(args.mqtt_host)
        qos = qos_profile_sensor_data
        self.create_subscription(
            JointState, "/kh/upper_body/observation/state/joint_states", self._arm, qos)
        self.create_subscription(
            JointState, "/kdex_3f/right/rel_angle/joint_state", self._right, qos)
        self.create_subscription(
            JointState, "/kdex_3f/left/rel_angle/joint_state", self._left, qos)
        self.create_subscription(String, "/hv1_vla/proposal", self._proposal, 10)
        self.stop_client = self.create_client(Trigger, self.profile["stop_service"])
        self.publisher = self.create_publisher(String, "/hv1_vla/guardian", 1)
        self.create_timer(1.0 / args.rate, self._tick)
        self.get_logger().info(
            f"guardian up: workspace_clear={self.workspace_clear} "
            f"hardware_watchdog={self.hardware_watchdog} release_allowed={self.release_allowed}. "
            "Ctrl-C stops the executor within one lease."
        )

    # -- observations ----------------------------------------------------

    def _arm(self, msg):
        try:
            self.arm = (named_positions(msg.name, msg.position, ARM_STATE), time.monotonic())
        except Rejected:
            self.arm = None

    def _right(self, msg):
        try:
            self.hand_r = (named_positions(msg.name, msg.position, HAND_STATE), time.monotonic())
        except Rejected:
            self.hand_r = None

    def _left(self, msg):
        try:
            self.hand_l = (named_positions(msg.name, msg.position, HAND_STATE), time.monotonic())
        except Rejected:
            self.hand_l = None

    # -- proposal qualification ------------------------------------------

    def _proposal(self, msg):
        try:
            payload = json.loads(msg.data)
        except ValueError:
            self.get_logger().warn("unparseable proposal; refused")
            return
        proposal_id = qualify_proposal(payload, self.joint_min, self.joint_max, self.max_step)
        if proposal_id is None:
            self.get_logger().warn("proposal refused", throttle_duration_sec=1.0)
            return
        self.approved.append(proposal_id)
        del self.approved[:-32]

    # -- flags ------------------------------------------------------------

    def _fresh(self, entry, now, limit=0.1):
        return entry is not None and 0 <= now - entry[1] <= limit

    def _exclusive(self):
        """Nobody but the executor may drive the arm while it runs."""
        for topic in TWIST_TOPICS:
            if self.get_publishers_info_by_topic(topic):
                return False
        return len(self.get_publishers_info_by_topic(JOINT_COMMAND_TOPIC)) <= 1

    def _tick(self):
        now = time.monotonic()
        arm_ok = self._fresh(self.arm, now) and bool(
            np.all(np.abs(self.arm[0] - self.start_q) <= self.start_tolerance))
        hands_ok = (
            self._fresh(self.hand_r, now)
            and self._fresh(self.hand_l, now)
            and bool(np.all(np.abs(self.hand_r[0] - np.array(RIGHT_HAND_START)) <= HAND_TOLERANCE))
            and bool(np.all(np.abs(self.hand_l[0] - np.array(LEFT_HAND_START)) <= HAND_TOLERANCE))
        )
        self.sequence += 1
        status = {
            "sequence": self.sequence,
            "monotonic": now,
            "review_id": self.profile["review_id"],
            "grasp_mode": 2,
            "exclusive": self._exclusive(),
            "workspace_clear": self.workspace_clear,
            "stop_ready": self.stop_client.service_is_ready(),
            "hardware_watchdog_ready": self.hardware_watchdog,
            "start_pose_verified": arm_ok,
            "narrow_open_verified": hands_ok,
            "release_allowed": self.release_allowed,
            "mqtt_age_s": self.raw.age(now),
            "approved_proposals": list(self.approved),
        }
        message = String()
        message.data = json.dumps(status)
        self.publisher.publish(message)

    def destroy_node(self):
        self.raw.close()
        return super().destroy_node()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, help="approved field profile JSON")
    parser.add_argument("--mqtt-host", required=True, help="broker carrying raw joint feedback")
    parser.add_argument("--rate", type=float, default=20.0)
    parser.add_argument(
        "--workspace-clear", action="store_true",
        help="operator states the workspace is clear and stays clear; Ctrl-C withdraws it")
    parser.add_argument(
        "--hardware-watchdog", action="store_true",
        help="operator states an independent stop exists, e.g. holding the E-stop throughout")
    parser.add_argument(
        "--release-allowed", action="store_true",
        help="operator states the tray release condition is met")
    args = parser.parse_args()
    if args.rate < 15:
        parser.error("lease is 100 ms; publish at 15 Hz or faster")
    rclpy.init()
    node = Guardian(args)
    try:
        # Stepping rather than rclpy.spin: a signal invalidates the context, and
        # spin surfaces that as a wait-set error. An operator withdrawing
        # qualification must see a clean exit, not a traceback that reads like
        # the stop failed.
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:
        # The signal handler can invalidate the context between ok() and the
        # step, and rclpy reports that as a bare RCLError with no importable
        # type. Anything raised while the context is still valid is a real
        # failure and must surface.
        if rclpy.ok():
            raise
    node.get_logger().warn("guardian withdrawn; executor stops within one lease")
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
