"""Walk the right arm back to the qualified starting pose, slowly.

Teleop is off during a VLA pilot, so after a run leaves the arm away from
start_q there is nothing left to drive it back. This publishes the same
JointTrajectory the executor would, ramped over a fixed duration so no single
command is a jump, and stops as soon as it is inside start_tolerance.

It is an operator tool, not part of the pilot: run it only while the executor
is stopped, or there will be two publishers on the command topic and the
guardian will withdraw `exclusive`.
"""

import argparse
import json
import sys

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

sys.path.insert(0, "/workspace/ros2/vla_ws/src/keti_humanoid_inference")
from keti_humanoid_inference.core import ARM_COMMAND
from keti_humanoid_inference.core import ARM_STATE
from keti_humanoid_inference.core import named_positions

RATE = 30.0


class Return(Node):
    def __init__(self, target, tolerance, seconds, max_gap):
        super().__init__("hv1_return_to_start")
        self.target = np.asarray(target, dtype=float)
        self.tolerance = np.asarray(tolerance, dtype=float)
        self.seconds = seconds
        self.max_gap = max_gap
        self.measured = None
        self.start = None
        self.ticks = 0
        self.create_subscription(
            JointState, "/kh/upper_body/observation/state/joint_states", self._state, qos_profile_sensor_data
        )
        self.publisher = self.create_publisher(JointTrajectory, "/kh/upper_body/action/joint", 1)
        self.create_timer(1.0 / RATE, self._tick)

    def _state(self, msg):
        try:
            self.measured = named_positions(msg.name, msg.position, ARM_STATE)
        except Exception:
            self.measured = None

    def _tick(self):
        if self.measured is None:
            return
        if self.start is None:
            gap = np.abs(self.target - self.measured)
            if gap.max() > self.max_gap:
                self.get_logger().error(
                    f"gap {gap.max():.4f} rad exceeds --max-gap {self.max_gap}; "
                    "move the arm closer by hand or raise the ceiling deliberately"
                )
                raise SystemExit(1)
            self.start = self.measured.copy()
            self.get_logger().info(
                f"start gap {np.round(gap, 4).tolist()} max {gap.max():.4f} rad, ramping over {self.seconds:.0f}s"
            )
        self.ticks += 1
        fraction = min(1.0, self.ticks / (self.seconds * RATE))
        # Ramp the command from where the arm was to where it belongs, and do
        # not re-anchor it on the measurement each tick: re-anchoring caps the
        # command at one step beyond the arm, so a joint that will not move
        # for that step never gets a larger error and never moves at all.
        # The command leads the arm by at most the initial gap, which is the
        # whole distance being travelled and is checked before starting.
        command = self.start + (self.target - self.start) * fraction
        message = JointTrajectory()
        message.header.stamp = self.get_clock().now().to_msg()
        message.joint_names = list(ARM_COMMAND)
        point = JointTrajectoryPoint()
        point.positions = [float(v) for v in command]
        message.points = [point]
        self.publisher.publish(message)
        gap = np.abs(self.measured - self.target)
        if fraction >= 1.0 and np.all(gap <= self.tolerance):
            self.get_logger().info(f"inside tolerance, max gap {gap.max():.4f} rad")
            raise SystemExit(0)
        if self.ticks > (self.seconds + 10) * RATE:
            self.get_logger().error(f"did not converge, max gap {gap.max():.4f} rad")
            raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, help="field profile holding start_q")
    parser.add_argument("--seconds", type=float, default=6.0, help="ramp duration")
    parser.add_argument("--max-gap", type=float, default=0.35, help="refuse to ramp a gap wider than this, in rad")
    args = parser.parse_args()
    with open(args.profile, encoding="utf-8") as handle:
        profile = json.load(handle)
    rclpy.init()
    node = Return(profile["start_q"], profile["start_tolerance"], args.seconds, args.max_gap)
    code = 0
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
    except SystemExit as done:
        code = done.code
    except KeyboardInterrupt:
        code = 1
    except Exception:
        if rclpy.ok():
            raise
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    raise SystemExit(code)


if __name__ == "__main__":
    main()
