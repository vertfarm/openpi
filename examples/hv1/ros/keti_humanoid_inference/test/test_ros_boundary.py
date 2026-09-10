"""ROS integration against fake hardware ONLY in isolated localhost domain 213."""

import argparse
import json
import os
import socket
import threading
import time

import numpy as np
import pytest

# Optional ROS imports must follow importorskip; ordinary CPU CI has no rclpy.
# ruff: noqa: E402

if os.environ.get("ROS_DOMAIN_ID") != "213" or os.environ.get("ROS_LOCALHOST_ONLY") != "1":
    pytest.skip("requires isolated test DDS domain 213 on localhost", allow_module_level=True)
rclpy = pytest.importorskip("rclpy")
from kdex_3f_ros2_msgs.action import Grasp
from kdex_3f_ros2_msgs.srv import SetOpen
from keti_humanoid_inference import node as implementation
from keti_humanoid_inference.core import ARM_COMMAND
from keti_humanoid_inference.core import ARM_STATE
from keti_humanoid_inference.core import CONTRACT
from keti_humanoid_inference.core import CONTRACT_SHA
from keti_humanoid_inference.core import CORE_SOURCE_SHA256
from keti_humanoid_inference.core import HAND_STATE
from rclpy.action import ActionServer
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory


class FakeClient:
    def __init__(self, *args, **kwargs):
        self.calls = 0

    def metadata(self):
        return dict(
            ready=True,
            contract=CONTRACT,
            contract_sha256=CONTRACT_SHA,
            action_horizon=15,
            denoise=10,
            snapshot_sha256="SYNTHETIC",
            adapter_core_sha256=CORE_SOURCE_SHA256,
            hand_state_envelope={
                "names": list(HAND_STATE),
                "q01": [-1.0] * 8,
                "q99": [1.0] * 8,
                "margin_rad": 0.05,
            },
        )

    def infer(self, request, sha):
        self.calls += 1
        action = np.zeros((15, 8))
        action[:, 7] = 1 if self.calls <= 2 else 0
        return action, {"server_ms": 1, "preprocess_ms": 0}


class FakeRaw:
    def __init__(self, host):
        self.positions = np.zeros(16)

    def age(self, now):
        return 0.001

    def close(self):
        pass


class Rig(Node):
    def __init__(self):
        super().__init__("hv1_vla_guardian")
        self.safe = True
        self.sequence = 0
        self.approvals = []
        self.arm_targets, self.grasps, self.opens, self.stops = [], [], [], []
        self.arm = self.create_publisher(
            JointState,
            "/kh/upper_body/observation/state/joint_states",
            qos_profile_sensor_data,
        )
        self.hand = self.create_publisher(JointState, "/kdex_3f/right/rel_angle/joint_state", qos_profile_sensor_data)
        self.cameras = {
            k: self.create_publisher(CompressedImage, t, qos_profile_sensor_data)
            for k, t in implementation.IMAGE_TOPICS.items()
        }
        self.guard = self.create_publisher(String, "/hv1_vla/guardian", 1)
        self.create_subscription(String, "/hv1_vla/proposal", self.proposal, 1)
        self.create_subscription(
            JointTrajectory,
            "/kh/upper_body/action/joint",
            self.arm_targets.append,
            1,
        )
        self.action = ActionServer(self, Grasp, "/kdex_3f/right/grasp", self.grasp)
        self.create_service(SetOpen, "/kdex_3f/right/set_open", self.open)
        self.create_service(Trigger, "/fake/verified_stop", self.stop)
        self.create_timer(0.02, self.publish)

    def publish(self):
        for pub, names in ((self.arm, ARM_STATE), (self.hand, HAND_STATE)):
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name, msg.position = list(names), [0.0] * len(names)
            pub.publish(msg)
        for pub in self.cameras.values():
            msg = CompressedImage()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.format, msg.data = "jpeg", b"FAKE_IMAGE_NO_CAMERA"
            pub.publish(msg)
        self.sequence += 1
        msg = String()
        msg.data = json.dumps(
            dict(
                sequence=self.sequence,
                monotonic=time.monotonic(),
                exclusive=self.safe,
                workspace_clear=True,
                stop_ready=True,
                hardware_watchdog_ready=True,
                review_id="SYNTHETIC-NEVER-FIELD",
                grasp_mode=2,
                mqtt_age_s=0.001,
                start_pose_verified=True,
                narrow_open_verified=True,
                release_allowed=True,
                approved_proposals=self.approvals[-4:],
            )
        )
        self.guard.publish(msg)

    def proposal(self, msg):
        # Synthetic zero trajectories only. Never use this as a field guardian.
        value = json.loads(msg.data)
        assert all(np.allclose(t["action"][:7], 0) for t in value["proposal"]["targets"])
        self.approvals.append(value["proposal_id"])

    def grasp(self, goal):
        self.grasps.append(goal.request)
        goal.succeed()
        result = Grasp.Result()
        result.success = True
        return result

    def open(self, request, response):
        self.opens.append(request)
        response.success = True
        return response

    def stop(self, request, response):
        self.stops.append(request)
        response.success = True
        return response


def wait_until(predicate, timeout=5):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("ROS test timed out")


@pytest.mark.parametrize("mode", ["shadow", "live"])
def test_ros_boundary(tmp_path, monkeypatch, mode):
    monkeypatch.setattr(implementation, "PolicyHTTP", FakeClient)
    monkeypatch.setattr(implementation, "RawFeedback", FakeRaw)
    profile = dict(
        approved=True,
        review_id="SYNTHETIC-NEVER-FIELD",
        contract_sha256=CONTRACT_SHA,
        snapshot_sha256="SYNTHETIC",
        qualification_evidence="fake hardware only",
        guardian_node="/hv1_vla_guardian",
        stop_service="/fake/verified_stop",
        joint_min=[-2.0] * 7,
        joint_max=[2.0] * 7,
        max_step=[0.1] * 7,
        max_tracking_error=[0.2] * 7,
        max_velocity=[2.0] * 7,
        max_acceleration=[100.0] * 7,
        start_q=[0.0] * 7,
        start_tolerance=[0.1] * 7,
    )
    profile_path = tmp_path / "synthetic.json"
    profile_path.write_text(json.dumps(profile))
    args = argparse.Namespace(
        mode=mode,
        port=8000,
        http_timeout=1,
        profile=str(profile_path),
        output=str(tmp_path / "run"),
        mqtt_host="FAKE-NOT-CONNECTED",
        seconds=0,
        grip_close_threshold=0.7,
        grip_open_threshold=0.3,
        grip_min_hold=0.0,
    )
    rclpy.init()
    rig, node = Rig(), implementation.DeployNode(args)
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(rig)
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    try:
        wait_until(lambda: len(node.obs.entries) == 5)
        if mode == "shadow":
            wait_until(lambda: node.inferences >= 4)
            assert not hasattr(node, "arm_pub")
            assert not hasattr(node, "close_client")
            assert not hasattr(node, "open_client")
            assert node.commands_sent == 0 and not rig.arm_targets
            assert not rig.grasps and not rig.opens
        else:
            wait_until(
                lambda: node.guard_received > 0
                and node.close_client.server_is_ready()
                and node.open_client.service_is_ready()
                and node.stop_client.service_is_ready()
            )
            # An actual local operator command; no DDS ARM service.
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.settimeout(3)
                client.connect(str(tmp_path / "run/operator.sock"))
                client.sendall(b"ARM SYNTHETIC-NEVER-FIELD\n")
                response = json.loads(client.recv(4096))
            assert response.get("state") == "RUNNING", response
            wait_until(lambda: len(rig.arm_targets) >= 9 and len(rig.opens) == 1)
            assert len(rig.grasps) == 1
            assert rig.grasps[0].wrap and rig.grasps[0].mode == 2
            assert rig.grasps[0].tip_distance == 0 and rig.opens[0].open == 0.6
            assert all(tuple(m.joint_names) == ARM_COMMAND and len(m.points) == 1 for m in rig.arm_targets)
            rig.safe = False
            wait_until(lambda: node.state == "FAULT" and len(rig.stops) == 1)
            count = len(rig.arm_targets)
            time.sleep(0.12)
            assert len(rig.arm_targets) == count
    finally:
        node.close()
        executor.shutdown(timeout_sec=2)
        thread.join(timeout=2)
        node.destroy_node()
        rig.destroy_node()
        rclpy.shutdown()
