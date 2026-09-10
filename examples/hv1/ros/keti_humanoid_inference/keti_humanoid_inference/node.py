"""ROS observations -> OpenPI -> shadow, or explicitly qualified live targets.

Shadow constructs NO robot command publisher, action client, or service client.
Live requires an independent field guardian and verified physical stop service;
this node is not a safety PLC and cannot qualify its own hardware interlock.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import queue
import socketserver
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from .core import ARM_COMMAND
from .core import ARM_STATE
from .core import CAMERAS
from .core import HAND_STATE
from .core import ChunkQueue
from .core import GripEdges
from .core import JsonLog
from .core import LiveGate
from .core import Observations
from .core import PolicyHTTP
from .core import Rejected
from .core import make_request
from .core import named_positions

IMAGE_TOPICS = {
    "head": "/kh/upper_body/head/color/image_raw/compressed",
    "hand_l": "/kdex_3f/left/camera/image_raw/compressed",
    "hand_r": "/kdex_3f/right/camera/image_raw/compressed",
}


class RawFeedback:
    """Receive-only MQTT observer; never uses the republished ROS stamp as proof."""

    def __init__(self, host):
        self.received = None
        self.positions = None
        self.connected = False
        self.client = None
        if not host:
            return
        import paho.mqtt.client as mqtt

        self.client = mqtt.Client(client_id=f"hv1-vla-observe-{time.time_ns()}")
        self.client.on_connect = self.connect
        self.client.on_disconnect = self.disconnect
        self.client.on_message = self.message
        self.client.connect_async(host, 1883, 10)
        self.client.loop_start()

    def connect(self, client, _data, _flags, code):
        self.connected = code == 0
        if self.connected:
            client.subscribe("/humanoid/upper/joint_state", qos=0)

    def disconnect(self, *_args):
        self.connected = False
        self.received = None

    def message(self, _client, _data, msg):
        if msg.retain or msg.topic != "/humanoid/upper/joint_state":
            return
        try:
            text = msg.payload.decode()
            values = np.asarray([float(x) for x in text.split("[", 1)[1].split("]", 1)[0].split(",") if x.strip()])
            if values.shape != (16,) or not np.isfinite(values).all():
                return
            self.positions = values
            self.received = time.monotonic()
        except (ValueError, IndexError, UnicodeError):
            pass

    def age(self, now):
        return None if not self.connected or self.received is None else now - self.received

    def close(self):
        if self.client:
            self.client.disconnect()
            self.client.loop_stop()


class OperatorServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True


class DeployNode(Node):
    def __init__(self, args):
        super().__init__("hv1_vla_inference")
        self.args = args
        self.live = args.mode == "live"
        self.client = PolicyHTTP(args.port, timeout=args.http_timeout)
        self.metadata = self.client.metadata()
        self.profile = json.loads(Path(args.profile).read_text()) if args.profile else None
        self.gate = LiveGate(self.profile) if self.live else None
        if self.live and self.profile["snapshot_sha256"] != self.metadata["snapshot_sha256"]:
            raise Rejected("field profile selects a different checkpoint")
        self.output = Path(args.output).resolve()
        if any(part.lower() in ("raw", "datasets") for part in self.output.parts):
            raise Rejected("deployment logs must stay outside immutable raw/dataset roots")
        self.output.mkdir(parents=True, exist_ok=False)
        self.log = JsonLog(self.output / "events.jsonl")
        self.log.write("startup", mode=args.mode, metadata=self.metadata, profile=self.profile)
        self.obs, self.chunks, self.grip = Observations(), ChunkQueue(), GripEdges()
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.future = None
        self.sequence = 0
        self.commands_sent = self.inferences = 0
        self.guard_status, self.guard_received, self.guard_sequence = {}, -1.0, -1
        self.operator_commands = queue.Queue()
        self.state = "DISARMED" if self.live else "SHADOW"
        self.fault = None
        self.started = time.monotonic()
        self.run_started = None
        self.status_last = 0.0
        self.last_output = None
        self.hand_waits = []
        self.grasp_handle = None
        self.operator_server = None
        self.raw = RawFeedback(args.mqtt_host)
        qos = qos_profile_sensor_data
        self.create_subscription(
            JointState,
            "/kh/upper_body/observation/state/joint_states",
            self.arm_state,
            qos,
        )
        self.create_subscription(JointState, "/kdex_3f/right/rel_angle/joint_state", self.hand_state, qos)
        for name, topic in IMAGE_TOPICS.items():
            self.create_subscription(CompressedImage, topic, lambda msg, key=name: self.camera(key, msg), qos)
        self.status_pub = self.create_publisher(String, "/hv1_vla/status", 1)
        # This topic is not connected to a robot control API.
        self.candidate_pub = self.create_publisher(String, "/hv1_vla/shadow/candidate", 1)
        self.proposal_pub = self.create_publisher(String, "/hv1_vla/proposal", 1)
        if self.live:
            self.setup_live()
        self.create_timer(0.01, self.tick)

    def source_age(self, msg):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if stamp <= 0:
            raise Rejected("missing source stamp")
        return self.get_clock().now().nanoseconds * 1e-9 - stamp

    def ingest(self, key, msg, value):
        try:
            self.obs.put(key, value, time.monotonic(), self.source_age(msg))
        except Rejected as error:
            self.obs.entries.pop(key, None)
            self.log.write("observation_rejected", key=key, reason=str(error))

    def arm_state(self, msg):
        try:
            self.ingest("arm", msg, named_positions(msg.name, msg.position, ARM_STATE))
        except Rejected as error:
            self.obs.entries.pop("arm", None)
            self.log.write("observation_rejected", key="arm", reason=str(error))

    def hand_state(self, msg):
        try:
            self.ingest("hand", msg, named_positions(msg.name, msg.position, HAND_STATE))
        except Rejected as error:
            self.obs.entries.pop("hand", None)
            self.log.write("observation_rejected", key="hand", reason=str(error))

    def camera(self, name, msg):
        self.ingest(name, msg, bytes(msg.data))

    def setup_live(self):
        from kdex_3f_ros2_msgs.action import Grasp
        from kdex_3f_ros2_msgs.srv import SetOpen
        from rclpy.action import ActionClient
        from std_srvs.srv import Trigger
        from trajectory_msgs.msg import JointTrajectory

        self.arm_pub = self.create_publisher(JointTrajectory, "/kh/upper_body/action/joint", 1)
        self.close_client = ActionClient(self, Grasp, "/kdex_3f/right/grasp")
        self.open_client = self.create_client(SetOpen, "/kdex_3f/right/set_open")
        self.stop_client = self.create_client(Trigger, self.profile["stop_service"])
        self.create_subscription(String, "/hv1_vla/guardian", self.guardian, 1)
        path = self.output / "operator.sock"
        if len(str(path).encode()) >= 100:
            raise Rejected("operator socket path too long")
        node = self

        class Handler(socketserver.StreamRequestHandler):
            def handle(self):
                self.request.settimeout(2)
                text = self.rfile.readline(512).decode().strip()
                result, done = {}, threading.Event()
                node.operator_commands.put((text, result, done))
                if done.wait(2):
                    self.wfile.write((json.dumps(result) + "\n").encode())

        self.operator_server = OperatorServer(str(path), Handler)
        path.chmod(0o600)
        threading.Thread(target=self.operator_server.serve_forever, daemon=True).start()

    def guardian(self, msg):
        try:
            status = json.loads(msg.data)
            seq = status["sequence"]
            stamp = float(status["monotonic"])
            now = time.monotonic()
            if type(seq) is not int or seq <= self.guard_sequence or not 0 <= now - stamp <= 0.1:
                raise Rejected("guardian replay or wrong clock")
            self.guard_status, self.guard_received, self.guard_sequence = status, stamp, seq
        except (ValueError, KeyError, TypeError):
            self.guard_received = -1.0

    def live_health(self, now, state):
        self.gate.guardian(self.guard_status, self.guard_received, now)
        publishers = self.get_publishers_info_by_topic("/hv1_vla/guardian")
        names = [p.node_namespace.rstrip("/") + "/" + p.node_name for p in publishers]
        if names != [self.profile["guardian_node"]]:
            raise Rejected("missing/duplicate/unexpected guardian")
        # A graph check complements, but does not replace, exclusive controller
        # ownership verified by the external guardian (including ROS actions).
        own = self.get_namespace().rstrip("/") + "/" + self.get_name()

        def publisher_name(info):
            return info.node_namespace.rstrip("/") + "/" + info.node_name

        for topic in (
            "/kh/upper_body/action/left/twist",
            "/kh/upper_body/action/right/twist",
        ):
            foreign = [publisher_name(info) for info in self.get_publishers_info_by_topic(topic) if publisher_name(info) != own]
            if foreign:
                raise Rejected(f"foreign twist publisher present: {foreign}")
        joint_publishers = [
            publisher_name(info)
            for info in self.get_publishers_info_by_topic("/kh/upper_body/action/joint")
        ]
        if joint_publishers != [own]:
            raise Rejected("joint command publisher conflict")
        age = self.raw.age(now)
        if age is None or not 0 <= age <= 0.1:
            raise Rejected("raw MQTT feedback missing/stale")
        if np.any(np.abs(self.raw.positions[9:16] - state[:7]) > self.profile["max_tracking_error"]):
            raise Rejected("ROS/MQTT state mismatch")
        if not (
            self.stop_client.service_is_ready()
            and self.open_client.service_is_ready()
            and self.close_client.server_is_ready()
        ):
            raise Rejected("stop/gripper endpoint unavailable")

    def operator_tick(self, now):
        while not self.operator_commands.empty():
            text, result, done = self.operator_commands.get_nowait()
            try:
                if text == "STOP":
                    self.halt("operator stop")
                elif text == "ARM " + self.profile["review_id"]:
                    if self.state != "DISARMED":
                        raise Rejected("restart node after STOP/FAULT; no implicit resume")
                    if self.future is not None:
                        raise Rejected("wait for shadow inference to finish before ARM")
                    self.obs.snapshot(now)
                    state = np.r_[self.obs.entries["arm"][0], self.obs.entries["hand"][0]]
                    self.live_health(now, state)
                    self.gate.arm(state[:7], self.guard_status, self.guard_received, now)
                    self.chunks.clear()
                    self.grip = GripEdges()
                    self.state, self.run_started = "RUNNING", now
                    self.last_output = now
                    self.log.write("local_arm", review_id=self.profile["review_id"])
                else:
                    raise Rejected("expected ARM <review_id> or STOP")
                result["state"] = self.state
            except Rejected as error:
                result["error"] = str(error)
            finally:
                done.set()

    def halt(self, reason):
        if self.fault:
            return
        self.fault, self.state = reason, "FAULT"
        self.chunks.clear()
        self.log.write("fault", reason=reason, commands_sent=self.commands_sent)
        if self.live:
            self.gate.disarm()
            if self.grasp_handle is not None:
                self.grasp_handle.cancel_goal_async()
            from std_srvs.srv import Trigger

            if self.stop_client.service_is_ready():
                fut = self.stop_client.call_async(Trigger.Request())
                fut.add_done_callback(
                    lambda f: self.log.write("stop_response", success=bool(f.result() and f.result().success))
                )
            else:
                self.log.write("stop_unavailable", physical_stop_confirmed=False)
        self.get_logger().error(reason)

    def submit(self, state, images, anchor, now, ages):
        self.sequence += 1
        request = make_request(self.sequence, state, images)
        # Save the exact received compressed images and numerical model inputs.
        np.savez(
            self.output / f"observation_{self.sequence:06d}.npz",
            state=state.astype(np.float32),
            **{c: np.frombuffer(images[c], np.uint8) for c in CAMERAS},
        )
        self.log.write(
            "request",
            sequence=self.sequence,
            anchor=anchor,
            state=state.tolist(),
            raw_mqtt_age_s=self.raw.age(now),
            source_ages_s=ages,
        )
        future = self.pool.submit(self.client.infer, request, self.metadata["snapshot_sha256"])
        self.future = (future, anchor, now, self.sequence)

    def hand_event(self, event, now):
        if event == "close":
            from kdex_3f_ros2_msgs.action import Grasp

            goal = Grasp.Goal()
            goal.mode, goal.wrap = 2, True
            goal.tip_distance, goal.speed, goal.squeeze_current = 0.0, 0.0, 0.0
            future = self.close_client.send_goal_async(goal)
            self.hand_waits.append(("accept", future, now + 1.0))
        elif event == "open":
            if any(kind in ("accept", "grasp") for kind, _, _ in self.hand_waits):
                raise Rejected("release requested before grasp operation completed")
            if self.guard_status.get("release_allowed") is not True:
                raise Rejected("release outside guardian-qualified tray condition")
            from kdex_3f_ros2_msgs.srv import SetOpen

            request = SetOpen.Request()
            request.open, request.speed = 0.6, 0.0
            self.hand_waits.append(("open", self.open_client.call_async(request), now + 1.0))
        if event:
            self.log.write("gripper_request", event_name=event, speed="existing_server_default")

    def check_hand(self, now):
        remaining = []
        for kind, future, deadline in self.hand_waits:
            if now > deadline:
                raise Rejected("gripper " + kind + " timeout")
            if not future.done():
                remaining.append((kind, future, deadline))
                continue
            result = future.result()
            if kind == "accept":
                if result is None or not result.accepted:
                    raise Rejected("grasp goal rejected")
                self.grasp_handle = result
                remaining.append(("grasp", result.get_result_async(), now + 5.0))
            else:
                response = result.result if kind == "grasp" else result
                if response is None or not response.success:
                    raise Rejected("gripper " + kind + " failed")
                if kind == "grasp":
                    self.grasp_handle = None
                self.log.write("gripper_result", kind=kind, success=True)
        self.hand_waits = remaining

    def emit(self, due, action, state, now):
        event = self.grip.update(float(action[7]))
        if self.live:
            self.live_health(now, state)
            self.gate.check(
                action[:7],
                state[:7],
                self.guard_status,
                self.guard_received,
                now,
                proposal_id=self.chunks.last_meta["proposal_id"],
            )
            self.hand_event(event, now)
            from trajectory_msgs.msg import JointTrajectory
            from trajectory_msgs.msg import JointTrajectoryPoint

            msg = JointTrajectory()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.joint_names = list(ARM_COMMAND)
            point = JointTrajectoryPoint()
            point.positions = action[:7].tolist()
            msg.points = [point]
            self.arm_pub.publish(msg)
            self.commands_sent += 1
        else:
            msg = String()
            msg.data = json.dumps({"action": action.tolist(), "gripper_event": event, "sent": False})
            self.candidate_pub.publish(msg)
        self.last_output = now
        self.log.write(
            "target",
            due=due,
            action=action.tolist(),
            measured=state[:7].tolist(),
            gripper_event=event,
            sent=self.live,
            lateness_ms=(now - due) * 1000,
            prediction=self.chunks.last_meta,
        )

    def tick(self):
        now = time.monotonic()
        try:
            if self.live:
                self.operator_tick(now)
            if self.fault:
                return
            state, images, anchor, ages = self.obs.snapshot(now)
            # Model input may use a buffered state to align cameras. Physical
            # tracking checks must always use the newest measured state.
            current_state = np.r_[self.obs.entries["arm"][0], self.obs.entries["hand"][0]]
            if self.live and self.state == "RUNNING":
                self.live_health(now, current_state)
                self.check_hand(now)
                if now - self.run_started >= 45:
                    raise Rejected("pilot duration reached; operator must label outcome")
            if self.future and self.future[0].done():
                future, req_anchor, requested, seq = self.future
                self.future = None
                actions, response = future.result()
                elapsed = (now - requested) * 1000
                self.inferences += 1
                self.log.write(
                    "response",
                    sequence=seq,
                    actions=actions.tolist(),
                    client_ms=elapsed,
                    **{k: response[k] for k in ("server_ms", "preprocess_ms")},
                )
                if self.live and self.state == "DISARMED":
                    pass
                else:
                    scheduled = self.chunks.offer(
                        actions, req_anchor, now, sequence=seq, lead_s=self.chunks.period if self.live else 0.002
                    )
                    self.log.write("schedule", sequence=seq, **scheduled)
                    proposal = String()
                    proposal.data = json.dumps({**scheduled, "snapshot_sha256": self.metadata["snapshot_sha256"]})
                    self.proposal_pub.publish(proposal)
            if self.live and self.state == "DISARMED":
                return  # ARM starts a fresh request from current observations.
            target = self.chunks.pop(now)
            if target:
                self.emit(*target, current_state, now)
            if self.live and self.last_output is not None and now - self.last_output > 0.2:
                raise Rejected("execution buffer underflow")
            if self.future is None and len(self.chunks.queue) <= 3:
                self.submit(state, images, anchor, now, ages)
            if self.future and now - self.future[2] > 0.2:
                raise Rejected("inference deadline exceeded")
            self.state = "RUNNING" if self.live else "SHADOW"
        except Exception as error:
            if self.live and self.state == "RUNNING":
                self.halt(str(error))
            else:
                # No robot output exists here. Recover as sensors become ready.
                self.state = "DISARMED" if self.live else "WAITING"
                self.chunks.clear()
                self.grip = GripEdges()
                if self.future and self.future[0].done():
                    self.future = None
                self.log.write("waiting", reason=str(error))
        finally:
            if now - self.status_last >= 1:
                report = {
                    "state": self.state,
                    "fault": self.fault,
                    "inferences": self.inferences,
                    "robot_commands_sent": self.commands_sent,
                    "raw_mqtt_age_s": self.raw.age(now),
                    "snapshot_sha256": self.metadata["snapshot_sha256"],
                }
                msg = String()
                msg.data = json.dumps(report)
                self.status_pub.publish(msg)
                self.log.write("status", **report)
                self.status_last = now

    def close(self):
        if self.live and self.state == "RUNNING":
            self.halt("client shutdown")
        self.raw.close()
        self.pool.shutdown(wait=True, cancel_futures=True)
        if self.operator_server:
            self.operator_server.shutdown()
            self.operator_server.server_close()
        self.log.write("shutdown", robot_commands_sent=self.commands_sent, inferences=self.inferences)
        self.log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("shadow", "live"), default="shadow")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--http-timeout", type=float, default=1.0)
    parser.add_argument("--mqtt-host", default="")
    parser.add_argument("--profile")
    parser.add_argument("--output", required=True, help="new directory outside raw datasets")
    parser.add_argument("--seconds", type=float, default=60)
    args, ros_args = parser.parse_known_args()
    if args.mode == "live" and (not args.profile or not args.mqtt_host):
        parser.error("live requires field profile and raw MQTT observer")
    rclpy.init(args=ros_args)
    node = None
    try:
        node = DeployNode(args)
        end = time.monotonic() + args.seconds if args.seconds > 0 else float("inf")
        while rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.01)
    except KeyboardInterrupt:
        pass
    finally:
        if node:
            node.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
