"""Verified BF16 snapshot -> loopback HTTP. This process has no robot APIs."""

from __future__ import annotations

import argparse
import base64
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading
import time

import numpy as np

from .ros.keti_humanoid_inference.keti_humanoid_inference.core import CAMERAS
from .ros.keti_humanoid_inference.keti_humanoid_inference.core import CONTRACT
from .ros.keti_humanoid_inference.keti_humanoid_inference.core import CONTRACT_SHA
from .ros.keti_humanoid_inference.keti_humanoid_inference.core import CORE_SOURCE_SHA256
from .ros.keti_humanoid_inference.keti_humanoid_inference.core import HAND_ENVELOPE_MARGIN_RAD
from .ros.keti_humanoid_inference.keti_humanoid_inference.core import HAND_STATE
from .ros.keti_humanoid_inference.keti_humanoid_inference.core import PROMPT
from .ros.keti_humanoid_inference.keti_humanoid_inference.core import Rejected
from .ros.keti_humanoid_inference.keti_humanoid_inference.core import validate_hand_envelope
from .ros.keti_humanoid_inference.keti_humanoid_inference.core import vector

# Must equal `pipeline.SCHEMA`; duplicated so the registry gate stays cheap.
REGISTRY_SCHEMA = "hv1_two_track_v1"


def prepare_images(payload):
    """Match recorder geometry and official OpenPI PIL preprocessing.

    HEVC compression in the training recordings is not reproduced on live input.
    Neither that lossy difference nor row alignment proves exposure synchronization.
    """
    import cv2
    from openpi_client.image_tools import resize_with_pad

    if set(payload["images"]) != set(CAMERAS):
        raise Rejected("three camera identities required")
    images = {}
    for name in CAMERAS:
        raw = base64.b64decode(payload["images"][name], validate=True)
        if payload.get("image_encoding") == "rgb224":
            if len(raw) != 224 * 224 * 3:
                raise Rejected("wrong RGB frame size")
            image = np.frombuffer(raw, np.uint8).reshape(224, 224, 3).copy()
        elif payload.get("image_encoding") == "ros_compressed":
            if not 0 < len(raw) <= 2_000_000:
                raise Rejected("compressed frame size")
            bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if bgr is None or bgr.shape[2] != 3:
                raise Rejected("cannot decode camera frame")
            if bgr.shape[:2] != (480, 640):
                bgr = cv2.resize(bgr, (640, 480), interpolation=cv2.INTER_AREA)
            image = resize_with_pad(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), 224, 224)
        else:
            raise Rejected("unsupported image encoding")
        images[name] = image
    return images


def load_snapshot(campaign, snapshot, denoise=10, *, registry=None):
    from filelock import FileLock

    from .artifacts import file_hash
    from .artifacts import read_json

    campaign, snapshot = Path(campaign).resolve(), Path(snapshot).resolve()
    # A registry is required. The alternative used to be a hardcoded allowance
    # for the finished A-F campaign's C/F-5000 snapshots, which by 2026-09-11
    # only served to reject every checkpoint anyone actually wanted to deploy.
    if registry is None:
        raise Rejected("a hash-bound SHADOW_ONLY registry is required")
    # Checked from the literal, not from `pipeline.SCHEMA`, so a bad registry is
    # refused without importing the training stack. `test_deploy` binds the two
    # together so the literal cannot drift.
    registry_schema = read_json(registry).get("schema")
    if registry_schema != REGISTRY_SCHEMA:
        raise Rejected(f"unsupported registry schema: {registry_schema!r}")
    from .pipeline_eval import registered_config

    config, record = registered_config(campaign, snapshot, registry)
    lock = FileLock(str(campaign.parent / "hv1-ml-gpu.lock"), timeout=0)
    lock.acquire()
    try:
        data_config = config.data.create(config.assets_dirs, config.model)
        state_stats = None if data_config.norm_stats is None else data_config.norm_stats.get("state")
        if state_stats is None or state_stats.q01 is None or state_stats.q99 is None:
            raise Rejected("state quantile statistics missing")
        q01, q99 = np.asarray(state_stats.q01), np.asarray(state_stats.q99)
        if q01.shape != (15,) or q99.shape != (15,):
            raise Rejected("state quantile statistics have wrong shape")
        from openpi.policies.policy_config import create_trained_policy

        policy = create_trained_policy(config, snapshot, default_prompt=PROMPT, sample_kwargs={"num_steps": denoise})
        dummy = {
            "state": np.zeros(15, np.float32),
            "images": {c: np.zeros((224, 224, 3), np.uint8) for c in CAMERAS},
            "prompt": PROMPT,
        }
        actions = np.asarray(policy.infer(dummy)["actions"])
        if actions.shape != (15, 8) or not np.isfinite(actions).all():
            raise Rejected("warmup inference failed")
    except BaseException:
        lock.release()
        raise
    meta = {
        "ready": True,
        "contract": CONTRACT,
        "contract_sha256": CONTRACT_SHA,
        "adapter_core_sha256": CORE_SOURCE_SHA256,
        "snapshot": str(snapshot),
        "snapshot_sha256": file_hash(snapshot / "snapshot.json"),
        "experiment": record["recipe"]["name"],
        "step": record["step"],
        "action_horizon": 15,
        "denoise": denoise,
        "camera_dropout": False,
        "norm_stats_sha256": config.policy_metadata["norm_stats_sha256"],
        "hand_state_envelope": {
            "names": list(HAND_STATE),
            "q01": q01[7:].tolist(),
            "q99": q99[7:].tolist(),
            "margin_rad": HAND_ENVELOPE_MARGIN_RAD,
        },
        "robot_commands_sent": 0,
    }
    return policy, meta, lock


def make_handler(policy, metadata):
    inference_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(5)

        def log_message(self, fmt, *args):
            pass

        def reply(self, code, value):
            body = json.dumps(value, allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.reply(
                200 if self.path == "/health" else 404, metadata if self.path == "/health" else {"error": "not found"}
            )

        def do_POST(self):
            if self.path != "/infer":
                self.reply(404, {"error": "not found"})
                return
            acquired = False
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 9_000_000:
                    raise Rejected("invalid body size")
                payload = json.loads(self.rfile.read(length))
                if payload.get("contract_sha256") != CONTRACT_SHA or payload.get("prompt") != PROMPT:
                    raise Rejected("request contract/prompt mismatch")
                sequence = payload["sequence"]
                if type(sequence) is not int or sequence < 0:
                    raise Rejected("invalid sequence")
                acquired = inference_lock.acquire(blocking=False)
                if not acquired:
                    self.reply(409, {"error": "one inference request at a time"})
                    return
                started = time.perf_counter()
                state = vector(payload["state"], 15).astype(np.float32)
                validate_hand_envelope(state[7:], metadata)
                images = prepare_images(payload)
                prepared = time.perf_counter()
                actions = np.asarray(policy.infer({"state": state, "images": images, "prompt": PROMPT})["actions"])
                if actions.shape != (15, 8) or not np.isfinite(actions).all():
                    raise Rejected("invalid model output")
                self.reply(
                    200,
                    {
                        "sequence": sequence,
                        "contract_sha256": CONTRACT_SHA,
                        "snapshot_sha256": metadata["snapshot_sha256"],
                        "actions": actions.tolist(),
                        "preprocess_ms": (prepared - started) * 1000,
                        "server_ms": (time.perf_counter() - started) * 1000,
                    },
                )
            except Rejected as error:
                self.reply(422, {"error": str(error)})
            except (ValueError, KeyError, TypeError) as error:
                self.reply(400, {"error": str(error)})
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as error:
                self.reply(500, {"error": type(error).__name__})
            finally:
                if acquired:
                    inference_lock.release()

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--registry", required=True, help="Hash-bound SHADOW_ONLY registry; never auto-arms ROS")
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("port must be 1024..65535")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    policy, meta, lock = load_snapshot(args.campaign, args.snapshot, registry=args.registry)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(policy, meta))
    print(json.dumps(meta), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        lock.release()


if __name__ == "__main__":
    main()
