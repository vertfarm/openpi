"""Recorded observations through the deployed HTTP server. Never imports ROS."""

import argparse
import base64
import json
from pathlib import Path
import time

import numpy as np

from .native import CAMERAS
from .native import PROMPT
from .native import read_numeric
from .ros.keti_humanoid_inference.keti_humanoid_inference.core import CONTRACT_SHA
from .ros.keti_humanoid_inference.keti_humanoid_inference.core import PolicyHTTP


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    import cv2
    from openpi_client.image_tools import resize_with_pad

    directory = Path(args.episode)
    state, action, info = read_numeric(directory / "data.hdf5")
    client = PolicyHTTP(args.port, timeout=10)
    metadata = client.metadata()
    frames = sorted({0, info["grasp_frame"], info["release_frame"], len(state) - 1})
    captures = {c: cv2.VideoCapture(str(directory / f"{c}.mp4")) for c in CAMERAS}
    results = []
    try:
        for sequence, frame in enumerate(frames):
            images = {}
            for camera, capture in captures.items():
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
                ok, image = capture.read()
                if not ok:
                    raise RuntimeError("cannot read recorded image")
                image = resize_with_pad(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 224, 224)
                images[camera] = base64.b64encode(image.tobytes()).decode()
            request = {
                "sequence": sequence,
                "contract_sha256": CONTRACT_SHA,
                "prompt": PROMPT,
                "state": state[frame].tolist(),
                "images": images,
                "image_encoding": "rgb224",
            }
            started = time.monotonic()
            prediction, response = client.infer(request, metadata["snapshot_sha256"])
            results.append(
                {
                    "frame": frame,
                    "roundtrip_ms": (time.monotonic() - started) * 1000,
                    "server_ms": response["server_ms"],
                    "first_arm_delta_rad": float(np.abs(prediction[0, :7] - state[frame, :7]).max()),
                    "gripper_raw": prediction[:, 7].tolist(),
                    "recorded_grip": float(action[frame, 7]),
                }
            )
    finally:
        for capture in captures.values():
            capture.release()
    result = {
        "metadata": metadata,
        "cases": results,
        "robot_commands_sent": 0,
        "scope": "recorded observations, not a robot rollout",
    }
    with Path(args.output).open("x") as output:
        json.dump(result, output, indent=2)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
