"""Compare the model's actual 224x224 input: training data vs the live cameras.

Run after changing camera mounting or configuration:
    ~/workspace/openpi-hv1/.venv/bin/python -B ~/workspace/hv1-check-camera-input.py

Writes a labelled comparison sheet and prints the geometry that matters.
The live frames go through the identical path deploy_server.prepare_images uses,
so what the sheet shows is what the policy sees.
"""

import subprocess
import sys

import av
import cv2
import numpy as np

sys.path.insert(0, "/home/keti/workspace/openpi-hv1")
from openpi_client.image_tools import resize_with_pad  # noqa: E402

CAMERAS = ("head", "hand_l", "hand_r")
EPISODE = "/home/keti/workspace/keti_humanoid_ros2/datasets/keti_humanoid_data_260910/episodes/episode_000000"
OUT = "/home/keti/workspace/hv1-camera-input-check.jpg"
TOPICS = {
    "head": "/kh/upper_body/head/color/image_raw/compressed",
    "hand_l": "/kdex_3f/left/camera/image_raw/compressed",
    "hand_r": "/kdex_3f/right/camera/image_raw/compressed",
}

GRAB = r"""
import rclpy, cv2, numpy as np
from sensor_msgs.msg import CompressedImage
rclpy.init(); n = rclpy.create_node("grab_check")
for name, topic in %s:
    box = [None]
    s = n.create_subscription(CompressedImage, topic, lambda m, b=box: b.__setitem__(0, m), 1)
    for _ in range(200):
        rclpy.spin_once(n, timeout_sec=0.1)
        if box[0] is not None:
            break
    if box[0] is None:
        raise SystemExit(f"no frame on {topic}")
    img = cv2.imdecode(np.frombuffer(box[0].data, np.uint8), cv2.IMREAD_COLOR)
    cv2.imwrite(f"/tmp/_check_{name}.jpg", img)
    n.destroy_subscription(s)
""" % repr([(c, TOPICS[c]) for c in CAMERAS])


def prepare(bgr):
    """Exactly what deploy_server.prepare_images does to a live frame."""
    if bgr.shape[:2] != (480, 640):
        bgr = cv2.resize(bgr, (640, 480), interpolation=cv2.INTER_AREA)
    return resize_with_pad(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), 224, 224)


def grab_live():
    with open("/tmp/_grab_check.py", "w", encoding="utf-8") as handle:
        handle.write(GRAB)
    subprocess.run(
        ["docker", "cp", "/tmp/_grab_check.py", "keti_humanoid_ros2_jazzy:/tmp/_grab_check.py"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "docker",
            "exec",
            "keti_humanoid_ros2_jazzy",
            "bash",
            "-lc",
            "source /opt/ros/jazzy/setup.bash; export ROS_DOMAIN_ID=10 "
            "RMW_IMPLEMENTATION=rmw_cyclonedds_cpp; python3 /tmp/_grab_check.py",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    frames = {}
    for c in CAMERAS:
        subprocess.run(
            ["docker", "cp", f"keti_humanoid_ros2_jazzy:/tmp/_check_{c}.jpg", f"/tmp/_check_{c}.jpg"],
            check=True,
            capture_output=True,
        )
        frames[c] = cv2.imread(f"/tmp/_check_{c}.jpg")
    return frames


def grab_train(frame_index=0):
    frames = {}
    for c in CAMERAS:
        container = av.open(f"{EPISODE}/{c}.mp4")
        for i, f in enumerate(container.decode(video=0)):
            if i == frame_index:
                frames[c] = f.to_ndarray(format="bgr24")
                break
        container.close()
    return frames


def label(panel, text, colour=(255, 255, 255)):
    out = panel.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(out, text, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
    return out


def mat_box(bgr):
    """Bounding box and area of the red work mat: a large, unambiguous landmark.

    Corners of the mat move with the scene layout, so comparing them against the
    training frame says which way to nudge the mat, which eyeballing does not.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue, sat, val = (hsv[:, :, i].astype(int) for i in range(3))
    mask = (((hue < 12) | (hue > 168)) & (sat > 90) & (val > 60)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return {
        "left": int(xs.min()),
        "right": int(xs.max()),
        "top": int(ys.min()),
        "bottom": int(ys.max()),
        "area%": round(100 * float(mask.mean()), 1),
    }


def main():
    """The recorder already normalises every camera to 640x480 before writing
    (frame_builder._image), and prepare_images does the same to live frames.
    So a stored-vs-published shape difference is expected and harmless; what
    must match is the 224x224 the policy actually receives.
    """
    train, live = grab_train(), grab_live()
    rows = []
    print(f"{'camera':8s}{'edge':8s}{'train':>7s}{'live':>7s}{'diff':>7s}")
    for c in CAMERAS:
        t = cv2.cvtColor(prepare(train[c]), cv2.COLOR_RGB2BGR)
        shot = cv2.cvtColor(prepare(live[c]), cv2.COLOR_RGB2BGR)
        box_t, box_l = mat_box(t), mat_box(shot)
        if box_t and box_l:
            for i, key in enumerate(("left", "right", "top", "bottom", "area%")):
                print(
                    f"{c if not i else '':8s}{key:8s}{box_t[key]:7.1f}{box_l[key]:7.1f}{box_l[key] - box_t[key]:+7.1f}"
                )
        else:
            print(f"{c:8s}{'(빨간 매트를 찾지 못함)':8s}")
        gap = np.full((224, 6, 3), 255, np.uint8)
        rows.append(
            np.hstack(
                [
                    label(t, f"TRAIN {c}"),
                    gap,
                    label(shot, f"LIVE  {c}", (0, 255, 0)),
                    gap,
                    label(cv2.addWeighted(t, 0.5, shot, 0.5, 0), f"BLEND {c}", (0, 255, 255)),
                ]
            )
        )
    sheet = np.vstack(
        [r for pair in zip(rows, [np.full((6, rows[0].shape[1], 3), 255, np.uint8)] * 3) for r in pair][:-1]
    )
    cv2.imwrite(OUT, sheet)
    print(f"\n{OUT}")
    print("BLEND에서 두 상이 겹치면 맞은 것이다. mat 경계 차이가 어느 방향으로")
    print("옮겨야 하는지 알려준다 — left가 음수면 매트가 왼쪽으로 밀린 것이다.")


if __name__ == "__main__":
    main()
