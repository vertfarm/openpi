"""Synthetic sources for the HV1 data contracts.

Nothing here reads a real recording. The raw sessions are read-only custody and
a test that needed them could not run anywhere but the field machine, so every
fixture writes the smallest file that still satisfies the source format:
`keti_hdf5_hevc_v1` numeric rows, one mp4 per camera, and the session
`metadata.json` the campaign scans.

`cycles` is the grasp timeline as (close_frame, release_frame) pairs. The
recorder writes the hand's *opening*, so a closed frame carries 0.0 and an open
one 0.6, and `native.read_numeric` turns that back into the intent channel.
"""

import json

import numpy as np
import pytest

from examples.hv1 import native

HAND_CLOSED, HAND_OPEN = 0.0, 0.6
LABELS = {
    "observation.state.upper_body.joint": native.ARM_STATE,
    "observation.state.hand.joint_r": native.HAND_STATE,
    "action.upper_body.joint": native.ARM_ACTION,
    "action.hand.command_r": ["mode", "open"],
}


def _write_numeric(path, frames, cycles, created_at, stale=False, hand=0.0, arm_step=0.0):
    import h5py

    opening = np.full(frames, HAND_OPEN, dtype=np.float32)
    for close, release in cycles:
        opening[close:release] = HAND_CLOSED
    with h5py.File(path, "w") as handle:
        meta = handle.create_group("meta")
        meta.attrs.update(labels=json.dumps(LABELS), num_frames=frames, created_at=created_at)
        for key, names in LABELS.items():
            handle.create_dataset(key, data=np.zeros((frames, len(names)), np.float32))
        handle["observation.state.hand.joint_r"][:] = hand
        # The arm command is a plain ramp: constant by default, so a test can
        # read `max_command_step_rad` off it, and varying when a test needs
        # normalization statistics that are not degenerate.
        handle["action.upper_body.joint"][:] = 0.1 + arm_step * np.arange(frames, dtype=np.float32)[:, None]
        handle["action.hand.command_r"][:] = np.stack([np.full(frames, 2.0), opening], axis=1)
        handle.create_dataset("timestamp", data=np.arange(frames) / 30 + 1)
        handle.create_dataset("stamp_ns", data=np.arange(frames, dtype=np.int64) * 33333333)
        handle.create_dataset("stale", data=np.full(frames, int(stale), np.uint32))


def level(tint, frame):
    """The flat brightness that marks one (episode, frame) in the videos.

    A test can read this back out of a loaded batch and say which row it got,
    which is the only way to see that a sampler index landed where it meant to.
    """
    return 25 + (tint + frame) % 200


def _write_videos(directory, frames, real, tint=0):
    for index, camera in enumerate(native.CAMERAS):
        path = directory / f"{camera}.mp4"
        if not real:
            # Only the bytes are read - source hashing and custody checks never
            # decode. Tests that decode ask for real=True and need `av`.
            path.write_bytes(b"inert placeholder for " + camera.encode())
            continue
        import av

        with av.open(str(path), mode="w") as out:
            stream = out.add_stream("mpeg4", rate=30)
            stream.width, stream.height, stream.pix_fmt = 640, 480, "yuv420p"
            for frame in range(frames):
                # One channel per camera, so a swapped camera key is visible too.
                image = np.zeros((480, 640, 3), np.uint8)
                image[..., index] = level(tint, frame)
                for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                    out.mux(packet)
            for packet in stream.encode():
                out.mux(packet)


@pytest.fixture
def source_episode():
    """Write one synthetic episode directory and return its numeric summary."""

    def build(
        directory,
        *,
        episode_id=None,
        frames=6,
        cycles=((2, 4),),
        video=False,
        stale=False,
        created_at="2026-09-10T03:00:00Z",
        prompt="fixture",
        hand=0.0,
        arm_step=0.0,
        tint=0,
    ):
        directory.mkdir(parents=True, exist_ok=True)
        _write_numeric(directory / "data.hdf5", frames, cycles, created_at, stale=stale, hand=hand, arm_step=arm_step)
        _write_videos(directory, frames, video, tint=tint)
        (directory / "tasks.json").write_text(
            json.dumps(
                dict(episode_id=episode_id or directory.name, num_frames=frames, main_prompt=prompt),
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return directory

    return build


@pytest.fixture
def source_session(source_episode):
    """Write a whole session root: `metadata.json` plus its `episodes/` subtree."""

    def build(
        root,
        session,
        episode_ids,
        *,
        frames=6,
        cycles=((2, 4),),
        video=False,
        hz=30,
        resolution=(640, 480),
        rotate=0,
        cameras=native.CAMERAS,
        declared=None,
        hand=0.0,
        arm_step=0.0,
        first_tint=0,
    ):
        root.mkdir(parents=True, exist_ok=True)
        for ordinal, episode_id in enumerate(sorted(episode_ids)):
            source_episode(
                root / "episodes" / episode_id,
                episode_id=episode_id,
                frames=frames,
                cycles=cycles,
                video=video,
                hand=hand,
                arm_step=arm_step,
                tint=first_tint + ordinal * frames,
            )
        (root / "metadata.json").write_text(
            json.dumps(
                dict(
                    dataset_name=session,
                    hz=hz,
                    cameras={c: dict(resolution=list(resolution), rotate=rotate) for c in cameras},
                    episodes=[
                        dict(id=episode_id, num_frames=frames)
                        for episode_id in sorted(declared if declared is not None else episode_ids)
                    ],
                )
            ),
            encoding="utf-8",
        )
        return root

    return build
