"""Re-solve a recorded session's right-arm joints for another arm, keeping each recorded tool pose.

The 2026-09-09/10 sessions were recorded before the 2026-09-30 wrist pitch
remount and the 2026-10-20 upper body. The same angles put today's tool
centimetres away and turned, so a policy trained on them reaches for the wrong
place. This writes a
new session whose right-arm observation and command rows are the new arm's
angles for the recorded tool poses; everything else - left arm and head, hands,
timing, tasks and videos - is copied unchanged (the cameras saw the tool where
it was). The source session is only read. Right arm only: that is the policy
contract, and its limits are the only ones measured.

Nothing is written unless, in every episode, the `--from-urdf` arm reproduces
the tool poses the recorder logged, both right-arm tables were found, and every
row was re-solved inside the field-measured limits without a jump. The first
check is what names the recording-time arm: today's controller URDF already
carries the remounted wrist and leaves centimetres against the log, the URDF of
the recording date leaves a fixed tool offset. The URDF limits are CAD
placeholders, so an approved field profile is required.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import h5py
import numpy as np

from . import native
from .artifacts import ContractError
from .artifacts import file_hash
from .artifacts import read_json
from .kinematics import Chain
from .kinematics import Remap
from .kinematics import field_limits
from .kinematics import fit_tool

TOOL = "Right_tool0"
RECORDED_POSE = "observation.state.upper_body.pose_r"
POSE_COLUMNS = [f"pose/position/{a}" for a in "xyz"] + [f"pose/orientation/{a}" for a in "xyzw"]
# The right model leaves micrometres against the log (the controller computed it
# from the same angles); a wrong wrist model leaves tens of millimetres.
POSE_MATCH_MM, POSE_MATCH_DEG = 1.0, 0.1
MIN_JOINT_SPAN_RAD = 0.05
COLUMNS = {
    "observation.state.upper_body.joint": native.ARM_STATE,
    "action.upper_body.joint": native.ARM_ACTION,
}


def _solve(remap, rows):
    """New-arm angles for every row, seeded from the previous answer; refuses on any miss or jump."""
    try:
        return remap.trajectory(rows, seed=rows[0])
    except ContractError:
        pass
    # Name every failing row before refusing, so the report says where.
    misses, seed = [], rows[0]
    for index, row in enumerate(rows):
        try:
            seed = remap.convert(row, seed)
        except ContractError:
            misses.append(index)
            seed = row
    if not misses:
        raise ContractError("rows re-solve one by one but jump between them (remap is discontinuous)")
    shown = ", ".join(map(str, misses[:5])) + (" ..." if len(misses) > 5 else "")
    raise ContractError(f"{len(misses)} of {len(rows)} rows unreachable (rows {shown})")


def _columns(f, labels, key, names):
    declared = labels.get(key, [])
    if key not in f or not set(names) <= set(declared):
        raise ContractError(f"{key} lacks the columns {names}")
    return [declared.index(name) for name in names]


def _logged_rows(path):
    """Right-arm joints and logged tool poses of one recording, rows without a logged pose left out."""
    with h5py.File(path, "r") as f:
        labels = json.loads(f["meta"].attrs["labels"])
        joints = _columns(f, labels, "observation.state.upper_body.joint", native.ARM_STATE)
        columns = _columns(f, labels, RECORDED_POSE, POSE_COLUMNS)
        q = f["observation.state.upper_body.joint"][:][:, joints].astype(np.float64)
        pose = f[RECORDED_POSE][:][:, columns].astype(np.float64)
    # Rows logged before the first pose message are padded, not measured.
    valid = (
        np.isfinite(pose).all(axis=1)
        & np.isfinite(q).all(axis=1)
        & (np.abs(np.linalg.norm(pose[:, 3:], axis=1) - 1.0) < 0.01)
        & (np.linalg.norm(pose[:, :3], axis=1) > 1e-6)
    )
    if valid.sum() < max(3, len(pose) // 2):
        raise ContractError(f"{RECORDED_POSE} has {int(valid.sum())} usable rows of {len(pose)}")
    return q[valid], pose[valid]


def check_source_arm(chain, paths):
    """Refuse unless `chain` reproduces the logged tool poses of every recording with one fixed tool transform.

    One transform for the whole session, not one per episode: the recorder's TCP
    is a constant, and six free parameters per episode would absorb a wrong
    joint in an episode where the arm barely moves. For the same reason every
    joint must move enough across the session for a wrong model of it to show.
    """
    rows = {}
    for episode, path in paths.items():
        try:
            rows[episode] = _logged_rows(path)
        except ContractError as error:
            raise ContractError(f"{episode}: {error}") from error
    q = np.concatenate([r[0] for r in rows.values()])
    pose = np.concatenate([r[1] for r in rows.values()])
    span = q.max(axis=0) - q.min(axis=0)
    if np.any(span < MIN_JOINT_SPAN_RAD):
        still = [native.ARM_STATE[i].removeprefix("position/") for i in np.flatnonzero(span < MIN_JOINT_SPAN_RAD)]
        raise ContractError(f"cannot verify the --from-urdf arm: {', '.join(still)} barely move in the recordings")
    _, offset, position, rotation = fit_tool(chain, q, pose[:, :3], pose[:, 3:])
    bounds = np.cumsum([0, *(len(r[0]) for r in rows.values())])
    per_episode = {
        episode: 1000 * float(np.median(position[a:b])) for episode, a, b in zip(rows, bounds[:-1], bounds[1:])
    }
    match = dict(
        rows=len(q),
        median_mm=1000 * float(np.median(position)),
        p90_mm=1000 * float(np.percentile(position, 90)),
        max_mm=1000 * float(position.max()),
        median_deg=float(np.degrees(np.median(rotation))),
        tool_offset_mm=[1000 * float(v) for v in offset],
        episode_median_mm=per_episode,
    )
    worst = max(per_episode, key=per_episode.get)
    if match["median_mm"] > POSE_MATCH_MM or match["median_deg"] > POSE_MATCH_DEG or per_episode[worst] > POSE_MATCH_MM:
        raise ContractError(
            f"the --from-urdf arm does not reproduce the recorded tool pose: {match['median_mm']:.2f} mm / "
            f"{match['median_deg']:.3f} deg median over {len(q)} rows after one fixed tool fit (limits "
            f"{POSE_MATCH_MM} mm / {POSE_MATCH_DEG} deg), worst episode {worst} {per_episode[worst]:.2f} mm; "
            "it is not the arm this was recorded on"
        )
    return match


def _retarget_file(path, remap):
    """Rewrite both right-arm tables of one `data.hdf5` in place; return the worst tool residual (m)."""
    worst, rewritten = 0.0, []
    with h5py.File(path, "r+") as f:
        labels = json.loads(f["meta"].attrs["labels"])
        for key, names in COLUMNS.items():
            columns = _columns(f, labels, key, names)
            table = f[key][:]
            rows = table[:, columns].astype(np.float64)
            solved = _solve(remap, rows)
            for q_old, q_new in zip(rows, solved):
                worst = max(worst, float(np.linalg.norm(remap.target.fk(q_new)[1] - remap.source.fk(q_old)[1])))
            table[:, columns] = solved.astype(table.dtype)
            f[key][:] = table
            rewritten.append(key)
    if rewritten != list(COLUMNS):
        raise ContractError("not every right-arm table was rewritten")
    return worst


def retarget_session(source, output, from_urdf, to_urdf, limits, *, episodes=None):
    """Write `output` as `source` re-solved from the `from_urdf` arm to the `to_urdf` arm.

    `limits` is `(joint_min, joint_max)` or `(joint_min, joint_max, profile_sha256)`
    for the right arm, from the approved field profile. With `episodes`, only
    those episode ids are re-solved and the session's other episodes are left
    out of the output, never copied through with old-arm rows; metadata.json
    stays whole because `prepare` checks the session inventory.
    """
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise ContractError(f"output exists: {output}")
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ContractError("output and source must not contain each other")
    old, new = Chain.from_urdf(from_urdf, TOOL), Chain.from_urdf(to_urdf, TOOL)
    if old.names != new.names or [f"position/{n}" for n in old.names] != native.ARM_STATE:
        raise ContractError("both URDFs must carry the contract's right-arm joints in order")
    remap = Remap(old, new.with_limits(limits[0], limits[1]))
    metadata = read_json(source / "metadata.json")
    listed = [episode["id"] for episode in metadata["episodes"]]
    keep = set(listed) if episodes is None else set(episodes)
    if not keep or keep - set(listed):
        raise ContractError(f"episodes not in the session: {sorted(keep - set(listed))}")
    kept = [e["id"] for e in metadata["episodes"] if e["id"] in keep]
    recorded = check_source_arm(old, {e: source / "episodes" / e / "data.hdf5" for e in kept})
    partial = output.with_name(f".{output.name}.partial")
    if partial.exists():
        raise ContractError(f"a previous partial output is in the way: {partial}")
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, partial)
        omitted = sorted(set(listed) - keep)
        for episode_id in omitted:
            # Inside the copy this call just made; the source keeps every episode.
            shutil.rmtree(partial / "episodes" / episode_id)
        solved = {}
        for episode in (e for e in metadata["episodes"] if e["id"] in keep):
            path = partial / "episodes" / episode["id"] / "data.hdf5"
            try:
                worst = _retarget_file(path, remap)
            except ContractError as error:
                raise ContractError(f"{episode['id']}: {error}") from error
            if file_hash(path) == file_hash(source / "episodes" / episode["id"] / "data.hdf5"):
                raise ContractError(f"{episode['id']}: nothing changed")
            solved[episode["id"]] = dict(
                source_sha256=file_hash(source / "episodes" / episode["id"] / "data.hdf5"),
                output_sha256=file_hash(path),
                max_tool_residual_mm=1000 * worst,
            )
        report = dict(
            purpose="right-arm joints re-solved from the recording-time arm for today's, keeping each tool pose",
            from_urdf_sha256=file_hash(from_urdf),
            to_urdf_sha256=file_hash(to_urdf),
            limits_right=[list(map(float, limits[0])), list(map(float, limits[1]))],
            limits_profile_sha256=limits[2] if len(limits) > 2 else None,
            recorded_pose_match=recorded,
            episodes=solved,
            omitted=omitted,
        )
        (partial / "retarget.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        partial.rename(output)
    except BaseException:
        # Only the directory this call created; the source is never touched.
        if partial.exists():
            shutil.rmtree(partial)
        raise
    return report


def campaign_episodes(manifest_path, session):
    """The episode ids a campaign manifest uses from one session."""
    ids = {e["episode_id"] for e in read_json(manifest_path)["episodes"] if e.get("session_id") == session}
    if not ids:
        raise ContractError(f"the campaign uses no episodes of {session}")
    return ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="recorded session root (metadata.json, episodes/)")
    parser.add_argument("--output", required=True, help="new session root; its name stays the dataset name")
    parser.add_argument("--from-urdf", required=True, help="the arm the session was recorded on")
    parser.add_argument("--to-urdf", required=True, help="the arm to re-solve for")
    parser.add_argument("--limits", required=True, help="approved field profile with the right-arm joint_min/max")
    parser.add_argument("--campaign-manifest", help="re-solve only the episodes this campaign uses; drop the rest")
    args = parser.parse_args()
    episodes = None
    if args.campaign_manifest:
        episodes = campaign_episodes(
            args.campaign_manifest, read_json(Path(args.source) / "metadata.json")["dataset_name"]
        )
    report = retarget_session(
        args.source, args.output, args.from_urdf, args.to_urdf, field_limits(args.limits), episodes=episodes
    )
    worst = max(e["max_tool_residual_mm"] for e in report["episodes"].values())
    print(json.dumps(dict(episodes=len(report["episodes"]), omitted=report["omitted"], max_tool_residual_mm=worst)))


if __name__ == "__main__":
    main()
