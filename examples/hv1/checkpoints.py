"""BF16 inference snapshots shared by overnight and readaptation workflows.

Model dependencies load only when saving. Identity checks never initialize a GPU.
"""

from __future__ import annotations

from pathlib import Path

from .artifacts import GIB
from .artifacts import ContractError
from .artifacts import file_hash
from .artifacts import read_json
from .artifacts import storage_gate
from .artifacts import tree_bytes
from .artifacts import write_new_json


def snapshot_identity(snapshot):
    snapshot = Path(snapshot).resolve()
    record = read_json(snapshot / "snapshot.json")
    if (
        record.get("complete") is not True
        or record.get("cpu_roundtrip_pass") is not True
        or not record.get("files_sha256")
    ):
        raise ContractError("incomplete snapshot")
    for name, sha in record["files_sha256"].items():
        p = (snapshot / name).resolve()
        if not p.is_relative_to(snapshot) or file_hash(p) != sha:
            raise ContractError("snapshot content hash mismatch")
    return record


def save_snapshot(campaign, config, state, data_loader, step, metrics):
    import jax
    import jax.numpy as jnp
    import numpy as np
    import orbax.checkpoint as ocp

    from openpi.models.model import restore_params
    from openpi.shared import normalize

    campaign = Path(campaign)
    final = campaign / "snapshots" / config.exp_name / f"step_{step:06d}"
    if final.exists():
        if (final / "snapshot.json").exists():
            return final
        raise ContractError("incomplete existing snapshot requires review")
    storage_gate(campaign, 8 * GIB)
    partial = final.with_name(final.name + ".partial")
    partial.mkdir(parents=True, exist_ok=False)
    # Transfer/cast on host. Do not allocate a second full model on the training GPU.
    params = jax.tree.map(lambda x: np.asarray(jax.device_get(x)).astype(jnp.bfloat16), state.params.to_pure_dict())
    with ocp.PyTreeCheckpointer() as saver:
        saver.save(str(partial / "params"), {"params": params})
    # Prove all serialized arrays round-trip, not only that a directory exists.
    restored = restore_params(partial / "params", dtype=jnp.bfloat16)
    left, right = jax.tree.leaves(params), jax.tree.leaves(restored)
    if jax.tree.structure(params) != jax.tree.structure(restored) or not all(
        np.array_equal(a, b) for a, b in zip(left, right, strict=True)
    ):
        raise ContractError("BF16 snapshot roundtrip mismatch")
    data = data_loader.data_config()
    normalize.save(partial / "assets" / data.asset_id, data.norm_stats)
    hashes = {str(p.relative_to(partial)): file_hash(p) for p in sorted(partial.rglob("*")) if p.is_file()}
    record = dict(
        config.policy_metadata,
        step=step,
        checkpoint_dtype="bfloat16",
        purpose="inference_not_optimizer_resume",
        metrics=metrics,
        files_sha256=hashes,
        bytes=tree_bytes(partial),
        cpu_roundtrip_pass=True,
        gpu_reload_status="pending",
        complete=True,
    )
    write_new_json(partial / "snapshot.json", record)
    partial.rename(final)
    return final
