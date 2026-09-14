"""Cache and checksum a pi05_droid_jointpos checkpoint (JAX params/ or PyTorch model.safetensors)."""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
import pathlib

import tyro

from openpi.shared import download


@dataclasses.dataclass
class Args:
    checkpoint_dir: str = "gs://openpi-assets/checkpoints/pi05_droid_jointpos"
    manifest_path: pathlib.Path = pathlib.Path(
        "/data/keti/snu/home/runtime/pi05-droid-jointpos-velocity/checkpoint-manifest.json"
    )
    reuse_manifest: bool = True


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_is_reusable(manifest_path: pathlib.Path, source: str) -> bool:
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        local_path = pathlib.Path(manifest["local_path"])
    except (KeyError, OSError, ValueError):
        return False
    return (
        manifest.get("source") == source
        and _has_weights(local_path)
        and (local_path / "assets" / "droid" / "norm_stats.json").is_file()
    )


def _has_weights(local_path: pathlib.Path) -> bool:
    return (local_path / "params").is_dir() or (local_path / "model.safetensors").is_file()


def main(args: Args) -> None:
    if args.reuse_manifest and _manifest_is_reusable(args.manifest_path, args.checkpoint_dir):
        print(f"checkpoint_manifest={args.manifest_path.resolve()}")
        print("CHECKPOINT_PREPARE_STATUS=REUSED")
        return

    local_path = download.maybe_download(args.checkpoint_dir)
    norm_stats_path = local_path / "assets" / "droid" / "norm_stats.json"
    if not _has_weights(local_path):
        raise FileNotFoundError(
            f"Missing model weights: expected JAX directory {local_path / 'params'} "
            f"or PyTorch file {local_path / 'model.safetensors'}"
        )
    weight_format = "jax" if (local_path / "params").is_dir() else "pytorch"
    if not norm_stats_path.is_file():
        raise FileNotFoundError(f"Missing DROID norm stats: {norm_stats_path}")

    files = []
    total_bytes = 0
    for path in sorted(candidate for candidate in local_path.rglob("*") if candidate.is_file()):
        size = path.stat().st_size
        total_bytes += size
        files.append(
            {
                "path": path.relative_to(local_path).as_posix(),
                "bytes": size,
                "sha256": _sha256(path),
            }
        )

    manifest = {
        "source": args.checkpoint_dir,
        "local_path": str(local_path),
        "weight_format": weight_format,
        "generated_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "file_count": len(files),
        "total_bytes": total_bytes,
        "norm_stats_sha256": _sha256(norm_stats_path),
        "files": files,
    }
    args.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = args.manifest_path.with_suffix(args.manifest_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(args.manifest_path)

    print(f"checkpoint_source={args.checkpoint_dir}")
    print(f"checkpoint_local={local_path}")
    print(f"checkpoint_weight_format={weight_format}")
    print(f"checkpoint_files={len(files)}")
    print(f"checkpoint_bytes={total_bytes}")
    print(f"norm_stats_sha256={manifest['norm_stats_sha256']}")
    print(f"checkpoint_manifest={args.manifest_path.resolve()}")
    print("CHECKPOINT_PREPARE_STATUS=PASS")


if __name__ == "__main__":
    main(tyro.cli(Args))
