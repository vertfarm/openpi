"""Shared contracts stay small, backwards compatible and fail closed."""

import ast
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from examples.hv1 import artifacts
from examples.hv1 import checkpoints
from examples.hv1 import overnight_common
from examples.hv1 import overnight_train
from examples.hv1 import readapt
from examples.hv1 import readapt_config
from examples.hv1 import readapt_eval
from examples.hv1 import readapt_train
from examples.hv1 import workflow


def test_old_imports_are_aliases_not_forked_implementations():
    for name in ("ContractError", "digest", "file_hash", "read_json", "write_new_json"):
        assert getattr(workflow, name) is getattr(artifacts, name)
    for name in ("atomic_json", "storage_gate", "tree_bytes"):
        assert getattr(overnight_common, name) is getattr(artifacts, name)
    assert readapt.checked is artifacts.checked
    assert readapt.sealed is artifacts.sealed
    assert overnight_train.save_snapshot is checkpoints.save_snapshot
    assert readapt_train.save_snapshot is checkpoints.save_snapshot
    assert readapt_eval.snapshot_identity is checkpoints.snapshot_identity
    assert readapt_train.configure is readapt_config.configure
    assert readapt_train.local_dataset is readapt_config.local_dataset


def test_shared_artifacts_have_only_standard_library_dependencies():
    tree = ast.parse(Path(artifacts.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(a.name.split(".")[0] in sys.stdlib_module_names for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0 and node.module.split(".")[0] in sys.stdlib_module_names


def test_readaptation_does_not_depend_on_overnight_or_evaluation_on_training():
    root = Path(readapt.__file__).parent
    for path in root.glob("readapt*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert "overnight" not in node.module
                if path.name in ("readapt_eval.py", "readapt_config.py"):
                    assert "readapt_train" not in node.module


def test_immutable_manifest_and_mutable_progress(tmp_path):
    manifest = tmp_path / "manifest.json"
    sealed = artifacts.sealed({"prompt": "은색 실린더"})
    artifacts.write_new_json(manifest, sealed)
    before = artifacts.file_hash(manifest)
    assert artifacts.checked(manifest) == sealed
    with pytest.raises(FileExistsError):
        artifacts.write_new_json(manifest, {"replacement": True})
    assert artifacts.file_hash(manifest) == before
    progress = tmp_path / "progress.json"
    artifacts.atomic_json(progress, {"step": 1})
    artifacts.atomic_json(progress, {"step": 2})
    assert artifacts.read_json(progress) == {"step": 2}
    assert not list(tmp_path.glob("*.partial-*"))


def test_disk_reserve_includes_pending_write(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts.shutil, "disk_usage", lambda _: SimpleNamespace(free=58 * artifacts.GIB))
    artifacts.storage_gate(tmp_path, 8 * artifacts.GIB)
    with pytest.raises(artifacts.ContractError, match="disk budget"):
        artifacts.storage_gate(tmp_path, 8 * artifacts.GIB + 1)


def test_shared_snapshot_identity_rejects_changed_bytes(tmp_path):
    weights = tmp_path / "weights.inert"
    weights.write_bytes(b"synthetic weights, not a model")
    record = dict(complete=True, cpu_roundtrip_pass=True, files_sha256={weights.name: artifacts.file_hash(weights)})
    artifacts.write_new_json(tmp_path / "snapshot.json", record)
    assert checkpoints.snapshot_identity(tmp_path) == record
    weights.write_bytes(b"modified")
    with pytest.raises(artifacts.ContractError, match="hash mismatch"):
        checkpoints.snapshot_identity(tmp_path)


@pytest.mark.parametrize("field", ["complete", "cpu_roundtrip_pass", "files_sha256"])
def test_shared_snapshot_requires_complete_evidence(tmp_path, field):
    record = dict(complete=True, cpu_roundtrip_pass=True, files_sha256={"inert": "unused"})
    record[field] = False
    artifacts.write_new_json(tmp_path / "snapshot.json", record)
    with pytest.raises(artifacts.ContractError, match="incomplete snapshot"):
        checkpoints.snapshot_identity(tmp_path)


@pytest.mark.parametrize(
    "module",
    [
        "readapt",
        "readapt_run",
        "readapt_train",
        "readapt_eval",
        "deploy_server",
        "overnight",
        "overnight_train",
        "overnight_eval",
    ],
)
def test_existing_cli_help_never_starts_training_or_ros(module):
    repo = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [sys.executable, "-B", "-m", f"examples.hv1.{module}", "--help"],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
