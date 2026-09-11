"""Shared contracts stay small, backwards compatible and fail closed."""

import ast
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from examples.hv1 import artifacts
from examples.hv1 import checkpoints
from examples.hv1 import metrics
from examples.hv1 import two_track
from examples.hv1 import two_track_config
from examples.hv1 import two_track_eval
from examples.hv1 import two_track_train
from examples.hv1 import workflow


def test_old_imports_are_aliases_not_forked_implementations():
    for name in ("ContractError", "digest", "file_hash", "read_json", "write_new_json"):
        assert getattr(workflow, name) is getattr(artifacts, name)
    assert two_track.checked is artifacts.checked
    assert two_track.sealed is artifacts.sealed
    assert two_track_train.save_snapshot is checkpoints.save_snapshot
    assert two_track_train.configure is two_track_config.configure
    assert two_track_train.local_dataset is two_track_config.local_dataset
    assert two_track_eval.snapshot_identity is checkpoints.snapshot_identity
    assert two_track_eval.transition_metrics is metrics.transition_metrics
    assert two_track_eval.condition_intents is metrics.condition_intents
    assert two_track_eval.cross_modal_matrix is metrics.cross_modal_matrix


def test_shared_artifacts_have_only_standard_library_dependencies():
    tree = ast.parse(Path(artifacts.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(a.name.split(".")[0] in sys.stdlib_module_names for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0 and node.module.split(".")[0] in sys.stdlib_module_names


@pytest.mark.parametrize("generation", ["overnight", "readapt"])
def test_a_finished_campaign_leaves_nothing_importable(generation):
    """`overnight_*` (A-F) and `readapt_*` (N/M) were both deleted on
    2026-09-11. A stale import is a half-removal that only fails when that path
    is finally taken, which on this project means during a run."""
    root = Path(artifacts.__file__).parent
    assert not list(root.glob(f"{generation}*.py"))
    for path in sorted(root.glob("*.py")) + sorted((root / "tests").glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert generation not in node.module, path.name
            elif isinstance(node, ast.Import):
                assert all(generation not in alias.name for alias in node.names), path.name


def test_evaluation_and_configuration_never_import_the_trainer():
    root = Path(artifacts.__file__).parent
    for path in root.glob("two_track*.py"):
        if path.name not in ("two_track_eval.py", "two_track_config.py"):
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert "two_track_train" not in node.module, path.name


def test_the_training_pipeline_never_touches_the_robot():
    """Robot I/O lives only under `ros/`. A trainer or evaluator that could
    publish a command would put a GPU job on the same wire as the arm."""
    root = Path(artifacts.__file__).parent
    for path in sorted(root.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for forbidden in ("import rclpy", "import paho", "create_publisher(", "ActionClient("):
            assert forbidden not in text, f"{path.name}: {forbidden}"
    ros = root / "ros/keti_humanoid_inference/keti_humanoid_inference"
    assert "import rclpy" in (ros / "node.py").read_text(encoding="utf-8")


def test_immutable_manifest_and_mutable_progress(tmp_path):
    manifest = tmp_path / "manifest.json"
    sealed = artifacts.sealed({"prompt": "은색 실린더"})
    artifacts.write_new_json(manifest, sealed)
    before = artifacts.file_hash(manifest)
    assert artifacts.checked(manifest) == sealed
    with pytest.raises(FileExistsError):
        artifacts.write_new_json(manifest, {"replacement": True})
    assert artifacts.file_hash(manifest) == before
    # The seal is the integrity check: an edit that keeps the file valid JSON
    # still has to fail, because a manifest is what every later hash is bound to.
    edited = dict(sealed, prompt="something else")
    manifest.write_text(json.dumps(edited), encoding="utf-8")
    with pytest.raises(artifacts.ContractError):
        artifacts.checked(manifest)
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


def test_a_snapshot_cannot_name_a_file_outside_itself(tmp_path):
    """`files_sha256` keys are joined onto the snapshot directory, so a relative
    escape would let a snapshot claim identity from bytes it does not own."""
    artifacts.write_new_json(
        tmp_path / "snapshot.json",
        dict(complete=True, cpu_roundtrip_pass=True, files_sha256={"../escape": "bad"}),
    )
    with pytest.raises(artifacts.ContractError):
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
        "deploy_server",
        "two_track",
        "two_track_run",
        "two_track_train",
        "two_track_eval",
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
