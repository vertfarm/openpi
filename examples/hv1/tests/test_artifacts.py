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
from examples.hv1 import pipeline
from examples.hv1 import pipeline_config
from examples.hv1 import pipeline_eval
from examples.hv1 import pipeline_train
from examples.hv1 import workflow


def test_old_imports_are_aliases_not_forked_implementations():
    for name in ("ContractError", "digest", "file_hash", "read_json", "write_new_json"):
        assert getattr(workflow, name) is getattr(artifacts, name)
    assert pipeline.checked is artifacts.checked
    assert pipeline.sealed is artifacts.sealed
    assert pipeline_train.save_snapshot is checkpoints.save_snapshot
    assert pipeline_train.configure is pipeline_config.configure
    assert pipeline_train.local_dataset is pipeline_config.local_dataset
    assert pipeline_eval.snapshot_identity is checkpoints.snapshot_identity
    assert pipeline_eval.transition_metrics is metrics.transition_metrics
    assert pipeline_eval.condition_intents is metrics.condition_intents
    assert pipeline_eval.cross_modal_matrix is metrics.cross_modal_matrix


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
    for path in root.glob("pipeline*.py"):
        if path.name not in ("pipeline_eval.py", "pipeline_config.py"):
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert "pipeline_train" not in node.module, path.name


def test_the_training_pipeline_never_touches_the_robot():
    """Robot I/O lives under `ros/` and `tools/` and nowhere else. A trainer or
    evaluator that could publish a command would put a GPU job on the same wire
    as the arm; the operator tools are supervised and run in the container."""
    root = Path(artifacts.__file__).parent
    for path in sorted(root.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for forbidden in ("import rclpy", "import paho", "create_publisher(", "ActionClient("):
            assert forbidden not in text, f"{path.name}: {forbidden}"
    ros = root / "ros/keti_humanoid_inference/keti_humanoid_inference"
    assert "import rclpy" in (ros / "node.py").read_text(encoding="utf-8")


def test_the_operator_tools_travel_with_the_repository():
    """They used to live loose in ~/workspace, where a machine change would lose
    them - and STATUS cites the camera one as the way to settle a geometry
    question that already cost an afternoon once."""
    tools = Path(artifacts.__file__).parent / "tools"
    names = {path.name for path in tools.glob("*.py")}
    assert {"check_camera_input.py", "return_to_start.py"} <= names
    for path in sorted(tools.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert text.startswith('"""'), f"{path.name} must say what it is on line one"
        # They drive the arm, so they belong to the supervised path, not the
        # host pipeline - and must never be imported by it.
        assert "import rclpy" in text


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
    """Expressed against MIN_FREE rather than a literal, so moving the floor
    changes the floor and not what the gate means."""
    reserve = 8 * artifacts.GIB
    free = artifacts.MIN_FREE + reserve
    monkeypatch.setattr(artifacts.shutil, "disk_usage", lambda _: SimpleNamespace(free=free))
    artifacts.storage_gate(tmp_path, reserve)
    with pytest.raises(artifacts.ContractError, match="disk budget"):
        artifacts.storage_gate(tmp_path, reserve + 1)


def test_a_full_training_run_still_has_to_fit():
    """pipeline_train reserves 40 GiB for a non-fine-tune run, so the floor plus
    that reservation is what the machine actually has to have free."""
    assert artifacts.MIN_FREE + 40 * artifacts.GIB < 100 * artifacts.GIB


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


# Every module a person can run, and why it exists. Keep this list honest: the
# only reason the surface grew to 33 commands across 12 modules by 2026-09-11 was
# that two retired generations kept their entry points after their callers were
# gone. `native` and `openpi_run` became libraries when theirs were removed.
CAMPAIGN_CLIS = ("pipeline", "pipeline_run", "pipeline_train", "pipeline_eval", "deploy_server")
OPERATOR_CLIS = ("deploy_smoke", "shadow_eval", "source_contract", "verify_export")
SYNTHETIC_CLIS = ("cli",)
LIBRARIES_ONLY = ("artifacts", "checkpoints", "metrics", "native", "openpi_run", "pipeline_config", "transforms")


def test_the_entry_point_surface_is_the_declared_one():
    """A module with a `main()` is something an operator can run, so adding one
    is a decision. This fails on a new entry point until it is declared here."""
    root = Path(artifacts.__file__).parent
    runnable = set()
    for path in sorted(root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if any(isinstance(node, ast.FunctionDef) and node.name == "main" for node in tree.body):
            runnable.add(path.stem)
    assert runnable == {*CAMPAIGN_CLIS, *OPERATOR_CLIS, *SYNTHETIC_CLIS}
    for name in LIBRARIES_ONLY:
        assert (root / f"{name}.py").is_file()
        assert name not in runnable, f"{name} grew an entry point"


@pytest.mark.parametrize("module", [*CAMPAIGN_CLIS, *OPERATOR_CLIS, *SYNTHETIC_CLIS])
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
