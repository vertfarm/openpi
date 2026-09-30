"""This repository is public. Nothing that locates or unlocks a machine may land in it.

On 2026-09-30 the history still held the MQTT broker's LAN address and an SSH
tunnel command with its account name - neither a secret, but both the kind of
line that turns into one. They were removed from the tree; this keeps the tree
that way. Loopback (127.0.0.1) is allowed: the inference server binds to it.
Use placeholders such as `<MQTT_HOST>` and pass real values on the command line.
"""

from pathlib import Path
import re

import pytest

REPO = Path(__file__).resolve().parents[3]
SCANNED = [REPO / "AGENTS.md", REPO / "CLAUDE.md"] + sorted(
    path
    for path in (REPO / "examples/hv1").rglob("*")
    if path.is_file()
    and path.suffix in {".py", ".md", ".json", ".yaml", ".yml", ".txt", ".html", ".cfg", ".xml", ".sh", ".toml"}
    and "__pycache__" not in path.parts
    and path.name != Path(__file__).name
)
FORBIDDEN = {
    "private IPv4 address": re.compile(r"\b(?:10\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])|192\.168)\.\d{1,3}\.\d{1,3}\b"),
    "ssh command with a port": re.compile(r"\bssh\s+(?:-\w+\s+)*-p\s*\d+"),
    "user@address": re.compile(r"\b[a-z_][\w.-]*@\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
    "private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "GitHub token": re.compile(r"\b(?:ghp|gho|ghs|ghu)_[A-Za-z0-9]{20,}|\bgithub_pat_\w{20,}"),
    # A literal value: quoted, or a bare word ending the line. Code such as
    # `token = secrets.token_urlsafe(32)` is an expression, not a credential.
    "inline credential": re.compile(
        r"(?im)\b(?:password|passwd|secret|api[_-]?key|token)\b\s*[:=]\s*"
        r"(?:[\"'](?!<)[^\"'\s]{6,}[\"']|(?!<)[\w!@#$%^&*+-]{6,}\s*$)"
    ),
}


def test_scan_covers_the_documents_and_code():
    names = {path.name for path in SCANNED}
    assert {"AGENTS.md", "STATUS.md", "DEPLOYMENT.md", "node.py", "core.py"} <= names


@pytest.mark.parametrize("path", SCANNED, ids=lambda path: str(path.relative_to(REPO)))
def test_no_access_details_in_the_public_tree(path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    found = [
        f"{kind}: line {text.count(chr(10), 0, match.start()) + 1}"
        for kind, pattern in FORBIDDEN.items()
        for match in pattern.finditer(text)
    ]
    assert not found, f"{path.relative_to(REPO)} - {found}. Use a placeholder such as <MQTT_HOST>."


def test_the_patterns_catch_what_they_are_for():
    samples = {
        "private IPv4 address": "--mqtt-host 192.168.0.9",
        "ssh command with a port": "ssh -p 2200 host",
        "user@address": "robot@10.0.0.5",
        "private key": "-----BEGIN OPENSSH PRIVATE KEY-----",
        "GitHub token": "ghp_" + "a" * 36,
        "inline credential": "password=hunter22",
    }
    samples_quoted = {
        "inline credential": 'api_key = "abcd1234efgh"',
    }
    for kind, sample in [*samples.items(), *samples_quoted.items()]:
        assert FORBIDDEN[kind].search(sample), kind
    for allowed in (
        "127.0.0.1:8000",
        "--mqtt-host <MQTT_HOST>",
        "token budget",
        "password: <ask on site>",
        "token = secrets.token_urlsafe(32)",
    ):
        assert not any(pattern.search(allowed) for pattern in FORBIDDEN.values()), allowed
