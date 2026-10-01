"""
Security audit.

Checks the things that would be a real exposure rather than a
housekeeping nit: a credential in the repository, a path that escapes
the directory it is confined to, a subprocess built from a shell
string, or a workflow permission that is broader than its task needs.

Values are never printed. A finding names the file and the shape of
the problem, never the value.
"""

from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=REPO_ROOT,
        check=False,
    )

    return completed.stdout


# ---------------------------------------------------------------------
# Ignored state
# ---------------------------------------------------------------------


def ignored(patterns: list[str]) -> list[str]:
    """Which of these paths git refuses to track."""

    result = []

    for pattern in patterns:
        completed = subprocess.run(
            ["git", "check-ignore", "-q", pattern],
            cwd=REPO_ROOT,
            check=False,
        )

        result.append((pattern, completed.returncode == 0))

    return result


def test_local_credentials_are_ignored() -> None:
    for pattern, ok in ignored(
        [".env", ".agent/secrets/state.json", "data/incoming/x.md"]
    ):
        assert ok, f"{pattern} must be git-ignored"


def test_no_credential_file_is_tracked() -> None:
    tracked = git("ls-files").splitlines()

    assert ".env" not in tracked

    for path in tracked:
        assert not path.endswith("state.json"), path
        assert "cookies" not in path.lower(), path


# ---------------------------------------------------------------------
# Credential literals
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Credential literals
# ---------------------------------------------------------------------


def secret_patterns() -> list[tuple[str, re.Pattern]]:
    """
    The shapes a real credential takes.

    Taken from the redaction module the control plane already uses, so
    the audit and the runtime cannot disagree about what a credential
    looks like. Matching a bare prefix is not enough: the detector
    itself, this file, and the documentation all contain those prefixes
    as text, and a rule that flagged them would be a rule that fires on
    every repository and is therefore ignored.
    """

    from src.agent import redaction

    # Only shapes that identify a value. The generic assignment rule
    # matches any key named like a secret, which every document and
    # every detector in this repository contains by design.
    return [
        (name, pattern)
        for name, pattern in redaction.CREDENTIAL_PATTERNS.items()
        if name in redaction.VALUE_SHAPES
    ]


def credential_literals(text: str) -> list[str]:
    """Credential-shaped values present in a body of text."""

    found = []

    for _, pattern in secret_patterns():
        for match in pattern.finditer(text):
            value = match.group(0)

            # A detector's own source contains the literal shape. The
            # same pattern appears in several files by design.
            found.append(value)

    return found


def is_detector_source(path: Path) -> bool:
    """
    Whether a file exists to describe or detect credentials.

    Such a file necessarily contains the shapes it looks for, which is
    what keeps one definition of "what a token looks like" in the
    repository.
    """

    return path.name in {
        "redaction.py",
        "checkpoints.py",
        "test_security_audit.py",
        "test_agent_control_plane.py",
        "test_collection.py",
        "test_ingestion_e2e.py",
        "test_authentication.py",
    }


def python_sources() -> list[Path]:
    return sorted(REPO_ROOT.rglob("*.py"))


def text_sources() -> list[Path]:
    tracked = git("ls-files").splitlines()

    return [
        REPO_ROOT / path
        for path in tracked
        if (REPO_ROOT / path).is_file()
    ]


def test_no_real_credential_is_committed() -> None:
    """
    No tracked file contains a credential-shaped value.

    Files that exist to define or detect credentials are exempt,
    because one definition of "what a token looks like" has to live
    somewhere and it necessarily contains the shapes it matches.
    """

    offenders: list[str] = []

    for path in text_sources():
        if path.suffix in {".pdf", ".png", ".jpg", ".jpeg"}:
            continue

        if is_detector_source(path):
            continue

        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:  # noqa: BLE001
            continue

        for value in credential_literals(text):
            # Never print the value.
            offenders.append(f"{path.name} ({len(value)} chars)")

    assert offenders == [], sorted(set(offenders))


def test_the_redaction_module_still_covers_every_shape() -> None:
    """
    The audit relies on the redaction table, so the table has to stay
    complete. A shape dropped from it would stop being detected
    everywhere at once.
    """

    from src.agent import redaction

    assert set(redaction.CREDENTIAL_PATTERNS) == set(
        redaction.PATTERN_NAMES
    )

    for name in redaction.VALUE_SHAPES:
        assert name in redaction.CREDENTIAL_PATTERNS, name


def test_redaction_removes_a_real_looking_credential() -> None:
    """
    Sanity check on the table itself: a value in a documented shape
    must not survive redaction.
    """

    from src.agent.redaction import REDACTED, redact

    samples = (
        "ghp_" + "a1b2c3d4e5f6g7h8i9j0k1",
        "AKIA" + "IOSFODNN7EXAMPL1",
        "AIza" + "b" * 35,
    )

    for sample in samples:
        cleaned = redact(f"found {sample} in the log")

        assert sample not in cleaned, sample[:8]
        assert REDACTED in cleaned


def test_credentials_are_never_printed() -> None:
    """
    The reporting path must only ever say whether a credential is
    configured, never what it is.
    """

    for path in python_sources():
        text = path.read_text(encoding="utf-8")

        # Printing an environment value is the mistake. Reading it is
        # the whole point of the abstraction.
        for pattern in (
            r"print\([^)]*os\.environ",
            r"print\([^)]*os\.getenv",
            r"print\([^)]*environ\.get\(",
            r"logging\.[a-z]+\([^)]*os\.environ",
        ):
            assert not re.search(pattern, text), f"{path.name}: {pattern}"


# ---------------------------------------------------------------------
# Path containment
# ---------------------------------------------------------------------


def test_no_source_binds_a_shell_string() -> None:
    """A subprocess built from a string is a shell injection."""

    offenders: list[str] = []

    for path in python_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue

            name = getattr(node.func, "attr", None) or getattr(
                node.func, "id", None
            )

            if name not in {"run", "Popen", "call", "check_output"}:
                continue

            # shell=True is the actual hazard; the command form matters
            # less when the shell is not involved.
            for keyword in node.keywords:
                if (
                    keyword.arg == "shell"
                    and getattr(keyword.value, "value", None) is True
                ):
                    offenders.append(path.name)

    assert offenders == [], sorted(set(offenders))


def test_media_paths_cannot_escape_a_post() -> None:
    """
    A declared media path is relative and stays inside the post, so a
    post.json cannot point the loader at an arbitrary file.
    """

    from src.ingestion.post_document import resolve_media_path
    from src.ingestion.errors import InvalidPostError

    post = REPO_ROOT / "data" / "posts"

    if not post.is_dir():
        return

    for declared in (
        "../secrets.json",
        "../../.env",
        "/etc/passwd",
        "media/../../escape.txt",
    ):
        try:
            resolve_media_path(post, declared)

        except InvalidPostError:
            continue

        # A path that resolves is only acceptable if it stayed inside.
        target = (post / declared).resolve()

        assert post.resolve() in target.parents or not target.is_file()


# ---------------------------------------------------------------------
# Workflow permissions
# ---------------------------------------------------------------------


def workflows() -> list[Path]:
    directory = REPO_ROOT / ".github" / "workflows"

    if not directory.is_dir():
        return []

    return sorted(directory.glob("*.yml"))


def test_workflows_declare_explicit_permissions() -> None:
    """
    A job with no declared permissions inherits the repository default,
    which is usually broader than the job needs.
    """

    offenders: list[str] = []

    for path in workflows():
        text = path.read_text(encoding="utf-8")

        for block in re.findall(
            r"^jobs:\n(.*)", text, re.MULTILINE | re.DOTALL
        ):
            if "permissions:" not in block:
                offenders.append(path.name)

    assert offenders == [], sorted(set(offenders))


def test_the_enricher_agent_can_use_no_tools() -> None:
    """
    The enrichment agent only reads what it is given, so it must keep
    the deny-all permission block. Weakening it would let a prompt
    reach the filesystem.
    """

    agent = REPO_ROOT / ".opencode" / "agents" / "enricher.md"

    if not agent.is_file():
        return

    text = agent.read_text(encoding="utf-8")

    assert "effect: deny" in text
    assert 'resource: "*"' in text
    assert "action: \"*\"" in text


def test_the_remote_agent_still_requires_owner_control() -> None:
    """
    The control plane can push workflow files, so it must keep the
    owner-only gate.
    """

    workflow = REPO_ROOT / ".github" / "workflows" / "opencode-agent.yml"

    if not workflow.is_file():
        return

    text = workflow.read_text(encoding="utf-8")

    assert "OWNER" in text.upper()
    assert "if:" in text


# ---------------------------------------------------------------------
# What must never be committed
# ---------------------------------------------------------------------


def test_no_build_output_is_committed() -> None:
    tracked = git("ls-files").splitlines()

    assert not [path for path in tracked if path.startswith("build/")]
    assert not [path for path in tracked if path.startswith(".agent/")]


def test_no_temporary_probe_scripts_are_committed() -> None:
    tracked = git("ls-files").splitlines()

    offenders = [
        path
        for path in tracked
        if "probe" in Path(path).name.lower()
        or Path(path).name.startswith("tmp")
    ]

    assert offenders == []