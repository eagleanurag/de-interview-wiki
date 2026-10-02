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
# Saved Items
# ---------------------------------------------------------------------


def saved_items_sources() -> list[Path]:
    """
    Every module the saved-items package is made of.

    The package rather than the whole pipeline, because the package is
    exclusively this phase: a capability it gained is a capability the
    phase has. The files it shares with other phases are checked
    separately, scoped to the code this phase added to them.
    """

    return sorted(
        (REPO_ROOT / "src" / "ingestion" / "saved_items").rglob("*.py")
    )


def orchestrator_sources() -> list[Path]:
    """
    Every module that decides how the model is called.

    The orchestrator, the retry loop, the run state and the failure
    classifier are the code that could turn "read a local post" into
    "send something somewhere" or "run what came back". They are checked
    as a group for the same reason the saved-items package is: a
    capability any of them gained is a capability the pipeline has.

    Held to the same boundary as a source, not to the looser one a
    general utility would get, because these modules run over
    user-supplied content and decide when to call a model about it.
    """

    paths: list[Path] = []

    paths.extend(sorted((REPO_ROOT / "src" / "enrichment").rglob("*.py")))

    for name in (
        "orchestrate.py",
        "paths.py",
        "freshness.py",
        "stages.py",
        "cli.py",
    ):
        paths.append(REPO_ROOT / "src" / "pipeline" / name)

    paths.append(REPO_ROOT / "src" / "ai" / "recovery.py")

    return paths


def imported_modules(path: Path) -> set[str]:
    """
    Every module a file imports, by name.

    Read from the tree rather than matched as text, because a file that
    *refuses* a column named "password" or "token" contains those words
    by design, and a text rule would flag the refusal as the use.
    """

    tree = ast.parse(path.read_text(encoding="utf-8"))

    modules: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name.split(".")[0])
                modules.add(alias.name)

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                modules.add(node.module.split(".")[0])
                modules.add(node.module)

    return modules


def qualified_calls(path: Path) -> set[str]:
    """
    Every call a file makes, as ``owner.name`` or a bare name.

    Qualified, because the difference matters: ``re.compile`` builds a
    pattern and ``compile`` runs a string as a program. A check that
    treated the two the same would fire on ordinary code and would
    therefore be switched off.
    """

    tree = ast.parse(path.read_text(encoding="utf-8"))

    names: set[str] = set()

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        target = node.func

        if isinstance(target, ast.Name):
            names.add(target.id)

        elif isinstance(target, ast.Attribute):
            owner = getattr(target.value, "id", None)

            names.add(
                f"{owner}.{target.attr}" if owner else target.attr
            )

    return names


def bare_calls(path: Path, only: tuple[str, ...] = ()) -> set[str]:
    """
    Unqualified calls, optionally inside some functions only.

    ``only`` scopes the check to the code this phase added to a file it
    shares with something else. ``collect_cli`` also hosts the LinkedIn
    sign-in path, which legitimately has capabilities this phase must
    not, and holding a whole file to the stricter rule would either fail
    on the wrong code or be loosened until it checked nothing.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))

    found: set[str] = set()

    def scan(node: ast.AST) -> None:
        for child in ast.walk(node):
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
            ):
                found.add(child.func.id)

    if not only:
        scan(tree)

        return found

    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in only
        ):
            scan(node)

    return found


def archive_sources() -> list[Path]:
    """
    Every module the local-archive importer is made of.

    Held to the same boundary as the saved-items package, because it
    runs on the same material: a saved-post archive is user-supplied
    content, and the fact that it arrived as one JSON file rather than a
    folder of drops does not make it any more trustworthy.
    """
    return sorted(
        (REPO_ROOT / "src" / "ingestion" / "linkedin_archive").rglob(
            "*.py"
        )
    )


def test_the_archive_importer_reaches_nothing_it_should_not() -> None:
    """
    No credential, no network, no browser, no process.

    The archive is a local folder. Every one of these capabilities
    would be the importer acquiring a way to act rather than read, and
    the whole reason the pipeline can be trusted with someone's saved
    material is that it cannot.
    """

    banned_modules = {
        "socket",
        "ssl",
        "http",
        "requests",
        "httpx",
        "aiohttp",
        "urllib.request",
        "ftplib",
        "telnetlib",
        "playwright",
        "selenium",
        "keyring",
        "netrc",
        "getpass",
        "subprocess",
        "shlex",
        "pty",
        "multiprocessing",
        "src.ingestion.credentials",
        "credentials",
        "src.agent",
    }

    offenders: list[str] = []

    for path in archive_sources():
        for name in imported_modules(path):
            if name in banned_modules or name.startswith("src.agent"):
                offenders.append(f"{path.name}: {name}")

        for name in bare_calls(path):
            if name in EXECUTION_CALLS:
                offenders.append(f"{path.name}: {name}()")

    assert offenders == [], sorted(set(offenders))


def test_the_archive_importer_reads_no_environment_variable() -> None:
    """
    Its input is a path the user gave. Reading the environment would
    give it a channel they did not choose.

    Checked as environment *access* rather than as the ``os`` import,
    because the importer legitimately uses ``os.link`` to place media
    without copying it and ``os.replace`` to write atomically. A rule
    that banned the module would have banned the right behaviour; a
    rule that bans the reads bans the thing that matters.
    """

    offenders: list[str] = []

    for path in archive_sources():
        source = path.read_text(encoding="utf-8")

        if "environ" in source or "getenv" in source:
            offenders.append(f"{path.name}: reads the environment")

        for name in imported_modules(path):
            if name in {"getpass", "dotenv", "pathlib.PureEnvPath"}:
                offenders.append(f"{path.name}: {name}")

        for name in bare_calls(path):
            if name in {"getenv", "putenv", "load_local_environment"}:
                offenders.append(f"{path.name}: {name}()")

    assert offenders == [], sorted(set(offenders))


def test_the_archive_importer_only_uses_the_filesystem_primitives_it_needs(
) -> None:
    """
    The two ``os`` calls it makes are a hard link and an atomic move.

    Named because a blanket ban on the module would have been the wrong
    rule, and this states the two that are actually justified: a second
    name for the same bytes, and a rename that cannot be observed
    half-finished.
    """

    used: set[str] = set()

    for path in archive_sources():
        for name in qualified_calls(path):
            if name.startswith("os."):
                used.add(name)

    assert used <= {"os.link", "os.replace"}, sorted(used)


def test_the_archive_session_directory_is_named_in_the_code() -> None:
    """
    A browser session is a credential in a different shape.

    Named as a constant rather than left alone, so the refusal is
    something a reader can see instead of an omission somebody has to
    notice.
    """
    from src.ingestion.linkedin_archive.archive import (
        FORBIDDEN_DIRECTORIES,
    )

    assert "chrome_session" in FORBIDDEN_DIRECTORIES


def test_no_archive_path_reaches_a_tracked_file() -> None:
    """
    The drop zone is git-ignored, so no committed file can hold one.

    Checked against the index rather than the working tree, because a
    path that is ignored now and force-added later is the case that
    would leak.
    """
    import subprocess

    tracked = subprocess.run(
        ["git", "ls-files"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    ).stdout.splitlines()

    offenders = [
        path
        for path in tracked
        if path.startswith("data/incoming/")
        or "linkedin_saved_archive" in path
        or "chrome_session" in path
    ]

    assert offenders == [], offenders


def test_no_committed_post_names_a_local_machine() -> None:
    """
    A published post may not carry the path it was imported from.

    Checked over the committed posts rather than over one fixture,
    because the leak this guards against only exists at scale: a
    single record that happened to include a path would pass every
    unit test and still put ``C:\\Users\\...`` on a public page.

    The check is on the whole document and then again on every
    path-shaped field, because a search for ``..`` alone is answered by
    the ordinary prose in a technical post -- "and..", "# ..." -- and a
    check that cries wolf on those would be a check nobody trusts.
    """

    import json

    posts = REPO_ROOT / "data" / "posts"

    if not posts.is_dir():
        pytest.skip("data/posts is absent")

    needles = (
        "C:\\Users",
        "C:/Users",
        "linkedin_saved_archive",
        "chrome_session",
        "AppData",
        str(REPO_ROOT),
    )

    offenders: list[str] = []

    for directory in sorted(posts.iterdir()):
        if not directory.is_dir():
            continue

        body = (directory / "post.json").read_text(encoding="utf-8")

        for needle in needles:
            if needle in body:
                offenders.append(f"{directory.name}: {needle!r}")

        payload = json.loads(body)

        def walk(node, where: str = "$") -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    walk(value, f"{where}.{key}")

            elif isinstance(node, list):
                for index, value in enumerate(node):
                    walk(value, f"{where}[{index}]")

            elif isinstance(node, str) and any(
                token in where.lower()
                for token in ("media", "path", "url", "dir", "file")
            ):
                # A field that is meant to hold a location, so a
                # traversal or an absolute path in one is a real
                # finding rather than a coincidence of prose.
                if (
                    "..\\" in node
                    or "../" in node
                    or node.startswith("/")
                    or node.startswith("\\\\")
                ):
                    offenders.append(f"{directory.name}: {where}={node[:60]!r}")

        walk(payload)

    assert offenders == [], sorted(set(offenders))


def test_no_committed_media_sits_outside_its_post() -> None:
    """
    Media is named relative to its own post, and only relative.

    The importer hard-links an archive's assets into a drop zone and
    the collector copies them under the post; a path that survived
    either step in absolute or traversing form would be the one place
    a reader's filesystem could be named.
    """

    posts = REPO_ROOT / "data" / "posts"

    if not posts.is_dir():
        pytest.skip("data/posts is absent")

    offenders: list[str] = []

    for directory in sorted(posts.iterdir()):
        if not directory.is_dir():
            continue

        media = directory / "media"

        if not media.is_dir():
            continue

        for item in media.rglob("*"):
            if not item.is_file():
                continue

            relative = item.relative_to(directory)

            if relative.is_absolute() or ".." in relative.parts:
                offenders.append(f"{directory.name}: {relative}")

    assert offenders == [], offenders


def test_the_orchestrator_reaches_nothing_it_should_not() -> None:
    """
    No credential, no network, no browser, no subprocess, no eval.

    The orchestrator's whole authority is deciding when to ask a local
    model about a local post. Anything that would let it reach outside
    the machine -- or run what came back -- would be a capability this
    project does not have and does not want.
    """

    banned_modules = {
        "socket",
        "ssl",
        "http",
        "requests",
        "httpx",
        "aiohttp",
        "urllib",
        "ftplib",
        "playwright",
        "selenium",
        "keyring",
        "netrc",
        "getpass",
        "subprocess",
        "pty",
        "multiprocessing",
        "ctypes",
        "pickle",
        "marshal",
        "src.ingestion.credentials",
        "src.agent",
    }

    offenders: list[str] = []

    for path in orchestrator_sources():
        for name in imported_modules(path):
            root = name.split(".")[0]

            if root in banned_modules or name.startswith("src.agent"):
                offenders.append(f"{path.name}: {name}")

        for name in bare_calls(path):
            if name in EXECUTION_CALLS:
                offenders.append(f"{path.name}: {name}()")

        for name in qualified_calls(path):
            if name.startswith("subprocess."):
                offenders.append(f"{path.name}: {name}()")

    assert offenders == [], sorted(set(offenders))


def test_the_orchestrator_reads_no_environment_variable() -> None:
    """
    Its inputs are a path and a model result.

    Reading the environment would give it a channel nobody chose, and
    in particular a place a credential could arrive from.
    """

    offenders: list[str] = []

    for path in orchestrator_sources():
        source = path.read_text(encoding="utf-8")

        if "environ" in source or "getenv" in source:
            offenders.append(f"{path.name}: reads the environment")

        for name in imported_modules(path):
            if name.split(".")[0] in {"getpass", "dotenv"}:
                offenders.append(f"{path.name}: {name}")

    assert offenders == [], sorted(offenders)


def test_the_failure_classifier_only_reads_the_failure() -> None:
    """
    It classifies an exception; it does not act on one.

    The classifier inspects an error message to decide whether to ask
    again. It must not open anything, run anything or reach the
    network, because a classifier that reached out would turn every
    provider error into an action.

    Checked from the import tree rather than from the text, because the
    module quotes provider wording as data -- "too many requests" is a
    pattern it matches, not a library it calls.
    """

    from src.ai import recovery

    source = Path(recovery.__file__)

    modules = imported_modules(source)

    for banned in (
        "subprocess",
        "socket",
        "urllib",
        "requests",
        "httpx",
        "open",
        "shutil",
    ):
        assert banned not in modules, banned

    for name in bare_calls(source):
        assert name != "open", name


def test_the_run_state_writes_only_where_it_says() -> None:
    """
    The state file is a record of a run, not of the repository.

    Its path is passed in, so nothing here decides where it goes; what
    can be checked is that the default is under the ignored build
    directory rather than beside the committed posts.
    """

    from src.pipeline.paths import RUN_STATE

    assert RUN_STATE.parts[0] == "build"


def test_committed_results_carry_no_local_path() -> None:
    """
    Every committed result becomes a published page.

    A Windows path in one is published with it, so the committed set is
    scanned rather than a fixture being trusted.
    """

    results = REPO_ROOT / "data" / "results"

    if not results.is_dir():
        pytest.skip("data/results is absent")

    needles = ("C:\\Users", "C:/Users", "AppData", "Downloads")

    offenders: list[str] = []

    for path in sorted(results.glob("cloud_worker_*.json")):
        body = path.read_text(encoding="utf-8")

        for needle in needles:
            if needle in body:
                offenders.append(f"{path.name}: {needle}")

    assert offenders == [], offenders[:10]


def test_superseded_results_are_kept_outside_the_aggregation_tree() -> None:
    """
    Archived enrichment is kept, and kept where the build cannot see it.

    The aggregator globs ``cloud_worker_*.json`` *recursively*, so an
    archive beside the live results -- even one directory down -- would
    be read as a second copy of every post and fail the run on a
    duplicate id. Filing the archive under ``data/results/`` was tried
    and was wrong for exactly that reason.
    """

    from src.aggregation.aggregator import WORKER_RESULT_GLOB

    live = REPO_ROOT / "data" / "results"

    archived = REPO_ROOT / "data" / "archive-results"

    if not archived.is_dir():
        pytest.skip("no archived results")

    assert not archived.is_relative_to(live)

    reachable = sorted(live.rglob(WORKER_RESULT_GLOB))

    names = {path.name for path in reachable}

    collisions = [
        path.name
        for path in archived.rglob(WORKER_RESULT_GLOB)
        if path.name in names
    ]

    assert collisions == [], collisions[:10]

    assert (archived / "README.md").is_file()


#: The functions this phase added to modules it shares with other work.
SAVED_ITEM_FUNCTIONS = (
    "saved_items_bundle_root",
    "run_saved_items",
    "run_saved_items_init",
    "run_saved_items_status",
    "run_saved_items_validate",
    "_preview",
    "_adopt_orphans",
    "_knowledge_counts",
    "_validation_exit",
    "saved_items_bundle_root",
)


#: Calls that would mean material from a capture was executed rather
#: than read. A module-level regex compiler is not one of them, which is
#: why these are matched as bare names.
EXECUTION_CALLS = {
    "eval",
    "exec",
    "compile",
    "__import__",
    "system",
    "popen",
    "spawn",
    "spawnl",
    "execv",
    "execve",
    "run",
    "Popen",
    "call",
    "check_output",
    "check_call",
}

PROCESS_MODULES = {
    "subprocess",
    "shlex",
    "pty",
    "commands",
    "multiprocessing",
}


def test_saved_items_needs_no_credential() -> None:
    """
    The saved-items path must not be able to reach a credential.

    A saved list is a file the user exported; there is nothing for it to
    authenticate with. A module that could read the credential store
    would mean the phase had quietly acquired a way to act as the user,
    which is the opposite of what it is for.
    """

    banned = {
        "src.ingestion.credentials",
        "credentials",
        "playwright",
        "keyring",
        "netrc",
        "getpass",
        "browser_cookie3",
    }

    offenders: list[str] = []

    for path in saved_items_sources():
        for name in imported_modules(path):
            if name in banned or name.startswith("src.agent"):
                offenders.append(f"{path.name}: {name}")

    assert offenders == [], sorted(set(offenders))


def test_saved_items_reads_no_environment_variable() -> None:
    """
    The saved-items path takes its input from files the user put in a
    directory. Reading the environment would give it a channel the user
    did not choose, and one that differs between a local run and CI.
    """

    banned = {"os", "getpass", "dotenv"}

    offenders: list[str] = []

    for path in saved_items_sources():
        for name in imported_modules(path):
            if name in banned:
                offenders.append(f"{path.name}: {name}")

    assert offenders == [], sorted(set(offenders))


def test_saved_items_opens_no_connection() -> None:
    """
    The phase exists so the pipeline does not depend on LinkedIn's web
    interface. Code that could open a connection would put that
    dependency back, and the whole point is that a change to a web
    interface cannot break this.
    """

    # ``urllib.parse`` is deliberately not here: it is string handling
    # with no way to open anything, and normalizing a URL needs it.
    # ``urllib.request`` is the module that can, and it is.
    banned = {
        "socket",
        "ssl",
        "http",
        "requests",
        "httpx",
        "aiohttp",
        "urllib.request",
        "ftplib",
        "telnetlib",
        "xmlrpc",
        "asyncio",
    }

    offenders: list[str] = []

    for path in saved_items_sources():
        for name in imported_modules(path):
            if name in banned:
                offenders.append(f"{path.name}: {name}")

    assert offenders == [], sorted(set(offenders))


def test_saved_items_needs_no_browser() -> None:
    """
    No saved-items module may drive a browser. The whole phase is local
    files; a browser dependency would mean it had started reaching for
    the site.
    """

    offenders: list[str] = []

    for path in saved_items_sources():
        for name in imported_modules(path):
            if "playwright" in name or "selenium" in name:
                offenders.append(f"{path.name}: {name}")

        for name in qualified_calls(path):
            if name.endswith(
                (".launch", ".new_page", ".new_context", ".goto")
            ):
                offenders.append(f"{path.name}: {name}()")

    assert offenders == [], sorted(set(offenders))


def test_saved_items_runs_no_command() -> None:
    """
    A capture is a file the user put in a directory. Nothing in it may
    ever be run, and nothing may be handed to a shell.
    """

    offenders: list[str] = []

    for path in saved_items_sources():
        for name in imported_modules(path):
            if name in PROCESS_MODULES:
                offenders.append(f"{path.name}: {name}")

        for name in bare_calls(path):
            if name in EXECUTION_CALLS:
                offenders.append(f"{path.name}: {name}()")

    assert offenders == [], sorted(set(offenders))


def test_the_saved_item_commands_run_nothing() -> None:
    """
    The command surface holds the same line.

    Scoped to the saved-items functions, because ``collect_cli`` also
    hosts the LinkedIn sign-in path, which legitimately has capabilities
    this phase must not have.
    """
    path = REPO_ROOT / "src" / "ingestion" / "collect_cli.py"

    offenders = sorted(
        bare_calls(path, only=SAVED_ITEM_FUNCTIONS)
        & EXECUTION_CALLS
    )

    assert offenders == []


def test_the_saved_item_commands_read_no_environment() -> None:
    path = REPO_ROOT / "src" / "ingestion" / "collect_cli.py"

    offenders = sorted(
        bare_calls(path, only=SAVED_ITEM_FUNCTIONS)
        & {"getenv", "environ", "putenv", "load_local_environment"}
    )

    assert offenders == []


def test_the_enrichment_check_runs_nothing() -> None:
    """
    Grounding reads the model's answer and the post's text. Neither is
    a program.
    """
    for path in (
        REPO_ROOT / "src" / "ai" / "grounding.py",
        REPO_ROOT / "src" / "wiki" / "saved_items.py",
    ):
        for name in imported_modules(path):
            assert name not in PROCESS_MODULES, f"{path.name}: {name}"

        for name in bare_calls(path):
            assert name not in EXECUTION_CALLS, f"{path.name}: {name}()"


def test_saved_items_never_looks_up_a_cookie_or_a_session() -> None:
    """
    Browser state is not this phase's business.

    Named explicitly because a cookie jar is the obvious thing to reach
    for when the goal is "the content the user already has", and reaching
    for it would turn a local import into a session reader.
    """

    banned = (
        "cookie",
        "storage_state",
        "browser_profile",
        "user_data_dir",
        "csrf",
    )

    offenders: list[str] = []

    for path in saved_items_sources():
        for name in imported_modules(path):
            for word in banned:
                if word in name.lower():
                    offenders.append(f"{path.name}: {name}")

        for name in qualified_calls(path):
            for word in banned:
                if word in name.lower():
                    offenders.append(f"{path.name}: {name}()")

    assert offenders == [], sorted(set(offenders))


def test_saved_html_is_parsed_as_data() -> None:
    """
    A saved page is a document, not a program.

    The phase parses HTML to read the text a page declares about itself.
    It must never run a script from that page, follow a link out of it,
    or fetch a resource it names, because a capture is untrusted input
    and a saved page is the easiest place to hide something hostile.
    """

    banned = EXECUTION_CALLS | {
        "load_module",
        "spec_from_file_location",
        "module_from_spec",
        "urlretrieve",
        "urlopen",
    }

    offenders: list[str] = []

    for path in saved_items_sources():
        for name in bare_calls(path):
            if name in banned:
                offenders.append(f"{path.name}: {name}()")

    assert offenders == [], sorted(set(offenders))


def test_saved_html_is_parsed_by_the_standard_library() -> None:
    """
    The HTML parser is the standard library's, and it parses.

    A third-party parser with a resolver, a network fetcher and a
    scripting hook attached would be able to do all three things the
    tests above forbid, while importing nothing this repository asked
    for.
    """

    allowed = {"html.parser", "html"}

    third_party = {"bs4", "lxml", "html5lib", "selectolax", "pyquery"}

    offenders: list[str] = []

    for path in saved_items_sources():
        for name in imported_modules(path):
            if name in third_party:
                offenders.append(f"{path.name}: {name}")

            if "parser" in name and name not in allowed:
                offenders.append(f"{path.name}: {name}")

    assert offenders == [], sorted(set(offenders))


def test_a_saved_page_cannot_reach_out_of_itself():
    """
    A capture folder holding a page that names a remote resource must
    not cause anything to be requested.

    Checked by construction as well as by import: the parser is handed
    the file's bytes and returns text, and nothing in the pipeline has
    anywhere to send a URL that a page supplied.
    """
    import tempfile

    from src.ingestion.saved_items.bundles import read_bundle

    with tempfile.TemporaryDirectory() as root:
        drop = Path(root) / "capture"
        drop.mkdir()

        (drop / "page.html").write_text(
            '<html><head><base href="https://example.com/">'
            '<link rel="stylesheet" href="https://example.com/x.css">'
            '<script src="https://example.com/x.js"></script>'
            '<img src="https://example.com/tracker.gif">'
            '<meta http-equiv="refresh" '
            'content="0;url=https://example.com/go">'
            "</head><body>"
            "<p>Real content that is long enough to be the body.</p>"
            '<iframe src="https://example.com/frame"></iframe>'
            "</body></html>",
            encoding="utf-8",
        )

        content = read_bundle("capture", root=root)

        # The text is read. Nothing is fetched, so the remote addresses
        # are simply not part of the result.
        assert "Real content" in content.text

        for address in ("tracker.gif", "x.js", "x.css", "example.com"):
            assert address not in content.text


def test_saved_items_reads_only_inside_the_drop_zone() -> None:
    """
    A capture may name a file, and it may only name one inside the
    directory the user pointed at.
    """
    import tempfile

    from src.ingestion.saved_items.bundles import _within

    with tempfile.TemporaryDirectory() as outside:
        escape = Path(outside) / "elsewhere.md"
        escape.write_text("private", encoding="utf-8")

        assert not _within(escape, REPO_ROOT)

        assert _within(REPO_ROOT / "src", REPO_ROOT)


def test_a_saved_item_cannot_hold_a_credential_field() -> None:
    """
    The record type itself must not offer a place to put one. Refusing
    a manifest column named for a credential is a second line of
    defence, not the first: the first is that the record has no such
    field to fill.
    """

    from src.ingestion.saved_items.model import SavedItem

    forbidden = (
        "password",
        "token",
        "cookie",
        "secret",
        "session",
        "storage_state",
        "authorization",
        "api_key",
    )

    for name in SavedItem.__dataclass_fields__:
        for word in forbidden:
            assert word not in name


def test_the_saved_items_drop_zone_is_ignored() -> None:
    for pattern, ok in ignored(
        [
            "data/incoming/",
            "data/incoming/saved-items/",
            "data/incoming/saved-items/manifest.csv",
            "data/incoming/saved-items/saved-items-manifest.json",
        ]
    ):
        assert ok, f"{pattern} is not ignored"


#: The committed synthetic fixture. A directory name, deliberately the
#: same shape as the drop zone so the reader can tell them apart.
SAVED_ITEMS_FIXTURE = "tests/fixtures/saved-items/"


def test_saved_items_state_is_not_committed() -> None:
    """
    The drop zone and the manifest are the user's material and do not
    belong in this repository's history.

    The rule names the drop zone rather than any path containing
    ``saved-items``, because a synthetic fixture lives at
    ``tests/fixtures/saved-items/`` on purpose and a rule that could not
    tell the two apart would be a rule somebody eventually switches off.
    What is refused is unchanged: the whole of ``data/incoming/``, and
    any manifest state file wherever it appears.

    The fixture is not simply allowed through. The next test proves it
    is synthetic, which is what makes allowing it safe.
    """

    tracked = git("ls-files").splitlines()

    offenders = [
        path
        for path in tracked
        if path.endswith("saved-items-manifest.json")
        or path.startswith("data/incoming/")
    ]

    assert offenders == [], sorted(offenders)


def test_the_committed_saved_items_fixture_is_synthetic() -> None:
    """
    The one committed saved-items directory is generated, not collected.

    Proved by rebuilding it and comparing, rather than by trusting a
    comment. A fixture is the only way to test this phase end to end, so
    the question is not whether to have one but whether it could be
    somebody's real export pasted in. It cannot: the builder writes it
    from constants in this file, and a rebuilt copy that differs from
    the committed one is a committed copy nobody can account for.
    """

    import filecmp
    import shutil
    import subprocess
    import tempfile

    from tests import build_fixtures

    committed = REPO_ROOT / SAVED_ITEMS_FIXTURE

    assert committed.is_dir(), "the fixture is not committed"

    names = {
        path.relative_to(committed).as_posix()
        for path in committed.rglob("*")
        if path.is_file()
    }

    assert names, "the fixture is empty"

    scratch = Path(tempfile.mkdtemp())

    try:
        original = build_fixtures.ROOT

        # The builder's root is already ``tests/fixtures``, so a
        # rebuild under a scratch root lands at ``<scratch>/saved-items``.
        build_fixtures.ROOT = scratch

        build_fixtures.build_saved_items()

        rebuilt = scratch / "saved-items"

        rebuilt_names = {
            path.relative_to(rebuilt).as_posix()
            for path in rebuilt.rglob("*")
            if path.is_file()
        }

        assert rebuilt_names == names, (
            "the committed fixture is not what the builder produces: "
            f"only rebuilt {sorted(rebuilt_names)[:3]}, "
            f"committed {sorted(names)[:3]}"
        )

        differing = [
            name
            for name in sorted(names)
            if not filecmp.cmp(
                committed / name, rebuilt / name, shallow=False
            )
        ]

        assert differing == [], differing

    finally:
        build_fixtures.ROOT = original

        shutil.rmtree(scratch, ignore_errors=True)


def test_saved_items_adds_no_read_only_violation() -> None:
    """
    The existing browser source is bounded to reading. A saved-items
    phase adds no way to act on LinkedIn, so nothing here may name an
    action verb, a private API or a challenge response.
    """

    banned = (
        "like(",
        "comment(",
        "share(",
        "repost",
        "follow(",
        "connect(",
        "sendMessage",
        "invite",
        "captcha",
        "recaptcha",
        "webdriver",
        "stealth",
        "user-agent",
    )

    offenders = []

    for path in sorted(
        (REPO_ROOT / "src" / "ingestion" / "saved_items").rglob("*.py")
    ):
        source = path.read_text(encoding="utf-8").lower()

        for name in banned:
            if name.lower() in source:
                offenders.append(f"{path.name}: {name}")

    assert offenders == []


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