"""
Running OpenCode and capturing its output.

The CLI is invoked non-interactively with `opencode run`, JSON output
format and auto-approve enabled, so the agent can read, edit and run
commands without a human at the keyboard. Explicit ``deny`` rules are
still honoured, which is what keeps ``.opencode/agents/enricher.md``
inert.

The version is pinned rather than tracking latest, so a control-plane
run cannot break because of an upstream CLI change.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


DEFAULT_VERSION = "2.0.20"

DEFAULT_MODEL = "opencode/space-bunny-free"

DEFAULT_AGENT = "remote-engineer"

DEFAULT_TIMEOUT_SECONDS = 3600

# The agent must be able to act; these are not the enrichment agent.
REQUIRED_TIMEOUT_ENV = "OPENCODE_TIMEOUT"

_EXIT_TIMEOUT = 124


class OpenCodeError(RuntimeError):
    """Raised when OpenCode cannot complete a request."""


@dataclass(frozen=True)
class OpenCodeResult:
    """The captured outcome of one OpenCode run."""

    text: str
    session_id: str | None
    exit_code: int
    stdout: str
    stderr: str

    @property
    def succeeded(self) -> bool:
        return self.exit_code == 0

    @property
    def timed_out(self) -> bool:
        return self.exit_code == _EXIT_TIMEOUT


class OpenCodeRunner:
    """Thin, testable wrapper around the OpenCode CLI."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        agent: str = DEFAULT_AGENT,
        version: str = DEFAULT_VERSION,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        working_directory: str | Path | None = None,
        extra_environment: Mapping[str, str] | None = None,
    ) -> None:
        self.model = model
        self.agent = agent
        self.version = version
        self.timeout_seconds = timeout_seconds
        self.working_directory = working_directory
        # Additions the control plane makes for the agent's own tools,
        # such as git's askpass helper when an external push credential
        # is configured. Applied after the inherited environment, and
        # before the invariants below, which always win.
        self.extra_environment = dict(extra_environment or {})

    def build_command(
        self,
        prompt: str,
        *,
        continue_session: bool = False,
    ) -> list[str]:
        """
        Build the argv for a non-interactive run.

        The prompt is passed as a single argv element, so untrusted
        task text is never re-interpreted by a shell.
        """

        command = [
            self.executable,
            "run",
            "--standalone",
            "--auto",
            "--format",
            "json",
            "--model",
            self.model,
            "--agent",
            self.agent,
        ]

        if continue_session:
            command.append("--continue")

        command.append(prompt)

        return command

    @property
    def executable(self) -> str:
        resolved = shutil.which("opencode")

        if not resolved:
            raise OpenCodeError(
                "Could not locate the opencode executable. "
                "Install it with: "
                "npm install --global @opencode/cli@"
                f"{self.version}"
            )

        return resolved

    def run(
        self,
        prompt: str,
        *,
        continue_session: bool = False,
    ) -> OpenCodeResult:
        """Execute one OpenCode run and capture everything."""

        command = self.build_command(
            prompt, continue_session=continue_session
        )

        environment = self.build_environment()

        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout_seconds,
                check=False,
                shell=False,
                cwd=str(self.working_directory)
                if self.working_directory
                else None,
                env=environment,
            )
        except FileNotFoundError as exc:
            raise OpenCodeError(
                "The opencode executable could not be started."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            stdout = _as_text(exc.stdout)
            stderr = _as_text(exc.stderr)

            return OpenCodeResult(
                text=extract_final_text(stdout),
                session_id=extract_session_id(stdout),
                exit_code=_EXIT_TIMEOUT,
                stdout=stdout,
                stderr=stderr,
            )

        return OpenCodeResult(
            text=extract_final_text(completed.stdout),
            session_id=extract_session_id(completed.stdout),
            exit_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )

    def build_environment(self) -> dict[str, str]:
        """
        Build the child environment.

        Auto-update is disabled so a runner cannot mutate its own
        toolchain mid-task, and the pinned version is recorded for
        diagnostics.
        """

        environment = dict(os.environ)
        environment.update(self.extra_environment)

        environment["OPENCODE_DISABLE_AUTOUPDATE"] = "true"
        environment.setdefault(
            "OPENCODE_DISABLE_DEFAULT_PLUGINS", "true"
        )
        environment["OPENCODE_AGENT_VERSION"] = self.version

        # The agent inherits the checkout but must not treat any
        # unrelated environment value as a task instruction.
        environment.pop("OPENCODE_PROMPT", None)

        return environment


def extract_final_text(output: str | None) -> str:
    """
    Pull the last text block out of an OpenCode JSON event stream.

    OpenCode emits newline-delimited JSON events. Anything that does
    not parse is ignored, so a stray log line cannot break parsing.
    """

    texts: list[str] = []

    for line in (output or "").splitlines():
        stripped = line.strip()

        if not stripped:
            continue

        try:
            event = json.loads(stripped)
        except json.JSONDecodeError:
            continue

        if not isinstance(event, dict):
            continue

        if event.get("type") != "text":
            continue

        part = event.get("part")

        if not isinstance(part, dict):
            continue

        text = part.get("text")

        if isinstance(text, str) and text.strip():
            texts.append(text.strip())

    if not texts:
        return ""

    return texts[-1]


def extract_session_id(output: str | None) -> str | None:
    """Find the first session id present in an event stream."""

    for line in (output or "").splitlines():
        stripped = line.strip()

        if not stripped:
            continue

        try:
            event = json.loads(stripped)
        except json.JSONDecodeError:
            continue

        if isinstance(event, dict):
            session_id = event.get("sessionID")

            if isinstance(session_id, str) and session_id:
                return session_id

    return None


def _as_text(value: object) -> str:
    if value is None:
        return ""

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    return str(value)


def install_opencode(
    version: str = DEFAULT_VERSION,
    *,
    log: object = print,
) -> None:
    """Install the pinned OpenCode CLI version."""

    package = f"@opencode/cli@{version}"

    log(f"Installing {package}")

    completed = subprocess.run(
        ["npm", "install", "--global", package],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        shell=False,
    )

    if completed.returncode != 0:
        raise OpenCodeError(
            f"Failed to install {package}: "
            f"{completed.stderr.strip()[:2000]}"
        )

    log(f"Installed {package}")


def agent_file_path(
    repository_root: str | Path,
    agent: str = DEFAULT_AGENT,
) -> Path:
    """Location of the coding agent definition."""

    return (
        Path(repository_root)
        / ".opencode"
        / "agents"
        / f"{agent}.md"
    )
