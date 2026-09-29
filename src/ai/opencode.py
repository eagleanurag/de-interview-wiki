from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class OpenCodeResult:
    """Normalized result returned by an OpenCode execution."""

    text: str
    data: Any
    session_id: str | None = None


class OpenCodeError(RuntimeError):
    """Raised when OpenCode cannot complete a request."""


class OpenCodeClient:
    """
    Python adapter around the OpenCode CLI.

    The adapter intentionally hides subprocess details from the rest
    of the application so the AI provider can later be executed locally
    or on a cloud worker.
    """

    def __init__(
        self,
        model: str = "opencode/space-bunny-free",
        timeout_seconds: int = 1800,
    ) -> None:
        self.model = model
        self.timeout_seconds = timeout_seconds

    def run(
        self,
        prompt: str,
        *,
        files: list[str] | None = None,
    ) -> OpenCodeResult:
        """Execute OpenCode non-interactively."""

        executable = self._resolve_executable()

        command = [
            executable,
            "run",
            "--auto",
            "--model",
            self.model,
            "--format",
            "json",
        ]

        for file_path in files or []:
            command.extend(["--file", file_path])

        command.append(prompt)

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
            )
        except FileNotFoundError as exc:
            raise OpenCodeError(
                f"OpenCode executable could not be started: {executable}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise OpenCodeError(
                f"OpenCode timed out after "
                f"{self.timeout_seconds} seconds."
            ) from exc

        if completed.returncode != 0:
            raise OpenCodeError(
                "OpenCode failed with exit code "
                f"{completed.returncode}.\n\n"
                f"STDOUT:\n{completed.stdout}\n\n"
                f"STDERR:\n{completed.stderr}"
            )

        return self._parse_output(completed.stdout)

    @staticmethod
    def _resolve_executable() -> str:
        """
        Resolve the real OpenCode executable.

        PowerShell can execute opencode.ps1, but Python subprocess
        cannot directly execute that PowerShell script.

        Resolution order:

        1. OPENCODE_EXECUTABLE environment variable.
        2. Windows executable discovered from the PowerShell shim.
        3. Normal PATH lookup.
        """

        configured = os.environ.get("OPENCODE_EXECUTABLE")

        if configured:
            path = Path(configured).expanduser()

            if path.exists():
                return str(path)

            raise OpenCodeError(
                "OPENCODE_EXECUTABLE is configured but does not exist: "
                f"{path}"
            )

        if os.name == "nt":

            commands = [
                "where.exe opencode",
                "where.exe opencode.exe",
            ]

            for command in commands:

                try:
                    result = subprocess.run(
                        command,
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        shell=True,
                        check=False,
                    )
                except OSError:
                    continue

                candidates = [
                    Path(line.strip())
                    for line in result.stdout.splitlines()
                    if line.strip()
                ]

                for candidate in candidates:

                    if candidate.suffix.lower() == ".exe":
                        if candidate.exists():
                            return str(candidate)

                    # `where.exe` may return the extension-less
                    # Node launcher. Check the actual CLI executable
                    # referenced by the installation.
                    if candidate.name.lower() == "opencode":
                        real_executable = (
                            candidate.parent
                            / "node_modules"
                            / "@opencode"
                            / "cli"
                            / "bin"
                            / "opencode.exe"
                        )

                        if real_executable.exists():
                            return str(real_executable)

        executable = shutil.which("opencode")

        if executable:
            return executable

        executable = shutil.which("opencode.exe")

        if executable:
            return executable

        raise OpenCodeError(
            "Could not locate OpenCode. "
            "Install OpenCode or set OPENCODE_EXECUTABLE."
        )

    @staticmethod
    def _parse_output(output: str) -> OpenCodeResult:
        """Parse OpenCode's JSON event stream."""

        text_events: list[str] = []
        session_id: str | None = None

        for line in output.splitlines():

            line = line.strip()

            if not line:
                continue

            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            if session_id is None:
                session_id = event.get("sessionID")

            if event.get("type") != "text":
                continue

            part = event.get("part", {})
            text = part.get("text")

            if isinstance(text, str) and text.strip():
                text_events.append(text.strip())

        if not text_events:
            raise OpenCodeError(
                "OpenCode completed successfully but no text response "
                "was found in its JSON event stream."
            )

        final_text = text_events[-1]

        return OpenCodeResult(
            text=final_text,
            data=_try_parse_json(final_text),
            session_id=session_id,
        )


def _try_parse_json(text: str) -> Any:
    """
    Parse JSON from common LLM response formats.

    Supports:
    - raw JSON
    - ```json fenced JSON
    - ``` fenced JSON
    - explanatory text surrounding a JSON object/array
    """

    cleaned = text.strip()

    # First try raw JSON.
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Remove Markdown code fences.
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()

        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        cleaned = "\n".join(lines).strip()

        # Remove optional language marker if still present.
        if cleaned.lower().startswith("json\n"):
            cleaned = cleaned[5:].strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

    # Last attempt: find the outermost JSON object/array.
    candidates = [
        (cleaned.find("{"), cleaned.rfind("}")),
        (cleaned.find("["), cleaned.rfind("]")),
    ]

    for start, end in candidates:
        if start == -1 or end == -1 or end <= start:
            continue

        candidate = cleaned[start : end + 1]

        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    # Not JSON — preserve the original response.
    return text