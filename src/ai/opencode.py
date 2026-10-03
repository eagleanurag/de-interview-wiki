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

    OpenCode remains isolated behind this class so the rest of the
    application can run locally, in GitHub Actions, or on a cloud VM.
    """

    def __init__(
        self,
        model: str = "opencode/space-bunny-free",
        timeout_seconds: int = 1800,
        standalone: bool = True,
        agent: str | None = "enricher",
    ) -> None:
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.standalone = standalone
        self.agent = agent

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
        ]

        if self.standalone:
            command.append("--standalone")

        command.extend(
            [
                "--auto",
                "--model",
                self.model,
            ]
        )

        # Omitted when there is no agent, rather than passed empty.
        #
        # An agent installs a system prompt, and a system prompt wins
        # over the instruction in the request: asked for a JSON array
        # with one object per image, the enricher agent answered with a
        # single object in a schema of its own, describing both images
        # together. It had read them correctly -- only the shape came
        # from the wrong prompt. A caller whose prompt is complete says
        # so here.
        if self.agent:
            command.extend(["--agent", self.agent])

        command.extend(["--format", "json"])

        for file_path in files or []:
            command.extend(
                [
                    "--file",
                    str(file_path),
                ]
            )

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
                f"OpenCode executable could not be started: "
                f"{executable}"
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

        Windows PowerShell resolves opencode.ps1, while Python
        subprocess needs the actual executable.
        """

        configured = os.environ.get(
            "OPENCODE_EXECUTABLE"
        )

        if configured:
            path = Path(
                configured
            ).expanduser()

            if path.exists():
                return str(path)

            raise OpenCodeError(
                "OPENCODE_EXECUTABLE is configured but does not "
                f"exist: {path}"
            )

        if os.name == "nt":
            try:
                result = subprocess.run(
                    ["where.exe", "opencode"],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    check=False,
                )

                candidates = [
                    Path(line.strip())
                    for line in result.stdout.splitlines()
                    if line.strip()
                ]

                for candidate in candidates:
                    if candidate.suffix.lower() == ".exe":
                        if candidate.exists():
                            return str(candidate)

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

            except OSError:
                pass

        executable = shutil.which(
            "opencode"
        )

        if executable:
            return executable

        executable = shutil.which(
            "opencode.exe"
        )

        if executable:
            return executable

        raise OpenCodeError(
            "Could not locate OpenCode. "
            "Install OpenCode or set "
            "OPENCODE_EXECUTABLE."
        )

    @staticmethod
    def _parse_output(
        output: str,
    ) -> OpenCodeResult:
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
                session_id = event.get(
                    "sessionID"
                )

            if event.get("type") != "text":
                continue

            part = event.get(
                "part",
                {},
            )

            text = part.get("text")

            if isinstance(text, str) and text.strip():
                text_events.append(
                    text.strip()
                )

        if not text_events:
            raise OpenCodeError(
                "OpenCode completed successfully but no text "
                "response was found in its JSON event stream."
            )

        final_text = text_events[-1]

        return OpenCodeResult(
            text=final_text,
            data=_try_parse_json(final_text),
            session_id=session_id,
        )


def _try_parse_json(
    text: str,
) -> Any:
    """
    Parse JSON from common LLM response formats.

    Supports raw JSON, Markdown fenced JSON, and JSON embedded
    inside explanatory text.
    """

    cleaned = text.strip()

    try:
        return json.loads(cleaned)

    except json.JSONDecodeError:
        pass

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()

        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        cleaned = "\n".join(
            lines
        ).strip()

        if cleaned.lower().startswith("json\n"):
            cleaned = cleaned[5:].strip()

        try:
            return json.loads(cleaned)

        except json.JSONDecodeError:
            pass

    object_start = cleaned.find("{")
    object_end = cleaned.rfind("}")

    if object_start != -1 and object_end > object_start:
        candidate = cleaned[
            object_start:object_end + 1
        ]

        try:
            return json.loads(candidate)

        except json.JSONDecodeError:
            pass

    array_start = cleaned.find("[")
    array_end = cleaned.rfind("]")

    if array_start != -1 and array_end > array_start:
        candidate = cleaned[
            array_start:array_end + 1
        ]

        try:
            return json.loads(candidate)

        except json.JSONDecodeError:
            pass

    return text