#!/usr/bin/env python3
"""
External repository credentials for the control plane.

The job's own ``GITHUB_TOKEN`` can push ordinary source, but it is a
GitHub App installation token, and GitHub refuses any ref update that
creates or updates a file under ``.github/workflows/`` unless that app
holds the "Workflows" repository permission. ``workflows`` is a GitHub
App repository permission, not a ``GITHUB_TOKEN`` scope, so no
``permissions:`` entry in a workflow can grant it. The only
authorization that can is an externally supplied repository credential,
so the control plane accepts one and uses it for the push alone.

That credential is optional. Without it the control plane behaves
exactly as before: the built-in token pushes source changes, and a
commit that touches a workflow file is rejected by GitHub with a clear
message that the report turns into a configuration request for the
repository owner.

The credential is read from an environment variable and is never
written to a file, printed, logged, committed or passed as a command
line argument. Git authenticates through an askpass helper that reads
the environment variable at call time, so the helper itself contains no
secret and the token never lands on disk.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping


# Named so that the redaction helpers in src.agent.redaction, which
# match environment names containing "token", treat the value as a
# secret everywhere it might be echoed.
PUSH_TOKEN_ENV = "AGENT_PUSH_TOKEN"

# A boolean, not the value. A step that only needs to know *whether*
# the owner supplied a credential takes this instead, filled by the
# workflow from `${{ secrets.OPENCODE_AGENT_TOKEN != '' }}`, so the
# credential itself stays confined to the step that hands it to git.
CREDENTIAL_ARMED_ENV = "AGENT_CREDENTIAL_ARMED"

_TRUTHY = frozenset({"true", "1", "yes", "on"})

ASKPASS_ENV = "GIT_ASKPASS"
TERMINAL_PROMPT_ENV = "GIT_TERMINAL_PROMPT"

# actions/checkout authenticates the built-in GITHUB_TOKEN with this
# local config entry. Git prefers a configured extraheader over an
# askpass helper, so it has to be removed before an external
# credential can be used for the push.
GITHUB_EXTRAHEADER = "http.https://github.com/.extraheader"

ASKPASS_FILENAME = "askpass-agent.sh"

# The helper holds no secret. It only forwards the environment
# variable to git, so the credential is never persisted on the runner.
ASKPASS_SCRIPT = """#!/bin/sh
# Written by src.agent.credentials.
#
# This file deliberately contains no credential. Git calls it when it
# needs a username or password, and the value is read from the
# environment at call time and printed straight to git's pipe.
printf '%s' "${AGENT_PUSH_TOKEN}"
"""

# Both halves of the message GitHub sends when a ref update is refused
# for touching a workflow file.
_REJECTION_MARKERS = (
    "create or update workflow",
    "workflows` permission",
)

WORKFLOW_REJECTION_HINT = (
    "GitHub refused the push because the commit creates or updates a "
    "file under .github/workflows/. The GITHUB_TOKEN in this job is a "
    "GitHub App installation token without the Workflows repository "
    "permission, and no `permissions:` entry can grant it."
)

MANUAL_CONFIGURATION = (
    "Repository owner action required. Create a fine-grained personal "
    "access token scoped to this repository only (Settings > "
    "Developer settings > Personal access tokens > Fine-grained "
    "tokens) with Contents: Read and write, Workflows: Read and write, "
    "Pull requests: Read and write, Issues: Read and write, Actions: "
    "Read and write, Checks: Read and write and Commit statuses: Read "
    "and write, store it as the repository secret OPENCODE_AGENT_TOKEN, "
    "then re-run the task. The token value is never read, printed or "
    "committed by this control plane."
)


@dataclass(frozen=True)
class PushAuthentication:
    """
    How the agent's git push will authenticate.

    ``configured`` is true only when an external credential was
    supplied. The credential value is deliberately absent from this
    object: it stays in the environment, and nothing here can print it.
    """

    configured: bool
    helper_path: str = ""
    reason: str = ""

    def describe(self) -> str:
        """A one-line, secret-free description for the run log."""

        if self.configured:
            return (
                "external repository credential armed for git "
                f"({PUSH_TOKEN_ENV} via askpass); commits may update "
                "files under .github/workflows/"
            )

        if self.reason:
            return (
                "no external repository credential configured; "
                f"commits that change files under .github/workflows/ "
                f"will be refused by GitHub ({self.reason})"
            )

        return (
            f"no external repository credential configured "
            f"({PUSH_TOKEN_ENV} is not set); commits that change files "
            "under .github/workflows/ will be refused by GitHub"
        )


def resolve_push_token(
    environ: Mapping[str, str] | None = None,
) -> str:
    """
    The external push credential, or an empty string.

    The value is returned to the caller and to nobody else: it is not
    logged, not written and not stored on the returned object.
    """

    source = os.environ if environ is None else environ

    return (source.get(PUSH_TOKEN_ENV) or "").strip()


def credential_armed(environ: Mapping[str, str] | None = None) -> bool:
    """
    Whether this run has an external push credential configured.

    Read from the boolean flag rather than from the credential, so the
    prompt can state the real workflow-file boundary without any step
    that renders text ever holding the secret. An unset flag, an
    empty one and an unreadable value are all "not armed": a prompt
    must never promise a push the job cannot make.
    """

    source = os.environ if environ is None else environ

    return (source.get(CREDENTIAL_ARMED_ENV) or "").strip().lower() in (
        _TRUTHY
    )


def write_askpass_helper(git_dir: str | Path) -> Path:
    """
    Install the askpass helper inside the git directory.

    The helper lives in ``.git/``, which is never committed, and holds
    no credential.
    """

    directory = Path(git_dir)
    directory.mkdir(parents=True, exist_ok=True)

    helper = directory / ASKPASS_FILENAME
    helper.write_text(ASKPASS_SCRIPT, encoding="utf-8")
    helper.chmod(0o700)

    return helper


def git_directory(repository_root: str | Path = ".") -> Path | None:
    """
    The absolute git directory, or None outside a checkout.

    Only an absolute answer is accepted. Outside a checkout git reports
    the current directory, and writing the helper there would put it
    somewhere the agent is not pushing from.
    """

    completed = _git(
        ["rev-parse", "--absolute-git-dir"],
        cwd=str(repository_root),
    )

    resolved = completed.stdout.strip()

    if completed.returncode != 0 or not resolved:
        return None

    candidate = Path(resolved)

    if not candidate.is_absolute():
        return None

    return candidate


def remove_checkout_extraheader(
    repository_root: str | Path = ".",
) -> bool:
    """
    Drop the built-in token header from the local git configuration.

    actions/checkout stores the GITHUB_TOKEN in
    ``http.https://github.com/.extraheader``, and git prefers that
    header over any askpass helper. Removing it is what lets an
    external credential be used, and it only affects this checkout.
    """

    completed = _git(
        ["config", "--local", "--unset-all", GITHUB_EXTRAHEADER],
        cwd=str(repository_root),
        check=False,
    )

    return completed.returncode == 0


def arm_push_authentication(
    *,
    repository_root: str | Path = ".",
    environ: Mapping[str, str] | None = None,
    log: Callable[[str], None] = print,
) -> PushAuthentication:
    """
    Prepare git to push with the external credential, when there is one.

    Without a credential nothing is changed at all, so the built-in
    GITHUB_TOKEN keeps authenticating exactly as it does today.
    """

    if not resolve_push_token(environ):
        authentication = PushAuthentication(configured=False)
        log(f"PUSH_AUTHENTICATION={authentication.describe()}")
        return authentication

    directory = git_directory(repository_root)

    if directory is None:
        authentication = PushAuthentication(
            configured=False,
            reason="this run is not inside a git checkout",
        )
        log(f"PUSH_AUTHENTICATION={authentication.describe()}")
        return authentication

    try:
        helper = write_askpass_helper(directory)
    except OSError:
        authentication = PushAuthentication(
            configured=False,
            reason="the askpass helper could not be written",
        )
        log(f"PUSH_AUTHENTICATION={authentication.describe()}")
        return authentication

    remove_checkout_extraheader(repository_root)

    authentication = PushAuthentication(
        configured=True, helper_path=str(helper)
    )
    log(f"PUSH_AUTHENTICATION={authentication.describe()}")

    return authentication


def push_environment(
    authentication: PushAuthentication,
    environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """
    The environment additions the agent's git push needs.

    Without a base environment the result is only the additions, ready
    to be layered onto whatever the agent inherits. An unconfigured
    credential contributes nothing at all, so an absent credential
    cannot change how the agent authenticates.
    """

    result = dict(environment) if environment is not None else {}

    if not authentication.configured:
        return result

    result[ASKPASS_ENV] = authentication.helper_path
    result[TERMINAL_PROMPT_ENV] = "0"

    return result


def is_workflow_push_rejection(output: str | None) -> bool:
    """
    Whether output shows GitHub refusing a workflow-file push.

    The message is GitHub's, not this repository's, so it is matched
    on its two stable halves rather than on a whole sentence.
    """

    lowered = (output or "").lower()

    return all(marker in lowered for marker in _REJECTION_MARKERS)


def workflow_push_remedy() -> str:
    """
    What a human has to do about a refused workflow-file push.

    The text names the configuration to add and never a credential
    value, because this control plane has no way to know one and must
    not ask for, echo or invent one.
    """

    return f"{WORKFLOW_REJECTION_HINT} {MANUAL_CONFIGURATION}"


def _git(
    arguments: list[str],
    *,
    cwd: str | None = None,
    check: bool = False,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *arguments],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=check,
        shell=False,
        cwd=cwd,
    )
