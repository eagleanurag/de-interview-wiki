"""
Local credential handling for an authorized source workflow.

Rules enforced here, in one place, so no caller can get them wrong:

* Credentials are read from the local environment only. A ``.env`` file
  is loaded if present, and is never written, logged, or committed.
* Nothing in this module ever returns a credential to a caller that
  intends to display it. The reporting helper returns a yes/no answer.
* A credential is never placed in a command-line argument, because
  process arguments are visible to other processes and end up in logs.
  Callers pass a reference, and the value is read at the moment of use.
* Files that contain or reference a credential are written with
  owner-only permissions where the platform supports it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


CREDENTIAL_ENVIRONMENT_VARIABLES = (
    "LINKEDIN_USERNAME",
    "LINKEDIN_PASSWORD",
)

# Files that must never be committed. Checked by the security tests.
FORBIDDEN_ARTIFACT_NAMES = (
    ".env",
    "cookies.json",
    "storage_state.json",
    "auth_state.json",
    "playwright_state.json",
)

BROWSER_PROFILE_DIRECTORY_NAMES = (
    ".browser-profile",
    "browser_profile",
    "playwright-profile",
)


class CredentialError(RuntimeError):
    """Raised when credentials are missing or mishandled."""


@dataclass(frozen=True)
class CredentialStatus:
    """
    What may be reported about credentials.

    Deliberately carries booleans and never values, so this object can
    be logged, written to a checkpoint, or printed without leaking
    anything.
    """

    username_configured: bool
    password_configured: bool

    @property
    def configured(self) -> bool:
        return self.username_configured and self.password_configured

    @property
    def incomplete(self) -> bool:
        """
        Exactly one half is present.

        This usually means a truncated ``.env``, and it is worth
        reporting precisely so the human can see which value is
        missing without revealing the other.
        """

        return self.username_configured != self.password_configured

    def describe(self) -> str:
        """The only form of this information that may be printed."""

        if self.configured:
            return "LinkedIn credentials configured: yes"

        if self.incomplete:
            missing = (
                "LINKEDIN_PASSWORD"
                if self.username_configured
                else "LINKEDIN_USERNAME"
            )
            return (
                "LinkedIn credentials configured: no "
                f"(missing {missing})"
            )

        return "LinkedIn credentials configured: no"


def load_local_environment(
    env_file: str | Path = ".env",
) -> None:
    """
    Load credentials from a local ``.env`` if one exists.

    Existing environment variables win, so an explicit export in the
    shell overrides the file.
    """

    if not Path(env_file).is_file():
        return

    from dotenv import load_dotenv

    load_dotenv(env_file, override=False)


def status(
    environ: dict[str, str] | None = None,
) -> CredentialStatus:
    """Report only whether each credential is present."""

    source = os.environ if environ is None else environ

    return CredentialStatus(
        username_configured=bool(
            source.get("LINKEDIN_USERNAME", "").strip()
        ),
        password_configured=bool(
            source.get("LINKEDIN_PASSWORD", "").strip()
        ),
    )


def require(
    environ: dict[str, str] | None = None,
) -> CredentialStatus:
    """
    Assert that credentials are usable.

    Raises with a message naming only the missing variable, never a
    value.
    """

    current = status(environ)

    if current.configured:
        return current

    if current.incomplete:
        missing = (
            "LINKEDIN_PASSWORD"
            if current.username_configured
            else "LINKEDIN_USERNAME"
        )
        raise CredentialError(
            f"Incomplete LinkedIn credentials: {missing} is not set."
        )

    raise CredentialError(
        "LinkedIn credentials are not configured. Set "
        "LINKEDIN_USERNAME and LINKEDIN_PASSWORD in the environment "
        "or in a local .env file. The file is ignored by git."
    )


def secrets_directory(root: str | Path = ".") -> Path:
    """
    Local directory for browser state.

    Kept inside ``.agent`` so a single git-ignore rule covers it.
    """

    return Path(root) / ".agent" / "secrets"


def ensure_secrets_directory(root: str | Path = ".") -> Path:
    """
    Create the secrets directory with owner-only permissions.

    Returns the path. The directory is inside the ignored ``.agent``
    tree, so nothing here can be committed by accident.
    """

    directory = secrets_directory(root)
    directory.mkdir(parents=True, exist_ok=True)
    _restrict(directory)

    return directory


def browser_profile_directory(
    root: str | Path = ".",
    name: str = "profile",
) -> Path:
    """Persistent browser profile, inside the ignored tree."""

    return ensure_secrets_directory(root) / name


def _restrict(path: Path) -> None:
    """
    Owner-only permissions where the platform supports them.

    On Windows this is effectively a no-op, which is acceptable: the
    directory is still inside the git-ignored tree and is never
    committed or uploaded.
    """

    if os.name == "nt":
        return

    try:
        path.chmod(0o700)
    except OSError:
        # A filesystem that cannot express the mode is not a reason to
        # fail the run; the ignore rules still protect the path.
        pass


def assert_no_credential_in_text(
    text: str,
    environ: dict[str, str] | None = None,
) -> None:
    """
    Guard against a credential reaching a log or a report.

    Called before anything user-visible is written. Raises if a
    configured credential value appears in the text.
    """

    current = status(environ)
    values: list[str] = []

    source = os.environ if environ is None else environ

    for name in CREDENTIAL_ENVIRONMENT_VARIABLES:
        value = source.get(name, "")

        if value and len(value) >= 8:
            values.append(value)

    del current

    for value in values:
        if value in text:
            raise CredentialError(
                "Refusing to write output containing a credential."
            )