"""
The ``post.json`` document layer.

``data/posts/<post_id>/post.json`` is the hand-authored entry point for
one manually captured post, and this module owns its shape: reading it,
writing it atomically, and keeping authored fields apart from the
fields the enrichment pipeline fills in later.

Two rules keep the file safe to hand-edit and safe to commit:

* a media entry declares its path relative to the post directory, so
  the repository stays portable and nothing depends on the working
  directory of whichever runner loaded it
* a declared path may never escape the post directory, so a post.json
  can never point the loader at an arbitrary file on the machine

The document written by :meth:`PostDocument.new` uses exactly the
structure the repository already ships, so an existing post can be
edited by hand, re-imported, or scaffolded from the command line
without any migration.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from src.ingestion.errors import InvalidPostError


POST_FILE_NAME = "post.json"

MEDIA_DIRECTORY_NAME = "media"

IMAGE_EXTENSIONS = frozenset(
    {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".gif",
        ".bmp",
    }
)

PDF_EXTENSIONS = frozenset(
    {
        ".pdf",
    }
)

MEDIA_TYPES = frozenset({"image", "pdf", "video", "other"})

# A post id becomes a directory name, part of a worker job id and a
# URL segment, so it is restricted to characters that are safe in all
# three. Lowercase only, so two ids can never collide on a
# case-insensitive filesystem.
POST_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")

MAX_POST_ID_LENGTH = 64

# Provenance for a post added by hand rather than by a named
# collection process.
DEFAULT_PLATFORM = "manual"

DEFAULT_DOMAIN = "Data Engineering"

#: The top-level block a source uses to record provenance the common
#: blocks do not model. The saved-items source keeps the link back to the
#: saved item here, so a post says which item it came from and how
#: complete that item was.
PROVENANCE_KEY = "saved_item"

#: What may be recorded in that block. An allowlist, for the same
#: reason the other blocks use one: a post is a committed file, and
#: nothing should reach it that was not deliberately put there.
PROVENANCE_FIELDS = (
    "saved_item_id",
    "saved_date",
    "canonical_url",
    "url_kind",
    "capture_state",
    "capture_match",
    "capture_quality",
    "capture_notes",
    "saved_notes",
    "metadata_only",
)

# The document keys the pipeline understands, in the order the
# repository writes them.
DOCUMENT_KEYS = (
    "id",
    "source",
    "original_text",
    "media",
    "ai_analysis",
    "interview_questions",
    "classification",
    "enrichment",
    PROVENANCE_KEY,
)

#: What an enrichment records about itself.
#:
#: Without this, enrichment can only be re-done wholesale, because
#: nothing says whether the analysis on disk still matches the text it
#: describes. With it, a run can tell a post whose content is unchanged
#: from one that needs the model again.
ENRICHMENT_KEYS = (
    "source_digest",
    "enricher_version",
    "enriched_at",
)

SOURCE_KEYS = (
    "platform",
    "url",
    "captured_at",
    "author",
    "published_at",
    "capture_method",
)

AI_ANALYSIS_KEYS = (
    "summary",
    "topics",
    "subtopics",
    "concepts",
    "image_descriptions",
)

CLASSIFICATION_KEYS = (
    "domain",
    "primary_topic",
    "secondary_topics",
    "interview_relevant",
)


def normalize_post_id(value: Any) -> str:
    """
    Validate a post id and return it in its canonical form.

    Post ids are used verbatim in job ids, artifact names and the
    generated site, so anything that could escape a directory or
    collide on another filesystem is rejected here rather than
    discovered later in a worker.
    """

    text = str(value or "").strip()

    if not text:
        raise InvalidPostError("post id must not be empty")

    if len(text) > MAX_POST_ID_LENGTH:
        raise InvalidPostError(
            f"post id must be at most {MAX_POST_ID_LENGTH} "
            f"characters: {text[:40]!r}"
        )

    if not POST_ID_PATTERN.match(text):
        raise InvalidPostError(
            f"post id must be lowercase and use only letters, "
            f"digits, dot, dash or underscore, and start with a "
            f"letter or digit: {text!r}"
        )

    if text.lower().endswith(".json"):
        raise InvalidPostError(
            f"post id must not end in .json, because a post "
            f"directory already contains post.json: {text!r}"
        )

    return text


def media_type_for(name: str) -> str:
    """Classify a media file by extension, the way the loader does."""

    suffix = Path(name).suffix.lower()

    if suffix in IMAGE_EXTENSIONS:
        return "image"

    if suffix in PDF_EXTENSIONS:
        return "pdf"

    return "other"


def type_matches_extension(media_type: str, name: str) -> bool:
    """
    Whether a declared media type agrees with the file extension.

    Only the types the pipeline can process are checked. ``other`` and
    ``video`` stay unconstrained, because nothing downstream reads
    them.
    """

    if media_type in {"image", "pdf"}:
        return media_type == media_type_for(name)

    return media_type in MEDIA_TYPES


def media_path_for(name: str) -> str:
    """The post-relative path used for a media file inside media/."""

    return f"{MEDIA_DIRECTORY_NAME}/{name}"


def empty_ai_analysis() -> dict[str, Any]:
    """The empty analysis block, awaiting the enrichment pipeline."""

    analysis: dict[str, Any] = {"summary": None}

    for key in AI_ANALYSIS_KEYS[1:]:
        analysis[key] = []

    return analysis


def resolve_media_path(
    post_directory: str | Path,
    relative: str,
) -> Path:
    """
    Resolve a declared media path inside its post directory.

    Rejects absolute paths and any path that climbs out of the post,
    which keeps a hand-edited post.json from reaching files elsewhere
    on the machine.
    """

    candidate = Path(relative)

    if candidate.is_absolute() or candidate.anchor:
        raise InvalidPostError(
            f"media path must be relative to the post directory: "
            f"{relative!r}"
        )

    root = Path(post_directory).resolve()
    target = (root / candidate).resolve()

    if target != root and root not in target.parents:
        raise InvalidPostError(
            f"media path must stay inside the post directory: "
            f"{relative!r}"
        )

    return target


@dataclass(frozen=True)
class MediaEntry:
    """
    One authored media declaration.

    Mirrors the ``MediaItem`` fields a human may fill in, so a declared
    entry is loadable by the worker without translation, and
    ``extracted_text`` is deliberately left to the enrichment pipeline.
    """

    type: str
    path: str
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": self.type,
            "path": self.path,
        }

        if self.description:
            payload["description"] = self.description

        return payload

    @classmethod
    def from_dict(
        cls,
        raw: Any,
        *,
        post_id: str = "",
    ) -> MediaEntry:
        label = _label(post_id)

        if not isinstance(raw, dict):
            raise InvalidPostError(
                f"{label}media entries must be JSON objects, got "
                f"{type(raw).__name__}"
            )

        media_type = raw.get("type")

        if not isinstance(media_type, str) or not media_type.strip():
            raise InvalidPostError(
                f"{label}every media entry needs a non-empty 'type'"
            )

        path = raw.get("path")

        if not isinstance(path, str) or not path.strip():
            raise InvalidPostError(
                f"{label}every media entry needs a non-empty 'path'"
            )

        description = raw.get("description") or ""

        if not isinstance(description, str):
            raise InvalidPostError(
                f"{label}media description must be a string: "
                f"{path!r}"
            )

        return cls(
            type=media_type.strip(),
            path=path.strip(),
            description=description.strip(),
        )


@dataclass
class PostDocument:
    """One ``post.json`` file, kept as the author's own JSON object."""

    post_id: str
    data: dict[str, Any]

    @classmethod
    def new(
        cls,
        post_id: str,
        *,
        text: str = "",
        platform: str = DEFAULT_PLATFORM,
        url: str | None = None,
        author: str | None = None,
        captured_at: str | None = None,
        published_at: str | None = None,
        domain: str = DEFAULT_DOMAIN,
        primary_topic: str | None = None,
        secondary_topics: tuple[str, ...] = (),
        interview_relevant: bool = False,
    ) -> PostDocument:
        """
        Build a complete, valid post document.

        The shape is exactly the one already committed under
        ``data/posts/``, so a scaffolded post needs no migration
        before the enrichment pipeline can read it.
        """

        identifier = normalize_post_id(post_id)

        document: dict[str, Any] = {
            "id": identifier,
            "source": {
                "platform": (
                    (platform or "").strip()
                    or DEFAULT_PLATFORM
                ),
                "url": (url or "").strip() or None,
                "captured_at": captured_at
                or datetime.now().astimezone().isoformat(),
                "author": (author or "").strip() or None,
                # When the source itself published the content, kept
                # exactly as rendered. Not normalised to a timestamp,
                # because a relative form like "2 days ago" cannot be
                # resolved without knowing the capture date, and a
                # wrong timestamp would be worse than the original.
                "published_at": (published_at or "").strip() or None,
            },
            "original_text": text or "",
            "media": [],
            "ai_analysis": empty_ai_analysis(),
            "interview_questions": [],
            "enrichment": {
                # Empty strings, not nulls. A post that has not been
                # enriched has no fingerprint, and the model declares
                # these as strings, so writing null here would produce a
                # document the project's own model refuses to load. That
                # is not hypothetical: an unenriched post is meant to be
                # aggregated rather than dropped, and it cannot be while
                # this block disagrees with the model.
                "source_digest": "",
                "enricher_version": "",
                "enriched_at": "",
            },
            "classification": {
                "domain": domain or DEFAULT_DOMAIN,
                "primary_topic": (primary_topic or "").strip() or None,
                "secondary_topics": [
                    topic for topic in secondary_topics if topic
                ],
                "interview_relevant": bool(interview_relevant),
            },
        }

        return cls(post_id=identifier, data=document)

    @classmethod
    def load(
        cls,
        post_directory: str | Path,
    ) -> PostDocument:
        """Read the post.json inside a post directory."""

        return cls.load_file(
            Path(post_directory) / POST_FILE_NAME
        )

    @classmethod
    def load_file(cls, path: str | Path) -> PostDocument:
        """Read one post.json file."""

        path = Path(path)

        try:
            raw = json.loads(
                path.read_text(encoding="utf-8-sig")
            )
        except json.JSONDecodeError as exc:
            raise InvalidPostError(
                f"{path} is not valid JSON: {exc}"
            ) from exc
        except OSError as exc:
            raise InvalidPostError(
                f"could not read {path}: {exc}"
            ) from exc

        if not isinstance(raw, dict):
            raise InvalidPostError(
                f"{path} must contain a JSON object, found "
                f"{type(raw).__name__}"
            )

        post_id = raw.get("id")

        if not isinstance(post_id, str) or not post_id.strip():
            raise InvalidPostError(
                f"{path} needs a non-empty string 'id'"
            )

        return cls(post_id=post_id.strip(), data=raw)

    def save(self, post_directory: str | Path) -> Path:
        """
        Write the document atomically.

        A partially written post.json would be picked up by post
        discovery and fail a whole pipeline run, so the file is
        written to a sibling temporary file and renamed into place.
        """

        directory = Path(post_directory)
        directory.mkdir(parents=True, exist_ok=True)

        path = directory / POST_FILE_NAME
        temporary = path.with_suffix(path.suffix + ".tmp")

        temporary.write_text(
            json.dumps(self.data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        temporary.replace(path)

        return path

    # -- authored fields ------------------------------------

    @property
    def source(self) -> dict[str, Any]:
        source = self.data.get("source")

        if not isinstance(source, dict):
            return {}

        return source

    @property
    def original_text(self) -> str:
        text = self.data.get("original_text")

        return text if isinstance(text, str) else ""

    def set_original_text(self, text: str) -> None:
        self.data["original_text"] = text

    # -----------------------------------------------------------------
    # Enrichment provenance
    # -----------------------------------------------------------------

    def enrichment_fingerprint(self) -> dict:
        """
        What is known about how this post was enriched.

        Values that are absent or null come back as an empty string, so
        a post written before this block existed reads the same as one
        that has simply not been enriched. Both mean the same thing, and
        returning a dict of nulls from one shape and empty strings from
        the other would make every caller handle both.
        """

        raw = self.data.get("enrichment")

        if not isinstance(raw, dict):
            return {}

        return {
            key: (raw.get(key) or "")
            if isinstance(raw.get(key), str)
            else ""
            for key in ENRICHMENT_KEYS
        }

    def mark_enriched(
        self,
        *,
        source_digest: str,
        enricher_version: str,
        enriched_at: str | None = None,
    ) -> None:
        """
        Record which content an enrichment describes.

        Stored so a later run can decide whether the analysis on disk
        still matches the text, instead of paying for the model again on
        a post that has not changed.
        """

        self.data["enrichment"] = {
            "source_digest": source_digest,
            "enricher_version": enricher_version,
            "enriched_at": enriched_at
            or datetime.now().astimezone().isoformat(),
        }

    def needs_enrichment(
        self,
        *,
        source_digest: str,
        enricher_version: str,
    ) -> bool:
        """
        Whether this post has to go back to the model.

        Re-enriched when the content changed, or when the enrichment
        logic itself changed, because an analysis produced by an older
        version no longer describes what the current version would say.
        A post that was never enriched is always enriched.
        """

        fingerprint = self.enrichment_fingerprint()

        if not fingerprint.get("source_digest"):
            return True

        if fingerprint.get("source_digest") != source_digest:
            return True

        return fingerprint.get("enricher_version") != enricher_version

    def merge_source(
        self,
        *,
        platform: str | None = None,
        url: str | None = None,
        author: str | None = None,
        captured_at: str | None = None,
        published_at: str | None = None,
        capture_method: str | None = None,
    ) -> None:
        """
        Update provenance without discarding what is already there.

        An import that does not know the author must not erase the
        author a human recorded earlier.
        """

        source = dict(self.source)

        for key, value in (
            ("platform", platform),
            ("url", url),
            ("author", author),
            ("captured_at", captured_at),
            ("published_at", published_at),
            ("capture_method", capture_method),
        ):
            if value is None:
                continue

            cleaned = value.strip() if isinstance(value, str) else ""

            source[key] = cleaned or None

        self.data["source"] = {
            key: source.get(key) for key in SOURCE_KEYS
        }

    def set_provenance(self, **fields: Any) -> None:
        """
        Record a named block of provenance at the top level.

        Used by a source that carries provenance the common blocks do
        not model. The block is rebuilt from an allowlist rather than
        merged key by key, so a field that is no longer meaningful is
        cleared instead of lingering with a stale value, and so an
        unexpected key cannot slip into a committed post.
        """

        block = self.data.get(PROVENANCE_KEY)

        if not isinstance(block, dict):
            block = {}

        for name in PROVENANCE_FIELDS:
            if name in fields:
                block[name] = fields[name]
            else:
                block.pop(name, None)

        if block:
            self.data[PROVENANCE_KEY] = block

        else:
            self.data.pop(PROVENANCE_KEY, None)

    def provenance(self) -> dict[str, Any]:
        """The recorded provenance block, or an empty one."""

        block = self.data.get(PROVENANCE_KEY)

        if isinstance(block, dict):
            return block

        return {}

    def media(self) -> list[MediaEntry]:
        """Declared media entries, in the order they are authored."""

        raw = self.data.get("media")

        if not isinstance(raw, list):
            return []

        return [
            MediaEntry.from_dict(entry, post_id=self.post_id)
            for entry in raw
        ]

    def upsert_media(self, entry: MediaEntry) -> bool:
        """
        Add a media declaration, or refresh an existing one.

        Returns True when the declaration changed. Entries are keyed
        by path, so re-importing a capture never duplicates a file.
        """

        entries = [
            MediaEntry.from_dict(item, post_id=self.post_id)
            for item in self.data.get("media", [])
            if isinstance(item, dict)
        ]

        for index, existing in enumerate(entries):
            if existing.path != entry.path:
                continue

            if entry.description and entry.description != (
                existing.description
            ):
                entries[index] = entry
                self.data["media"] = [
                    item.to_dict() for item in entries
                ]
                return True

            return False

        entries.append(entry)
        self.data["media"] = [item.to_dict() for item in entries]

        return True

    def has_enrichment(self) -> bool:
        """Whether enrichment has already filled this in."""

        analysis = self.data.get("ai_analysis")

        if isinstance(analysis, dict):
            if any(analysis.get(key) for key in AI_ANALYSIS_KEYS):
                return True

        return bool(self.data.get("interview_questions"))


def _label(post_id: str) -> str:
    return f"{post_id}: " if post_id else ""
