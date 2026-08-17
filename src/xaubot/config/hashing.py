"""Content-addressing for configuration.

A feature set is identified by the hash of the parameters that produced it, so
changing any feature parameter writes to a new directory instead of silently
mutating an existing one. That is what makes "which config produced this
result?" answerable months later.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def canonical_json(obj: Any) -> str:
    """Serialise to JSON deterministically (sorted keys, no whitespace drift)."""
    return json.dumps(_normalise(obj), sort_keys=True, separators=(",", ":"), default=str)


def _normalise(obj: Any) -> Any:
    if isinstance(obj, BaseModel):
        return _normalise(obj.model_dump(mode="json"))
    if isinstance(obj, dict):
        return {str(k): _normalise(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_normalise(v) for v in obj]
    if isinstance(obj, Path):
        return obj.as_posix()
    if isinstance(obj, set | frozenset):
        return sorted(_normalise(v) for v in obj)
    return obj


def config_hash(obj: Any, length: int = 12) -> str:
    """Stable short hash of any config object or plain structure."""
    digest = hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()
    return digest[:length]


def version_tag(prefix: str, obj: Any, length: int = 10) -> str:
    """Human-readable version identifier, e.g. ``fs_9a3c1b7e02``."""
    return f"{prefix}_{config_hash(obj, length)}"


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of a file's contents, used to pin raw data provenance."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
