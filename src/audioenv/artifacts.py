"""Content-addressed local artifact storage."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class StoredArtifact:
    id: str
    sha256: str
    size_bytes: int
    media_type: str
    path: Path


class LocalArtifactStore:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def put_bytes(self, payload: bytes, *, media_type: str = "application/octet-stream") -> StoredArtifact:
        digest = sha256(payload).hexdigest()
        path = self.root / digest[:2] / digest[2:]
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(payload)
        return StoredArtifact(digest, digest, len(payload), media_type, path)

    def put_json(self, value: Any) -> StoredArtifact:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        return self.put_bytes(payload, media_type="application/json")

    def get_bytes(self, artifact_id: str) -> bytes:
        if len(artifact_id) != 64 or any(ch not in "0123456789abcdef" for ch in artifact_id):
            raise ValueError("artifact id must be a lowercase sha256 digest")
        path = self.root / artifact_id[:2] / artifact_id[2:]
        resolved = path.resolve()
        if not resolved.is_relative_to(self.root):
            raise ValueError("artifact path escapes the artifact root")
        return resolved.read_bytes()

    def describe(self, artifact_id: str, *, media_type: str = "application/octet-stream") -> StoredArtifact:
        payload = self.get_bytes(artifact_id)
        path = (self.root / artifact_id[:2] / artifact_id[2:]).resolve()
        return StoredArtifact(artifact_id, artifact_id, len(payload), media_type, path)

