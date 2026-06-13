"""Document processing tracker — content-hash idempotency.

Lets re-ingestion skip files that haven't changed. Adapted from the reference MCP
server's tracker with one deliberate upgrade: the file signature is a **sha256 of the
file's bytes**, not ``(size, mtime)``.

**Why content hash over mtime/size (the reference's choice):**

- ``mtime`` is a *lying signal*. ``git checkout``/clone rewrites it, so an mtime-based
  tracker re-embeds content that didn't change. And some edits preserve size+mtime, so
  it can *miss* a real change. A content hash answers the exact question we care about —
  "did the bytes change?" — with no false positives or false negatives.
- The cost is reading each file's bytes to hash them. For a docs corpus that's
  negligible, and ingestion is about to read the file to chunk it anyway.

State is a JSON map ``{file_path: {"sha256": ..., "chunks": N}}``, persisted between
runs next to the vector store.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List


def content_hash(file_path) -> str:
    """Return the sha256 hex digest of a file's raw bytes."""
    return hashlib.sha256(Path(file_path).read_bytes()).hexdigest()


class DocumentProcessingTracker:
    def __init__(self, tracking_file):
        self.tracking_file = Path(tracking_file)
        self.records: Dict[str, Dict] = self._load()

    def _load(self) -> Dict[str, Dict]:
        if self.tracking_file.exists():
            try:
                return json.loads(self.tracking_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return {}  # corrupt/empty tracking file → treat as fresh
        return {}

    def is_processed(self, file_path) -> bool:
        """True only if the file is tracked AND its content is unchanged."""
        rec = self.records.get(str(Path(file_path)))
        return rec is not None and rec.get("sha256") == content_hash(file_path)

    def mark_processed(self, file_path, chunk_count: int) -> None:
        self.records[str(Path(file_path))] = {
            "sha256": content_hash(file_path),
            "chunks": chunk_count,
        }

    def is_known(self, file_path) -> bool:
        """True if the file has a record at all (changed or not) — i.e. an update vs a
        brand-new file."""
        return str(Path(file_path)) in self.records

    def remove(self, file_key: str) -> None:
        self.records.pop(file_key, None)

    def tracked_keys(self) -> List[str]:
        return list(self.records.keys())

    def clear(self) -> None:
        self.records.clear()

    def save(self) -> None:
        self.tracking_file.parent.mkdir(parents=True, exist_ok=True)
        self.tracking_file.write_text(json.dumps(self.records, indent=2), encoding="utf-8")
