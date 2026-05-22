"""
memory/session_store.py
=======================
Lightweight in-memory key-value store scoped to a single pipeline run.

Each pipeline run creates its own SessionStore instance, so there is no
cross-run state contamination. Agents write their validated outputs here;
the context builder reads from here to construct minimum-sufficient inputs
for downstream agents.

This is intentionally simple. For production use, replace the internal
dict with a Redis client, DynamoDB table, or similar persistent store
to enable run history, cross-session learning, and audit trails.
"""
from __future__ import annotations

import uuid
from typing import Any


class SessionStore:
    """Per-run in-memory state store."""

    def __init__(self, run_id: str | None = None) -> None:
        self.run_id: str = run_id or str(uuid.uuid4())[:8]
        self._store: dict[str, Any] = {}

    # ── Core operations ───────────────────────────────────────────────────────

    def set(self, key: str, value: Any) -> None:
        """Store a value under the given key."""
        self._store[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a value, returning default if the key does not exist."""
        return self._store.get(key, default)

    def has(self, key: str) -> bool:
        """Return True if the key exists in the store."""
        return key in self._store

    def get_all(self) -> dict[str, Any]:
        """Return a shallow copy of the entire store."""
        return dict(self._store)

    def clear(self) -> None:
        """Remove all entries from the store."""
        self._store.clear()

    def __repr__(self) -> str:
        keys = list(self._store.keys())
        return f"SessionStore(run_id={self.run_id!r}, keys={keys})"
