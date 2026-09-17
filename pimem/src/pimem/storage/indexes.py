from __future__ import annotations

from pimem.storage.sqlite import SQLiteStore


def rebuild(store: SQLiteStore) -> int:
    return store.rebuild_indexes()

