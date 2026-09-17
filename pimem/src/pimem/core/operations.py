from __future__ import annotations

from typing import Any, Dict, Optional

from pimem.storage.sqlite import SQLiteStore


def record(store: SQLiteStore, operation_type: str, *, target_id: Optional[str] = None,
           idempotency_key: str, trace_id: Optional[str] = None,
           details: Optional[Dict[str, Any]] = None) -> None:
    store.record_operation(operation_type, target_id, idempotency_key, trace_id, details or {})

