from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import asdict
from typing import Any, Dict, Optional

from pimem.core.models import Scope, stable_hash
from pimem.core.policy import validate_scope
from pimem.runtime import MemoryRuntime


class GenericMemoryAdapter:
    """Model- and agent-neutral adapter for observations and structured context."""

    def __init__(self, runtime: MemoryRuntime) -> None:
        self.runtime = runtime

    def ingest(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if payload.get("type", "observation") != "observation":
            raise ValueError("generic adapter only ingests observation payloads")
        content = payload.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("observation.content is required")
        scope = self._scope(payload.get("scope"))
        session_id = payload.get("session_id")
        turn_index = payload.get("turn_index")
        metadata = dict(payload.get("metadata") or {})
        if session_id is not None:
            metadata["session_id"] = session_id
        if turn_index is not None:
            metadata["turn_index"] = turn_index
        if payload.get("role") is not None:
            metadata["role"] = payload["role"]
        observation = self.runtime.observe(
            content=content,
            source_type=str(payload.get("source_type") or "generic_observation"),
            source_ref=str(payload.get("source_ref") or payload.get("id") or ("generic-" + stable_hash({
                "content": content, "scope": scope.as_dict(), "session_id": session_id, "turn_index": turn_index,
            })[:24])),
            scope=scope,
            observed_at=str(payload.get("observed_at") or datetime.now(timezone.utc).isoformat()),
            idempotency_key=payload.get("idempotency_key"),
            trace_id=payload.get("trace_id"),
            metadata=metadata,
        )
        result = asdict(observation)
        result["scope"] = observation.scope.as_dict()
        return {"accepted": True, "observation": result}

    def query(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if payload.get("type", "query") != "query":
            raise ValueError("generic adapter expects query payloads")
        query = payload.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query.query is required")
        scope = self._scope(payload.get("scope"))
        return self.runtime.prepare_context(
            query, scope, as_of=payload.get("as_of"),
            token_budget=int(payload.get("token_budget", 256)),
            limit=int(payload.get("limit", 50)), options=payload.get("options"),
            semantic_mode=str(payload.get("semantic_mode", "legacy_claim_text")),
        )

    @staticmethod
    def _scope(value: Optional[Dict[str, Any]]) -> Scope:
        if not isinstance(value, dict):
            raise ValueError("scope is required")
        validate_scope(value)
        return Scope(repository=str(value["repository"]), branch=value.get("branch"), service=value.get("service"),
                     module=value.get("module"), file=value.get("file"), session=value.get("session"), task=value.get("task"))
