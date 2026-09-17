from __future__ import annotations

from typing import Any, Dict, List, Protocol


class Reranker(Protocol):
    def rank(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Return the same candidates in relevance order without mutating storage."""


class LexicalReranker:
    def rank(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return sorted(candidates, key=lambda item: (-float(item.get("score", 0.0)), item.get("claim_id", "")))
