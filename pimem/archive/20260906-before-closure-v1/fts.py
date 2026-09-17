from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional

from pimem.core.models import Claim, Scope
from pimem.storage.sqlite import SQLiteStore

from .query_plan import QueryPlan, expand_query_tokens
from .reranker import Reranker


def _numeric_tokens(value: str) -> set[str]:
    return set(re.findall(r"(?:\d+(?:[.,]\d+)?)|(?:[$€£]\s?\d+(?:[.,]\d+)?)", value))


def _score_record(record: Dict[str, Any], query_tokens: List[str], plan: Optional[QueryPlan]) -> float:
    claim = record.get("claim")
    text = str(record.get("content") or "")
    statement = claim.statement() if claim else text
    haystack = f"{statement} {text}".lower()
    exact = sum(1 for token in query_tokens if token.lower() in haystack)
    phrase = 8.0 if plan and str(plan.query).lower() in haystack and len(str(plan.query)) > 4 else 0.0
    numbers = len(_numeric_tokens(str(plan.query) if plan else "") & _numeric_tokens(haystack)) * 4.0
    score = exact * 10.0 + phrase + numbers - float(record.get("fts_score", 0.0))
    if plan:
        if plan.primary_intent == "preference" and record.get("preference_source") == "explicit_user_preference":
            score += 30.0
        if plan.primary_intent in {"latest", "temporal"} and (record.get("event_time") or record.get("observed_at")):
            score += 3.0
        for intent in plan.secondary_intents:
            if intent in {"latest", "temporal"} and (record.get("event_time") or record.get("observed_at")):
                score += 2.0
    if claim:
        if claim.truth_status == "supported":
            score += 2.0
        if claim.use_policy == "verification_required":
            score += 1.0
    return score


def _claim_area(claim: Claim) -> str:
    if claim.truth_status == "supported" and claim.use_policy == "context_allowed":
        return "FACT"
    if claim.truth_status in {"candidate", "disputed"} and claim.use_policy == "reference_only":
        return "REFERENCE"
    return "VERIFY"


def recall(store: SQLiteStore, query: str, scope: Scope, limit: int = 10, *,
           as_of: Optional[str] = None, plan: Optional[QueryPlan] = None,
           reranker: Optional[Reranker] = None, include_observations: bool = False) -> Dict[str, Any]:
    started = time.perf_counter()
    query_tokens = expand_query_tokens(query)
    raw_records = store.search_claim_records(query, scope, limit=max(limit * 4, 20), as_of=as_of, tokens=query_tokens)
    if include_observations:
        raw_records.extend(store.search_observation_records(query, scope, limit=max(limit * 4, 20), tokens=query_tokens))

    claimed_observation_ids = {
        str(record.get("observation_id"))
        for record in raw_records
        if record.get("record_type") == "claim" and record.get("observation_id")
    }
    raw_records = [
        record for record in raw_records
        if not (
            record.get("record_type") == "observation"
            and str(record.get("observation_id")) in claimed_observation_ids
        )
    ]

    grouped: Dict[str, Dict[str, Any]] = {}
    for record in raw_records:
        record_type = record.get("record_type", "claim")
        key = f"{record_type}:{record.get('record_id') or record.get('claim_id') or record.get('observation_id')}"
        item = grouped.setdefault(key, {"record": record, "records": [], "score": float("-inf")})
        item["records"].append(record)
        item["score"] = max(item["score"], _score_record(record, query_tokens, plan))

    ranked = sorted(grouped.values(), key=lambda value: (-value["score"], str(value["record"].get("record_id", ""))))
    items: List[Dict[str, Any]] = []
    trace_candidates: List[Dict[str, Any]] = []
    for ranked_item in ranked[:max(limit, 1)]:
        primary = ranked_item["record"]
        claim = primary.get("claim")
        records = ranked_item["records"]
        source_ids = list(dict.fromkeys(str(row["observation_id"]) for row in records if row.get("observation_id")))
        metadata = primary.get("metadata") or {}
        if claim:
            item = {
                "record_type": "claim", "record_id": claim.claim_id, "claim_id": claim.claim_id,
                "subject": claim.subject, "predicate": claim.predicate, "object": claim.object,
                "statement": claim.statement(), "valid_time": {"from": claim.valid_from, "to": claim.valid_to, "precision": claim.time_precision},
                "truth_status": claim.truth_status, "lifecycle_status": claim.lifecycle_status, "use_policy": claim.use_policy,
                "scope": claim.scope.as_dict(), "source_ids": source_ids or list(claim.source_observation_ids),
                "session_id": metadata.get("session_id"), "turn_index": metadata.get("turn_index"), "role": metadata.get("role"),
                "observed_at": primary.get("observed_at"), "event_time": claim.event_time or metadata.get("event_time"),
                "text": primary.get("content", claim.statement()), "score": round(ranked_item["score"], 4),
                "area": _claim_area(claim), "evidence_kind": "claim", "state_eligible": True,
                "reason": ["fts_match", "scope_match", "scored_candidate"],
            }
        else:
            item = {
                "record_type": "observation", "record_id": primary.get("observation_id"), "claim_id": None,
                "subject": None, "predicate": None, "object": primary.get("content"), "statement": primary.get("content", ""),
                "valid_time": {"from": None, "to": None, "precision": None}, "truth_status": "unknown",
                "lifecycle_status": "active", "use_policy": "reference_only", "scope": scope.as_dict(),
                "source_ids": source_ids, "session_id": metadata.get("session_id"), "turn_index": metadata.get("turn_index"),
                "role": metadata.get("role"), "observed_at": primary.get("observed_at"),
                "event_time": metadata.get("event_time"), "text": primary.get("content", ""),
                "score": round(ranked_item["score"], 4), "area": "REFERENCE", "evidence_kind": "unstructured_observation",
                "state_eligible": False, "reason": ["observation_fts_match", "scope_match", "observation_fallback"],
            }
        items.append(item)
        trace_candidates.append({"record_id": item["record_id"], "record_type": item["record_type"],
                                 "score": item["score"], "source_ids": source_ids, "selected": True})

    if reranker and hasattr(reranker, "rank") and items:
        items = reranker.rank(query, items)[:limit]
    selected_ids = {item["record_id"] for item in items}
    for ranked_item in ranked[limit:]:
        record = ranked_item["record"]
        record_id = record.get("record_id") or record.get("claim_id") or record.get("observation_id")
        trace_candidates.append({"record_id": record_id, "record_type": record.get("record_type", "claim"),
                                 "score": round(ranked_item["score"], 4), "selected": record_id in selected_ids,
                                 "filtered_reason": "limit"})
    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    required_verification = [item["record_id"] for item in items if item.get("area") == "VERIFY"]
    result = {
        "query": query, "scope": scope.as_dict(), "items": items, "warnings": [],
        "required_verification": required_verification, "token_estimate": sum(len(item.get("statement", "").split()) for item in items),
        "latency_ms": elapsed_ms, "retrieval_count": len(items), "memory_mode": "explicit_tool",
        "retrieval_trace": {"raw_query": query, "expanded_tokens": query_tokens, "candidate_count": len(ranked),
                             "candidates": trace_candidates, "as_of": as_of, "intent": plan.primary_intent if plan else None,
                             "include_observations": include_observations},
    }
    store.record_operation("recall", None, f"recall:{scope.repository}:{query}:{time.time_ns()}", None,
                           {"query": query, "scope": scope.as_dict(), "count": len(items), "latency_ms": elapsed_ms,
                            "candidate_count": len(ranked), "include_observations": include_observations})
    return result
