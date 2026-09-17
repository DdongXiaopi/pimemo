from __future__ import annotations

import re

import hashlib
import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from pimem.core.models import Claim, Scope
from pimem.core.state_machine import resolve_claim_state
from pimem.storage.sqlite import SQLiteStore

from .budget import ConservativeTokenEstimator, compact_summary, estimate_payload
from .fts import recall
from .episodic_retriever import retrieve_episodic_evidence, session_contract_rank, _contract_item_rank
from .query_plan import QueryPlan, build_query_plan, slot_matches

_slot_matches = slot_matches


def _normalized_value(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).lower()


def _compact_trace(trace: Dict[str, Any], query: str) -> Dict[str, Any]:
    candidates = trace.get("candidates") or []
    return {
        "raw_query": query,
        "expanded_tokens": trace.get("expanded_tokens", []),
        "candidate_count": trace.get("candidate_count", len(candidates)),
        "include_observations": trace.get("include_observations", False),
        "trace_ref": hashlib.sha256(json.dumps(trace, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest(),
    }


def _compact_episodic_trace(trace: Dict[str, Any]) -> Dict[str, Any]:
    compact = {
        "target_episode_count": trace.get("target_episode_count", 0),
        "candidate_count": trace.get("candidate_count", 0),
        "selected_candidate_count": trace.get("selected_candidate_count", 0),
        "target_episode_hit": bool(trace.get("target_episode_hit")),
        "answer_candidate_hit": bool(trace.get("answer_candidate_hit")),
        "answer_span_coverage": trace.get("answer_span_coverage", 0),
        "closure_truncated": bool(trace.get("closure_truncated")),
        "closure_count": trace.get("closure_count", 0),
    }
    compact["trace_ref"] = hashlib.sha256(
        json.dumps(trace, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return compact


def _item_cost(item: Dict[str, Any], level: str) -> int:
    statement_cost = len(str(item.get("statement") or item.get("text") or "").split())
    if level == "L0":
        return statement_cost
    return statement_cost + len(" ".join(str(value) for value in item.get("source_ids", [])).split()) + 2


def _conflict_groups(items: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    groups: Dict[Tuple[str, str, str], Dict[str, List[str]]] = {}
    for item in items:
        if item.get("record_type") != "claim" or not item.get("claim_id"):
            continue
        family = str(item.get("assertion_family_key") or f"{item.get('subject')}\\0{item.get('predicate')}")
        scope = json.dumps(item.get("scope") or {}, ensure_ascii=False, sort_keys=True)
        groups.setdefault((family, scope, str(item.get("event_type") or "state")), {}).setdefault(
            _normalized_value(item.get("object")), []).append(str(item.get("claim_id")))
    conflicts: Dict[str, List[str]] = {}
    for variants in groups.values():
        if len(variants) < 2:
            continue
        ids = [claim_id for values in variants.values() for claim_id in values]
        for claim_id in ids:
            conflicts[claim_id] = ids
    return conflicts


def _render_item(item: Dict[str, Any], level: str, conflict_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    rendered = dict(item)
    rendered["level"] = level
    rendered["token_estimate"] = _item_cost(item, level)
    rendered["reasons"] = list(item.get("reason", []))
    if conflict_ids:
        rendered["area"] = "VERIFY"
        rendered["reasons"].append("conflict")
        rendered["conflict_with"] = [value for value in conflict_ids if value != item.get("claim_id")]
        rendered["caveat"] = "Conflicting claims require current-state verification."
    elif level == "L1":
        rendered["caveat"] = "Source observation ids are provided for traceability; verify mutable project state."
    if level == "L1":
        rendered["evidence"] = [{"observation_id": source_id, "relation": "supports"}
                                 for source_id in item.get("source_ids", [])]
    return rendered


def _item_from_claim(claim: Claim, store: SQLiteStore) -> Dict[str, Any]:
    source_ids = list(claim.source_observation_ids)
    content = claim.statement()
    metadata: Dict[str, Any] = {}
    observed_at = claim.updated_at
    event_time = claim.event_time
    text = content
    for source_id in source_ids[:8]:
        try:
            observation = store.get_observation(source_id)
        except KeyError:
            continue
        metadata = observation.metadata or {}
        observed_at = observation.observed_at
        event_time = event_time or metadata.get("event_time")
        text = observation.content
        break
    return {
        "record_type": "claim", "record_id": claim.claim_id, "claim_id": claim.claim_id,
        "subject": claim.subject, "predicate": claim.predicate, "object": claim.object,
        "statement": content, "valid_time": {"from": claim.valid_from, "to": claim.valid_to, "precision": claim.time_precision},
        "truth_status": claim.truth_status, "lifecycle_status": claim.lifecycle_status, "use_policy": claim.use_policy,
        "scope": claim.scope.as_dict(), "source_ids": source_ids, "session_id": metadata.get("session_id"),
        "turn_index": metadata.get("turn_index"), "role": metadata.get("role"), "observed_at": observed_at,
        "event_time": event_time, "text": text, "area": "FACT" if claim.use_policy == "context_allowed" else "REFERENCE",
        "evidence_kind": "claim", "state_eligible": True, "assertion_family_key": claim.assertion_family_key,
        "event_type": claim.event_type, "reason": ["state_closure"],
    }


def _preference_source(item: Dict[str, Any]) -> str:
    text = f"{item.get('text', '')} {item.get('statement', '')}".lower()
    role = str(item.get("role") or "").lower()
    predicate = str(item.get("predicate") or "")
    if role == "assistant":
        return "assistant_suggestion"
    if role == "user" and any(token in text for token in ("prefer", "like", "love", "hate", "dislike", "喜欢", "偏好", "不喜欢")):
        return "explicit_user_preference"
    if predicate == "user_code_preference":
        return "explicit_user_preference"
    if role == "user":
        return "historical_behavior"
    return "assistant_suggestion" if role == "assistant" else "inferred_preference"


def _record_identity(item: Dict[str, Any]) -> str:
    return f"{item.get('record_type', 'claim')}:{item.get('record_id') or item.get('claim_id') or item.get('observation_id')}"


def _build_slots(plan: QueryPlan, items: List[Dict[str, Any]], omitted: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    slots = []
    all_items = items + omitted
    for raw_slot in plan.requested_slots:
        matched_claims = [item.get("claim_id") for item in items if item.get("claim_id") and slot_matches(raw_slot, item)]
        matched_obs = [item.get("observation_id") for item in items if item.get("observation_id") and slot_matches(raw_slot, item)]
        related = [item.get("record_id") for item in all_items if item.get("record_id") and slot_matches(raw_slot, item)]
        exact = bool(matched_claims or matched_obs)
        status = "known" if exact else "unknown"
        if not exact and related:
            status = "partially_known"
        slots.append({**raw_slot, "object": raw_slot.get("entity"), "matched_claim_ids": matched_claims, "matched_observation_ids": matched_obs,
                      "related_evidence_ids": related, "status": status,
                      "reason": "exact entity/field evidence" if exact else "no exact requested-object evidence"})
    return slots


def _sort_items(items: List[Dict[str, Any]], plan: QueryPlan) -> List[Dict[str, Any]]:
    def timestamp(item: Dict[str, Any]) -> str:
        return str(item.get("event_time") or item.get("observed_at") or "")
    def score(item: Dict[str, Any]) -> Tuple[Any, ...]:
        preference = 0
        if "preference" in {plan.primary_intent, *plan.secondary_intents}:
            preference = {"explicit_user_preference": 0, "conditional_user_preference": 1,
                          "historical_behavior": 2, "assistant_suggestion": 3, "inferred_preference": 4}.get(
                              str(item.get("preference_source") or ""), 5)
        return (preference, -float(item.get("score", 0.0)), timestamp(item), str(item.get("record_id", "")))
    result = sorted(items, key=score)
    if plan.primary_intent in {"latest", "temporal"}:
        result.sort(key=lambda item: (timestamp(item) == "", timestamp(item)), reverse=plan.primary_intent == "latest")
    return result


def _aggregation_hints(plan: QueryPlan, items: List[Dict[str, Any]]) -> Dict[str, Any]:
    operator = (plan.aggregation or {}).get("operator")
    groups: Dict[str, Dict[str, Any]] = {}
    for item in items:
        object_key = _normalized_value(item.get("object")) if item.get("object") is not None else "unknown"
        entity_key = str(item.get("subject") or item.get("session_id") or "unknown")
        key = entity_key + ":" + object_key if operator in {"compare", "sum"} else object_key
        group = groups.setdefault(key, {"key": key, "object": item.get("object"), "claim_ids": [], "observation_ids": [], "source_values": []})
        if item.get("claim_id"):
            group["claim_ids"].append(item["claim_id"])
        group["observation_ids"].extend(item.get("source_ids", []))
        group["source_values"].append({"claim_id": item.get("claim_id"), "value": item.get("object"), "source_ids": item.get("source_ids", [])})
    return {"operator": operator, "dedupe_key": (plan.aggregation or {}).get("dedupe_key"),
            "groups": list(groups.values()), "aggregation_incomplete": any(item.get("status") != "known" for item in plan.requested_slots),
            "requires_external_calculation": operator in {"count", "sum", "compare"}}


def _evidence_priority(item: Dict[str, Any]) -> Tuple[int, float, float, str]:
    is_candidate = int(item.get("record_type") == "answer_candidate")
    direct_entity_hits = int(item.get("entity_hits") or 0)
    requested_entity_hits = int(item.get("requested_entity_hits") or 0)
    direct_temporal_hits = int(item.get("temporal_hits") or 0)
    direct_query_hits = int(item.get("answer_overlap") or item.get("query_overlap") or 0)
    currency_signal = int(bool(item.get("currency_relevance"))) if item.get("currency_request") else 0
    paired_question = int(bool(item.get("question_turn_id")))
    historical_user_answer = int(bool(item.get("role") == "user" and item.get("speech_act") == "user_statement"
                                     and item.get("question_turn_id")))
    requested_entity = int(item.get("question_entity_hits") or 0)
    question_overlap = int(item.get("question_overlap") or 0)
    source_span = int(bool(item.get("source_span") or item.get("answer_span")))
    speech_act = int(item.get("speech_act") in {"assistant_answer", "assistant_recommendation", "user_statement"})
    score = float(item.get("score", 0.0))
    shape_signal = int(bool(item.get("structured_facts") or item.get("answer_span")))
    relation_hits = int(item.get("relation_hits") or 0)
    literal_hits = int(item.get("literal_hits") or 0)
    literal_full_hits = int(item.get("literal_full_hits") or 0)
    literal_max_length = int(item.get("literal_max_length") or 0)
    numeric_hits = int(item.get("numeric_hits") or 0)
    typed_relation = int(item.get("typed_relation_score") or 0)
    typed_field = int(bool(item.get("typed_field_match")))
    ordinal = int(bool(item.get("ordinal_match")))
    answer_units = item.get("answer_units") or []
    source_span_count = len(item.get("source_span") or [])
    list_or_table = int(any(unit.get("kind") in {"list_item", "table_coordinate", "typed_sentence", "quoted_span"}
                            for unit in answer_units))
    best_unit_score = max((int(unit.get("unit_score") or 0) for unit in answer_units), default=0)
    attribute_anchor = int(any(unit.get("kind") in {"attribute_sentence", "typed_sentence"}
                               for unit in answer_units))
    text = str(item.get("text") or item.get("statement") or "")
    direct_recommendation = int(bool(re.search(
        r"\b(?:i|we)\s+(?:would\s+)?recommend(?:ed|s)?\b", text.lower())))
    # Hard-budget victim selection still prefers typed/structured answerability
    # over noisy lexical score. The compiler preserves retriever order when
    # choosing initial anchors, so this priority is only a budget guardrail.
    quality = (
        typed_relation * 260
        + typed_field * 220
        + requested_entity_hits * 190
        + relation_hits * 150
        + ordinal * 130
        + literal_full_hits * 260
        + min(literal_max_length, 120)
        + literal_hits * 120
        + requested_entity * 120
        + list_or_table * 70
        + attribute_anchor * 180
        + min(best_unit_score, 1200)
        + direct_recommendation * 600
        + numeric_hits * 55
        + direct_entity_hits * 40
        + direct_temporal_hits * 32
        + question_overlap * 95
        + paired_question * 30
        + historical_user_answer * 220
        + direct_query_hits * 3
        + currency_signal * 48
        + shape_signal * 20
        + min(source_span_count, 8) * 7
        + source_span * 8
        + speech_act * 5
    )
    return (is_candidate, quality, score, str(item.get("record_id", "")))


def _choose_budget_victim(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not items:
        raise ValueError("cannot choose a budget victim from an empty list")
    return min(items, key=_evidence_priority)


def _budget_removable(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ordinary = [item for item in items if item.get("record_type") != "answer_candidate"]
    if ordinary:
        return ordinary
    candidates = [item for item in items if item.get("record_type") == "answer_candidate"]
    if len(candidates) > 1:
        return candidates
    return []


def _compact_item(item: Dict[str, Any]) -> Dict[str, Any]:
    return {"record_id": item.get("record_id"), "record_type": item.get("record_type"), "area": item.get("area"),
             "score": item.get("score"), "truth_status": item.get("truth_status"), "source_ids": item.get("source_ids", [])}


def _compact_omitted_item(item: Dict[str, Any]) -> Dict[str, Any]:
    return {key: item.get(key) for key in (
        "record_id", "claim_id", "observation_id", "record_type", "reason", "area",
    ) if item.get(key) is not None}


def _compact_episodic_candidate(item: Dict[str, Any]) -> Dict[str, Any]:
    compact = {key: item.get(key) for key in (
        "record_type", "record_id", "candidate_id", "observation_id", "source_ids", "episode_id", "session_id",
        "turn_id", "turn_index", "role", "speech_act", "answer_shape", "source_span", "confidence",
        "answer_span", "area", "evidence_kind", "state_eligible", "score")}
    compact["text"] = str(item.get("text") or item.get("statement") or "")
    compact["evidence_text"] = _answer_evidence_text(item)
    compact["object"] = compact["text"]
    compact["provenance"] = item.get("provenance", {})
    for key in ("question_turn_id", "question_turn_index", "question_text", "question_overlap", "question_entity_hits",
                 "query_overlap", "answer_overlap", "entity_hits", "requested_entity_hits", "turn_entity_hits", "temporal_hits", "currency_hits",
                 "currency_relevance", "currency_request", "relation_hits", "literal_hits", "numeric_hits",
                 "structured_facts", "facets", "answer_units"):
        if key in item:
            compact[key] = item[key]
    return compact


def _select_hard_episode_anchors(items: List[Dict[str, Any]], limit: int,
                                 plan: Optional[QueryPlan] = None,
                                 episode_bundles: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """Select anchors by required evidence coverage before optional relevance."""
    if limit <= 0:
        return []

    def field_match(item: Dict[str, Any]) -> int:
        fields = set(plan.fields if plan else [])
        kinds = {str(unit.get("kind") or "") for unit in (item.get("answer_units") or [])}
        kinds.update(str(facet.get("kind") or "") for facet in (item.get("facets") or []))
        if not fields:
            return 0
        field_kinds = {
            "phone_number": {"phone_number"},
            "amount": {"numeric_value", "percentage", "ratio"},
            "percentage": {"percentage", "numeric_value"},
            "ratio": {"ratio", "numeric_value"},
            "year": {"year", "numeric_value"},
            "date": {"date", "year", "numeric_value"},
            "duration": {"numeric_value"},
            "count": {"numeric_value"},
            "attribute": {"attribute_value", "attribute_sentence", "typed_sentence", "quoted_span", "list_item"},
            "value": {"attribute_value", "attribute_sentence", "typed_sentence", "quoted_span", "list_item"},
        }
        return int(any(kinds & field_kinds.get(field, {field}) for field in fields))

    def literal_match(item: Dict[str, Any]) -> int:
        if not plan or not plan.literals:
            return 0
        kinds = {str(unit.get("kind") or "") for unit in (item.get("answer_units") or [])}
        return int(item.get("literal_full_hits") or 0) > 0 or "special_literal" in kinds
    def priority(item: Dict[str, Any]) -> Tuple[int, ...]:
        return _contract_item_rank(item, plan)

    def coverage(item: Dict[str, Any]) -> Tuple[int, int, int]:
        slots = sum(1 for slot in (plan.requested_slots if plan else []) if _slot_matches(slot, item))
        structured = sum(1 for unit in (item.get("answer_units") or []) if unit.get("kind") in {
            "table_coordinate", "table_cell", "table_row", "list_item", "quoted_span",
            "typed_sentence", "attribute_sentence",
        })
        literals = int(item.get("literal_hits") or 0) + int(item.get("numeric_hits") or 0)
        return slots, field_match(item), literal_match(item), structured, literals

    if any(item.get("record_type") != "answer_candidate" for item in items):
        grouped: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {}
        for index, item in enumerate(items):
            grouped.setdefault(str(item.get("session_id") or ""), []).append((index, item))
        selected: List[Dict[str, Any]] = []
        selected_ids = set()
        for session_id in grouped:
            session_items = sorted(
                grouped[session_id],
                key=lambda pair: (coverage(pair[1]), priority(pair[1])),
                reverse=True,
            )
            if not session_items:
                continue
            item = session_items[0][1]
            record_id = str(item.get("record_id") or "")
            if record_id in selected_ids:
                continue
            selected.append(item)
            selected_ids.add(record_id)
            if len(selected) >= limit:
                return selected
        remaining = sorted(
            ((index, item) for index, item in enumerate(items)
             if str(item.get("record_id") or "") not in selected_ids),
            key=lambda pair: (coverage(pair[1]), priority(pair[1]), -pair[0]),
            reverse=True,
        )
        for _, item in remaining:
            record_id = str(item.get("record_id") or "")
            if record_id in selected_ids:
                continue
            selected.append(item)
            selected_ids.add(record_id)
            if len(selected) >= limit:
                break
        return selected

    bundle_by_session = {
        str(bundle.get("session_id") or ""): bundle
        for bundle in (episode_bundles or [])
        if bundle.get("session_id")
    }

    def bundle_primary_ids(bundle: Dict[str, Any]) -> set[str]:
        return {
            str(record_id)
            for record_id in (
                bundle.get("primary_answer_candidate_ids")
                or bundle.get("anchor_candidates")
                or []
            )
            if record_id
        }

    grouped: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {}
    for index, item in enumerate(items):
        grouped.setdefault(str(item.get("session_id") or ""), []).append((index, item))

    def item_key(pair: Tuple[int, Dict[str, Any]]) -> Tuple[Any, ...]:
        index, item = pair
        bundle = bundle_by_session.get(str(item.get("session_id") or ""), {})
        primary_ids = bundle_primary_ids(bundle)
        in_answer_closure = int(str(item.get("record_id") or "") in primary_ids)
        primary_turn_ids = set(bundle.get("primary_answer_turn_ids") or bundle.get("answer_turn_ids") or [])
        in_primary_turn = int(str(item.get("turn_id") or "") in primary_turn_ids)
        return in_answer_closure, in_primary_turn, priority(item), coverage(item), -index

    session_blocks: List[Tuple[Tuple[Any, ...], List[Dict[str, Any]]]] = []
    for session_id, session_pairs in grouped.items():
        bundle = bundle_by_session.get(session_id, {})
        primary_ids = bundle_primary_ids(bundle)
        primary_pairs = [pair for pair in session_pairs
                         if str(pair[1].get("record_id") or "") in primary_ids]
        if primary_pairs:
            block_pairs = sorted(primary_pairs, key=item_key, reverse=True)
        else:
            block_pairs = sorted(session_pairs, key=item_key, reverse=True)
        block = [item for _, item in block_pairs]
        best = max(block, key=lambda item: (priority(item), coverage(item)))
        support_items = [item for _, item in session_pairs]
        historical_signal = max(
            (
                int(
                    bool(plan and plan.speech_act_target == "historical_answer")
                    and item.get("role") == "user"
                    and item.get("speech_act") == "user_statement"
                )
                for item in support_items
            ),
            default=0,
        )
        support_key = (
            max((int(item.get("requested_entity_hits") or 0) for item in support_items), default=0),
            max((int(item.get("question_entity_hits") or 0) for item in support_items), default=0),
            max((int(item.get("typed_relation_score") or 0) for item in support_items), default=0),
            max((int(item.get("relation_hits") or 0) for item in support_items), default=0),
            max((field_match(item) for item in support_items), default=0),
            max((literal_match(item) for item in support_items), default=0),
            max((int(item.get("question_overlap") or 0) for item in support_items), default=0),
            max((int(item.get("answer_anchor_hits") or 0) for item in support_items), default=0),
        )
        contract_key = session_contract_rank(support_items, plan, primary_ids)
        first_index = min((index for index, _ in session_pairs), default=0)
        block_slots = {
            str(slot.get("slot_id"))
            for item in block
            for slot in (plan.requested_slots if plan else [])
            if slot.get("slot_id") and _slot_matches(slot, item)
        }
        session_key = (
            int(bool(primary_pairs)),
            int(bundle.get("coverage", {}).get("primary_answer", False)),
            contract_key,
            len(block_slots),
            historical_signal,
            support_key,
            priority(best),
            coverage(best),
            -first_index,
        )
        session_blocks.append((session_key, block, block_slots))

    selected: List[Dict[str, Any]] = []
    selected_ids = set()
    requested_slots = {
        str(slot.get("slot_id"))
        for slot in (plan.requested_slots if plan else [])
        if slot.get("slot_id")
    }
    covered_slots = set()
    ranked_blocks = sorted(session_blocks, key=lambda value: value[0], reverse=True)
    for _, block, block_slots in ranked_blocks:
        if selected and requested_slots and not (block_slots - covered_slots):
            continue
        for item in block:
            record_id = str(item.get("record_id") or "")
            if record_id in selected_ids:
                continue
            selected.append(item)
            selected_ids.add(record_id)
            covered_slots.update(
                str(slot.get("slot_id"))
                for slot in (plan.requested_slots if plan else [])
                if slot.get("slot_id") and _slot_matches(slot, item)
            )
            if len(selected) >= limit:
                return selected
        if not requested_slots or requested_slots <= covered_slots:
            return selected
    return selected


def _compact_hard_episodic_candidate(item: Dict[str, Any]) -> Dict[str, Any]:
    compact = {key: item.get(key) for key in (
        "record_type", "record_id", "candidate_id", "observation_id", "source_ids", "episode_id", "session_id",
        "turn_id", "turn_index", "role", "speech_act", "answer_shape", "confidence", "answer_span",
        "area", "evidence_kind", "state_eligible", "score", "question_turn_id", "question_turn_index",
        "answer_units", "facets", "structured_facts", "question_text", "question_overlap", "question_entity_hits",
    )}
    answer_text = _answer_evidence_text(item)
    source_text = str(item.get("text") or item.get("statement") or "")
    if answer_text and answer_text not in source_text:
        source_text = answer_text + "\n" + source_text
    compact["text"] = source_text[:900]
    compact["evidence_text"] = answer_text
    compact["object"] = compact["text"]
    source_spans = list(item.get("source_span") or [])
    priority_kinds = {"table_coordinate", "table_cell", "table_header", "table_row",
                      "list_item", "attribute_sentence", "typed_sentence", "quoted_span", "answer_span"}
    prioritized = [span for span in source_spans if span.get("kind") in priority_kinds]
    remainder = [span for span in source_spans if span not in prioritized]
    compact["source_span"] = (prioritized + remainder)[-12:]
    compact["provenance"] = item.get("provenance", {})
    return compact


def _answer_evidence_text(item: Dict[str, Any]) -> str:
    """Render the smallest answer-bearing text before falling back to turn text."""
    priority = {
        "table_coordinate": 0, "table_cell": 1, "table_row": 2,
        "list_item": 3, "quoted_span": 4, "attribute_value": 5,
        "attribute_sentence": 6, "typed_sentence": 7, "sentence": 8,
        "answer_span": 9,
    }
    values = []
    for unit in sorted(item.get("answer_units") or [], key=lambda value: priority.get(value.get("kind"), 99)):
        source_span = unit.get("source_span") or {}
        value = unit.get("value") or unit.get("text") or source_span.get("text")
        if value and str(value).strip() not in values:
            values.append(str(value).strip())
    if values:
        answer_span_text = str((item.get("answer_span") or {}).get("text") or "").strip()
        unit_text = "\n".join(values).casefold()
        if answer_span_text and answer_span_text.casefold() != unit_text:
            values.append(answer_span_text)
        return "\n".join(values)
    answer_span = item.get("answer_span") or {}
    if answer_span.get("text"):
        return str(answer_span["text"])
    for span in item.get("source_span") or []:
        if span.get("kind") in priority and span.get("text"):
            return str(span["text"])
    return str(item.get("candidate_text") or item.get("text") or item.get("statement") or "")


def _compact_selected_item(item: Dict[str, Any]) -> Dict[str, Any]:
    if item.get("record_type") == "answer_candidate":
        return _compact_episodic_candidate(item)
    compact = dict(item)
    compact["statement"] = str(item.get("statement") or item.get("text") or "")[:600]
    compact["text"] = str(item.get("text") or item.get("statement") or "")[:600]
    return compact


def _compact_hard_selected_item(item: Dict[str, Any]) -> Dict[str, Any]:
    if item.get("record_type") != "answer_candidate":
        return _compact_selected_item(item)
    return {key: item.get(key) for key in (
        "record_id", "record_type", "source_ids", "observation_id", "session_id", "turn_id",
        "turn_index", "role", "speech_act", "answer_shape", "area", "score",
    )}


def _compact_episode_bundle(bundle: Dict[str, Any], *, include_answer_candidates: bool = True,
                            include_turn_text: bool = True) -> Dict[str, Any]:
    compact = {key: bundle.get(key) for key in (
        "bundle_id", "episode_id", "session_id", "anchor_candidates", "question_turn_ids", "answer_turn_ids",
        "primary_answer_candidate_ids", "primary_answer_turn_ids", "question_pair_ids",
        "required_answer_units", "optional_related_turn_ids",
        "related_turn_ids", "chronological_order", "role_sequence", "source_observations", "coverage",
        "confidence", "reasons", "structured_facts", "facets", "answer_units", "source_spans")}
    if include_turn_text:
        for key in ("question_turns", "answer_turns", "related_turns"):
            compact[key] = bundle.get(key, [])
    if include_answer_candidates:
        compact["answer_candidates"] = [_compact_episodic_candidate(item) for item in bundle.get("answer_candidates", [])]
    return compact


def _pack_candidate(pack_base: Dict[str, Any], selected: List[Dict[str, Any]], omitted: List[Dict[str, Any]],
                    conflicts: List[Dict[str, Any]], slots: List[Dict[str, Any]], estimator: ConservativeTokenEstimator) -> Dict[str, Any]:
    memory_view = [{"record_id": item.get("record_id"), "statement": str(item.get("statement") or item.get("text") or "")[:240], "area": item.get("area"),
                    "truth_status": item.get("truth_status"), "state_eligible": item.get("state_eligible", False)} for item in selected]
    evidence_view = [{"record_id": item.get("record_id"), "record_type": item.get("record_type"),
                      "observation_ids": item.get("source_ids", []), "session_id": item.get("session_id"),
                      "turn_index": item.get("turn_index"), "observed_at": item.get("observed_at"),
                      "event_time": item.get("event_time"), "evidence_trust": "untrusted_historical_data"} for item in selected]
    pack = dict(pack_base)
    bundle_map: Dict[str, Dict[str, Any]] = {}
    for item in selected:
        session_id = item.get("session_id") or "unknown-session"
        bundle = bundle_map.setdefault(session_id, {"session_id": session_id, "record_ids": [], "claim_ids": [], "observation_ids": []})
        bundle["record_ids"].append(item.get("record_id"))
        if item.get("claim_id"):
            bundle["claim_ids"].append(item["claim_id"])
        bundle["observation_ids"].extend(item.get("source_ids", []))
    episode_bundles = [dict(bundle) for bundle in (pack.get("episode_bundles") or [])]
    for bundle in episode_bundles:
        if "anchor_candidates" in bundle:
            bundle["anchor_candidates"] = list(bundle.get("anchor_candidates") or [])
        if "answer_candidates" in bundle:
            bundle["answer_candidates"] = list(bundle.get("answer_candidates") or [])
    if pack.get("semantic_mode") == "legacy_claim_text":
        selected_view = selected
    elif pack.get("budget_mode") == "hard":
        selected_view = [_compact_hard_selected_item(item) for item in selected]
    else:
        selected_view = [_compact_selected_item(item) for item in selected]
    compact_candidate = _compact_hard_episodic_candidate if pack.get("budget_mode") == "hard" else _compact_episodic_candidate
    episodic_view = [compact_candidate(item) for item in selected if item.get("record_type") == "answer_candidate"]
    selected_sessions = {item.get("session_id") for item in selected}
    episode_bundles = [bundle for bundle in episode_bundles if bundle.get("session_id") in selected_sessions]
    selected_candidate_ids = {item.get("record_id") for item in selected}
    for bundle in episode_bundles:
        bundle["anchor_candidates"] = [candidate_id for candidate_id in bundle.get("anchor_candidates", [])
                                        if candidate_id in selected_candidate_ids]
        if "answer_candidates" in bundle:
            bundle["answer_candidates"] = [item for item in bundle["answer_candidates"]
                                            if item.get("record_id") in selected_candidate_ids]
    omitted_view = omitted if pack.get("budget_mode") != "hard" else []
    pack.update({"memory_view": memory_view, "evidence_view": evidence_view, "selected": selected_view,
                 "items": selected if pack.get("semantic_mode") == "legacy_claim_text" else [_compact_item(item) for item in selected], "omitted": omitted_view, "conflicts": conflicts,
                 "unknowns": slots, "evidence_bundles": list(bundle_map.values()),
                 "episodic_candidates": episodic_view if pack.get("semantic_mode") == "episode_evidence" else pack.get("episodic_candidates", []),
                 "episode_bundles": episode_bundles,
                 "required_verification": sorted({item.get("record_id") for item in selected if item.get("area") == "VERIFY" or item.get("use_policy") == "verification_required"}),
                 "omitted_summary": {**compact_summary(omitted), "decision_completeness": 1.0},
                 "aggregation_hints": pack.get("aggregation_hints", {}), "selected_count": len(selected),
                 "omitted_count": len(omitted), "conflict_count": len(conflicts)})
    if episode_bundles:
        pack["evidence_bundles"] = episode_bundles
        pack["episodic_candidates"] = list(pack.get("episodic_candidates") or [])
        pack["episodic_trace"] = dict(pack.get("episodic_trace") or {})
    serialized = json.dumps(pack, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    pack["payload_token_estimate"] = estimator.estimate(serialized)
    pack["token_estimate"] = pack["payload_token_estimate"]
    pack["budget_overflow"] = pack["token_estimate"] > int(pack["token_budget"])
    return pack


def _minimal_budget_pack(pack_base: Dict[str, Any], omitted: List[Dict[str, Any]],
                         selected: List[Dict[str, Any]], conflicts: List[Dict[str, Any]],
                         slots: List[Dict[str, Any]],
                         estimator: ConservativeTokenEstimator) -> Dict[str, Any]:
    """Build a bounded fallback without dropping the only answer evidence."""
    compact_plan = {key: (pack_base.get("query_plan") or {}).get(key)
                    for key in ("primary_intent", "answer_shape", "requested_slots", "fields", "entity_candidates",
                                "relation_terms", "literals", "subject_ref", "speech_act_target", "memory_mode")
                    if key in (pack_base.get("query_plan") or {})}
    candidates = [item for item in selected if item.get("record_type") == "answer_candidate"]
    if not candidates:
        candidates = list(selected)

    def compact_anchor(item: Dict[str, Any]) -> Dict[str, Any]:
        if item.get("record_type") == "answer_candidate":
            source_spans = list(item.get("source_span") or [])
            priority_kinds = {"table_coordinate", "table_cell", "list_item", "attribute_sentence",
                              "typed_sentence", "quoted_span", "answer_span"}
            prioritized = [span for span in source_spans if span.get("kind") in priority_kinds]
            remainder = [span for span in source_spans if span not in prioritized]
            return {
                "record_type": "answer_candidate", "record_id": item.get("record_id"),
                "candidate_id": item.get("candidate_id"), "observation_id": item.get("observation_id"),
                "source_ids": list(item.get("source_ids") or []), "episode_id": item.get("episode_id"),
                "session_id": item.get("session_id"), "turn_id": item.get("turn_id"),
                "turn_index": item.get("turn_index"), "role": item.get("role"),
                "speech_act": item.get("speech_act"), "answer_shape": item.get("answer_shape"),
                "score": item.get("score"),
                "text": _answer_evidence_text(item)[:700],
                "answer_span": item.get("answer_span"),
                "source_span": (prioritized + remainder)[:4],
                "provenance": {"episode_id": item.get("episode_id"), "turn_id": item.get("turn_id"),
                               "observation_id": item.get("observation_id")},
            }
        return _compact_hard_selected_item(item)

    anchors: List[Dict[str, Any]] = [compact_anchor(item) for item in candidates]
    if not anchors and selected:
        anchors.append(compact_anchor(selected[0]))

    source_ids = [source_id for item in anchors for source_id in (item.get("source_ids") or [])]
    episode_ids = list(dict.fromkeys(item.get("episode_id") for item in anchors if item.get("episode_id")))
    session_ids = list(dict.fromkeys(item.get("session_id") for item in anchors if item.get("session_id")))
    compact_bundles = []
    for bundle in pack_base.get("episode_bundles") or []:
        if bundle.get("session_id") not in session_ids and bundle.get("episode_id") not in episode_ids:
            continue
        compact_bundles.append({
            "bundle_id": bundle.get("bundle_id"), "episode_id": bundle.get("episode_id"),
            "session_id": bundle.get("session_id"),
            "anchor_candidates": [item.get("record_id") for item in anchors],
            "question_turn_ids": list(bundle.get("question_turn_ids") or [])[:1],
            "answer_turn_ids": list(bundle.get("answer_turn_ids") or [])[:2],
            "primary_answer_candidate_ids": list(bundle.get("primary_answer_candidate_ids") or [])[:4],
            "primary_answer_turn_ids": list(bundle.get("primary_answer_turn_ids") or [])[:4],
            "question_pair_ids": list(bundle.get("question_pair_ids") or [])[:4],
            "required_answer_units": list(bundle.get("required_answer_units") or [])[:12],
            "optional_related_turn_ids": list(bundle.get("optional_related_turn_ids") or [])[:8],
            "source_observations": source_ids[:4],
            "coverage": bundle.get("coverage", {}),
            "answer_candidates": [item.get("record_id") for item in anchors[:2]],
        })
        if len(compact_bundles) >= 2:
            break

    pack = {
        "schema_version": pack_base.get("schema_version", "semantic-context-pack.v0.6"),
        "semantic_mode": pack_base.get("semantic_mode", "episode_evidence"),
        "budget_mode": "hard", "query": pack_base.get("query"), "scope": pack_base.get("scope", {}),
        "query_plan": compact_plan, "token_budget": int(pack_base.get("token_budget", 0)),
        "budget_unit": pack_base.get("budget_unit", "pimem_internal_estimate"),
        "memory_view": [{"record_id": item.get("record_id"), "statement": item.get("text", "")[:180],
                         "area": item.get("area"), "state_eligible": False} for item in anchors],
        "evidence_view": [{"record_id": item.get("record_id"), "record_type": item.get("record_type"),
                           "observation_ids": item.get("source_ids", []), "session_id": item.get("session_id"),
                           "turn_index": item.get("turn_index")} for item in anchors],
        "selected": anchors, "episodic_candidates": anchors,
        "episode_bundles": compact_bundles, "evidence_bundles": compact_bundles,
        "omitted": [_compact_omitted_item(item) for item in omitted],
        "omitted_summary": {**compact_summary(omitted), "decision_completeness": 1.0},
        "unknowns": slots[:8],
        "conflicts": [{"claim_ids": item.get("claim_ids", [])} for item in conflicts],
        "required_verification": [], "aggregation_hints": {}, "budget_compressed": True,
        "content_hash": "0" * 64, "payload_token_estimate": 0, "token_estimate": 0,
        "budget_overflow": False, "retrieval_count": len(anchors), "candidate_count": len(selected),
        "selected_count": len(anchors), "omitted_count": len(omitted), "conflict_count": len(conflicts),
    }
    # Remove optional fields deterministically until the minimal pack fits; never
    # remove the anchor text/source provenance itself.
    optional_keys = ("memory_view", "evidence_view", "omitted", "omitted_summary",
                     "unknowns", "conflicts", "items", "selected", "query")
    for _ in range(len(optional_keys) + 1):
        serialized = json.dumps(pack, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        estimate = estimator.estimate(serialized)
        pack["token_estimate"] = estimate
        pack["payload_token_estimate"] = estimate
        if estimate <= int(pack["token_budget"]):
            break
        if not optional_keys:
            break
        pack.pop(optional_keys[0], None)
        optional_keys = optional_keys[1:]
    serialized = json.dumps(pack, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    pack["content_hash"] = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    pack["token_estimate"] = estimator.estimate(json.dumps(pack, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    pack["payload_token_estimate"] = pack["token_estimate"]
    for key in ("memory_view", "evidence_view", "selected", "items",
                "episodic_candidates", "required_verification", "aggregation_hints"):
        if pack["token_estimate"] <= int(pack["token_budget"]):
            break
        if key in pack and not pack.get(key):
            pack.pop(key, None)
            pack["token_estimate"] = estimator.estimate(json.dumps(pack, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            pack["payload_token_estimate"] = pack["token_estimate"]
    pack["budget_overflow"] = pack["token_estimate"] > int(pack["token_budget"])
    return pack


def compile_context_pack_v06(store: SQLiteStore, query: str, scope: Scope, *, as_of: Optional[str] = None,
                              token_budget: int = 256, limit: int = 50,
                              options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if token_budget < 1:
        raise ValueError("token_budget must be positive")
    started = time.perf_counter()
    options = dict(options or {})
    budget_mode = str(options.get("budget_mode", "hard"))
    if budget_mode not in {"hard", "unbounded_diagnostic", "unbounded_focused"}:
        raise ValueError("budget_mode must be hard, unbounded_diagnostic, or unbounded_focused")
    semantic_mode = str(options.get("semantic_mode", "episode_evidence"))
    plan = build_query_plan(query, as_of=as_of)
    estimator = options.get("token_estimator") or ConservativeTokenEstimator()
    reranker = options.get("reranker")
    recall_limit = max(limit, 1)
    if semantic_mode == "episode_evidence" and plan.memory_mode in {"episodic_recall", "hybrid"}:
        if budget_mode in {"hard", "unbounded_diagnostic", "unbounded_focused"}:
            recall_limit = max(recall_limit, 10000)
        else:
            recall_limit = max(recall_limit, limit * 4, 200)
    recalled = recall(store, query, scope, limit=recall_limit, as_of=as_of, plan=plan,
                      reranker=reranker, include_observations=True)
    candidates: Dict[str, Dict[str, Any]] = {}
    for item in recalled.get("items", []):
        identity = _record_identity(item)
        if identity not in candidates or float(item.get("score", 0)) > float(candidates[identity].get("score", 0)):
            candidates[identity] = dict(item)
    claim_ids = [str(item["claim_id"]) for item in candidates.values() if item.get("record_type") == "claim" and item.get("claim_id")]
    closure_trace = {"seed_count": len(claim_ids), "closure_count": len(claim_ids), "closure_truncated": False}
    state_claims: List[Claim] = []
    if claim_ids and hasattr(store, "claim_state_closure"):
        state_claims, closure_trace = store.claim_state_closure(claim_ids, as_of=as_of, max_claims=max(limit * 8, 64))
        for claim in state_claims:
            identity = "claim:" + claim.claim_id
            candidates.setdefault(identity, _item_from_claim(claim, store))
    state_claim_by_id = {claim.claim_id: claim for claim in state_claims}
    relations = []
    if state_claims and hasattr(store, "list_claim_relations"):
        families = {claim.assertion_family_key for claim in state_claims if claim.assertion_family_key}
        for family in families:
            relations.extend(store.list_claim_relations(family, as_of=as_of))
    _, state_trace = resolve_claim_state(state_claims, as_of=as_of, relations=relations)
    resolved_ids: set[str] = set()
    for group in state_trace.get("groups", []):
        resolved_ids.update(group.get("selected_claim_ids", []))
    conflicts = _conflict_groups(list(candidates.values()))
    conflict_records = [{"claim_ids": sorted(set(ids)), "reason": "conflicting_assertion_values", "required_verification": True}
                       for ids in sorted({tuple(sorted(values)) for values in conflicts.values()})]
    episode_seed_order = list(candidates.values())
    working: List[Dict[str, Any]] = []
    for item in candidates.values():
        item.setdefault("reasons", item.get("reason", []))
        if item.get("record_type") == "claim" and state_claims and item.get("claim_id") not in resolved_ids:
            item["state_omitted"] = True
            item["state_omitted_reason"] = "not_current_state_candidate"
        if "preference" in {plan.primary_intent, *plan.secondary_intents}:
            item["preference_source"] = _preference_source(item)
        if item.get("claim_id") in conflicts:
            item["area"] = "VERIFY"
            item["reasons"] = list(item.get("reasons", [])) + ["conflict"]
            item["conflict_with"] = [value for value in conflicts[item["claim_id"]] if value != item["claim_id"]]
        working.append(item)
    working = [item for item in working if not item.get("state_omitted")]
    working = _sort_items(working, plan)
    preference_items = [item for item in working if item.get("preference_source") == "explicit_user_preference"]
    ordinary_items = [item for item in working if item not in preference_items]
    ordered = preference_items + ordinary_items if plan.primary_intent == "preference" else working
    episodic_candidates: List[Dict[str, Any]] = []
    episode_bundles: List[Dict[str, Any]] = []
    episode_candidate_records: List[Dict[str, Any]] = []
    episode_bundle_records: List[Dict[str, Any]] = []
    episodic_trace: Dict[str, Any] = {}
    if semantic_mode == "episode_evidence" and plan.memory_mode in {"episodic_recall", "hybrid"}:
        episode_retrieval_limit = max(limit, 1)
        if budget_mode in {"hard", "unbounded_focused"}:
            episode_retrieval_limit = max(episode_retrieval_limit, 10000)
        else:
            episode_retrieval_limit = max(episode_retrieval_limit, 10000)
        episode_seed_items = ordered
        episodic_candidates, episode_bundles, episodic_trace = retrieve_episodic_evidence(
            store, scope, plan, episode_seed_items, limit=episode_retrieval_limit,
            focused=budget_mode in {"unbounded_focused", "hard"},
            prefer_seed_order=False,
            preserve_all_session_candidates=budget_mode == "hard")
        if episodic_candidates:
            has_state_claim = any(item.get("record_type") == "claim" for item in ordered)
            if plan.primary_intent == "preference" and has_state_claim:
                existing = {_record_identity(item) for item in ordered}
                ordered = ordered + [item for item in episodic_candidates if _record_identity(item) not in existing]
            else:
                ordered = episodic_candidates
        candidate_cap = len(episodic_candidates) if budget_mode in {"unbounded_diagnostic", "unbounded_focused"} else 4
        bundle_cap = len(episode_bundles) if budget_mode in {"unbounded_diagnostic", "unbounded_focused"} else 4
        if budget_mode == "hard":
            episode_candidate_records = list(episodic_candidates)
            episode_bundle_records = list(episode_bundles)
            episodic_candidates = []
            episode_bundles = []
        else:
            episodic_candidates = [_compact_episodic_candidate(item) for item in episodic_candidates[:candidate_cap]]
            episode_bundles = [_compact_episode_bundle(
                bundle,
                include_answer_candidates=True,
                include_turn_text=True,
            ) for bundle in episode_bundles[:bundle_cap]]
    selected: List[Dict[str, Any]] = []
    omitted: List[Dict[str, Any]] = []
    for item in ordered:
        item = dict(item)
        if item.get("record_type") == "observation":
            item["area"] = "REFERENCE"
            item["state_eligible"] = False
        selected.append(item)
    if budget_mode == "hard" and semantic_mode == "episode_evidence":
        selected = _select_hard_episode_anchors(selected, 4, plan, episode_bundle_records)
        selected_record_ids = {str(item.get("record_id") or "") for item in selected}
        selected_session_ids = {str(item.get("session_id") or "") for item in selected}
        selected_episode_candidates = [
            item for item in episode_candidate_records
            if str(item.get("record_id") or "") in selected_record_ids
        ]
        episodic_candidates = [_compact_episodic_candidate(item) for item in selected_episode_candidates]
        episode_bundles = [_compact_episode_bundle(
            bundle,
            include_answer_candidates=False,
            include_turn_text=False,
        ) for bundle in episode_bundle_records
           if str(bundle.get("session_id") or "") in selected_session_ids]
    base = {"schema_version": "semantic-context-pack.v0.6", "semantic_mode": semantic_mode,
            "budget_mode": budget_mode,
            "capabilities": ["query_plan", "observation_fallback", "state_closure", "requested_slots", "hard_budget"],
            "query": query, "scope": scope.as_dict(), "as_of": as_of, "level": "L1", "mode": "shadow",
            "memory_mode": "retrieval_shadow", "injected": False, "query_plan": plan.as_dict(),
            "token_budget": token_budget, "budget_unit": "pimem_internal_estimate", "estimator_type": type(estimator).__name__,
            "estimator_version": getattr(estimator, "version", "custom"), "retrieval_trace": _compact_trace(recalled.get("retrieval_trace", {}), query),
            "state_resolution_trace": {**state_trace, "closure": closure_trace}, "retrieval_count": len(selected),
            "candidate_count": len(candidates), "closure_truncated": bool(closure_trace.get("closure_truncated")),
            "evidence_trust": "untrusted_historical_data", "episodic_candidates": episodic_candidates,
            "episode_bundles": episode_bundles,
            "episodic_trace": episodic_trace if budget_mode == "unbounded_diagnostic" else _compact_episodic_trace(episodic_trace),
            "content_hash": "0" * 64, "payload_token_estimate": 0, "token_estimate": 0,
            "budget_overflow": False}
    if semantic_mode == "episode_evidence":
        base["capabilities"].extend(["episodes", "turns", "answer_candidates", "evidence_bundles", "source_spans"])
    base["aggregation_hints"] = _aggregation_hints(plan, ordered) if plan.memory_mode == "semantic_state" else {
        "operator": (plan.aggregation or {}).get("operator"), "dedupe_key": (plan.aggregation or {}).get("dedupe_key"),
        "groups": [], "aggregation_incomplete": True, "requires_external_calculation": False}
    base["budget_compressed"] = False
    slots = _build_slots(plan, selected, [])
    for _ in range(len(selected) + 1):
        pack = _pack_candidate(base, selected, omitted, conflict_records, slots, estimator)
        if base.get("semantic_mode") == "legacy_claim_text" or budget_mode in {"unbounded_diagnostic", "unbounded_focused"} or pack["token_estimate"] <= token_budget or not selected:
            break
        if budget_mode == "hard" and semantic_mode == "episode_evidence" and all(
                item.get("record_type") == "answer_candidate" for item in selected):
            break
        removable = [item for item in selected if item.get("area") != "VERIFY" and item.get("preference_source") != "explicit_user_preference"]
        removable = _budget_removable(removable)
        if not removable:
            removable = [item for item in selected if item.get("record_type") != "answer_candidate"]
        if not removable:
            break
        victim = _choose_budget_victim(removable)
        selected.remove(victim)
        omitted.append({"record_id": victim.get("record_id"), "claim_id": victim.get("claim_id"), "observation_id": victim.get("observation_id"),
                        "record_type": victim.get("record_type", "claim"), "reason": "budget", "token_estimate": _item_cost(victim, "L1")})
        slots = _build_slots(plan, selected, omitted if budget_mode == "unbounded_diagnostic" else [])
    pack = _pack_candidate(base, selected, omitted, conflict_records, slots, estimator)
    pack["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
    while base.get("semantic_mode") != "legacy_claim_text" and budget_mode == "hard" and pack["token_estimate"] > token_budget and selected:
        if semantic_mode == "episode_evidence" and all(item.get("record_type") == "answer_candidate" for item in selected):
            break
        removable = [item for item in selected if item.get("area") != "VERIFY" and item.get("preference_source") != "explicit_user_preference"]
        removable = _budget_removable(removable)
        if not removable:
            break
        victim = _choose_budget_victim(removable)
        selected.remove(victim)
        omitted.append({"record_id": victim.get("record_id"), "claim_id": victim.get("claim_id"), "observation_id": victim.get("observation_id"),
                        "record_type": victim.get("record_type", "claim"), "reason": "budget", "token_estimate": _item_cost(victim, "L1")})
        slots = _build_slots(plan, selected, omitted if budget_mode == "unbounded_diagnostic" else [])
        pack = _pack_candidate(base, selected, omitted, conflict_records, slots, estimator)
        pack["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
    if base.get("semantic_mode") != "legacy_claim_text" and budget_mode == "hard" and pack["token_estimate"] > token_budget:
        pack = _minimal_budget_pack(base, omitted, selected, conflict_records, slots, estimator)
        pack["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
    pack["budget_policy"] = "evidence_anchor_priority_v1" if budget_mode == "hard" else budget_mode
    if budget_mode == "hard":
        # Final metadata (latency/budget_policy/hash placeholders) can push a
        # minimal pack slightly over tiny budgets. Trim only empty optional
        # containers here; never remove non-empty answer anchors or provenance.
        for key in ("omitted", "omitted_summary", "unknowns", "conflicts", "memory_view", "evidence_view",
                    "episode_bundles", "evidence_bundles", "selected", "items", "episodic_candidates",
                    "required_verification", "aggregation_hints"):
            final_serialized = json.dumps(pack, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            if estimator.estimate(final_serialized) <= token_budget:
                break
            if key in pack and not pack.get(key):
                pack.pop(key, None)
    final_serialized = json.dumps(pack, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    pack["payload_token_estimate"] = estimator.estimate(final_serialized)
    pack["token_estimate"] = pack["payload_token_estimate"]
    pack["budget_overflow"] = False if budget_mode in {"unbounded_diagnostic", "unbounded_focused"} else pack["token_estimate"] > token_budget
    pack["content_hash"] = hashlib.sha256(final_serialized.encode("utf-8")).hexdigest()
    store.record_operation("context_prepare", None, f"context:{pack['content_hash']}", None,
                           {"query": query, "scope": scope.as_dict(), "intent": plan.primary_intent,
                            "candidate_count": len(candidates), "selected_count": len(selected), "omitted_count": len(omitted),
                            "token_budget": token_budget, "token_estimate": pack["token_estimate"],
                            "budget_overflow": pack["budget_overflow"], "closure_count": closure_trace.get("closure_count", 0)})
    return pack


def compile_context_pack(store: SQLiteStore, query: str, scope: Scope, *, token_budget: int = 256,
                         level: str = "L0", limit: int = 50) -> Dict[str, Any]:
    if token_budget < 1:
        raise ValueError("token_budget must be positive")
    if level not in {"L0", "L1"}:
        raise ValueError("level must be L0 or L1")
    started = time.perf_counter()
    recalled = recall(store, query, scope, limit=limit)
    candidates = []
    seen = set()
    for item in recalled["items"]:
        if item["claim_id"] not in seen:
            candidates.append(item)
            seen.add(item["claim_id"])
    conflicts = _conflict_groups(candidates)
    enriched = []
    for index, item in enumerate(candidates):
        conflict_ids = conflicts.get(item["claim_id"])
        area = "VERIFY" if conflict_ids else item["area"]
        high_risk = area == "VERIFY" or item["use_policy"] in {"verification_required", "blocked"}
        enriched.append((0 if high_risk else 1, index, item, conflict_ids, high_risk))
    enriched.sort(key=lambda value: (value[0], value[1]))
    selected = []
    omitted = []
    used = 0
    budget_overflow = False
    for _, _, item, conflict_ids, high_risk in enriched:
        cost = _item_cost(item, level)
        if high_risk or used + cost <= token_budget:
            selected.append(_render_item(item, level, conflict_ids))
            used += cost
            budget_overflow = budget_overflow or used > token_budget
        else:
            omitted.append({"claim_id": item["claim_id"], "reason": "budget", "token_estimate": cost, "area": item["area"]})
    conflict_records = [{"claim_ids": sorted(claim_ids), "reason": "conflicting_claims", "required_verification": True}
                        for claim_ids in sorted({tuple(sorted(ids)) for ids in conflicts.values()})]
    required_verification = sorted({item["claim_id"] for item in selected if item["area"] == "VERIFY"})
    as_of = datetime.now(timezone.utc).isoformat()
    content_hash = hashlib.sha256(json.dumps(selected, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    pack = {"schema_version": "semantic-context-pack.v0.5", "query": query, "scope": scope.as_dict(), "as_of": as_of,
            "level": level, "mode": "shadow", "memory_mode": "retrieval_shadow", "injected": False,
            "content_hash": content_hash, "candidate_count": len(candidates), "selected_count": len(selected),
            "items": selected, "selected": selected, "omitted": omitted, "conflicts": conflict_records,
            "warnings": ["budget_overflow"] if budget_overflow else [], "required_verification": required_verification,
            "token_budget": token_budget, "token_estimate": used, "retrieval_count": len(selected), "latency_ms": elapsed_ms}
    store.record_operation("retrieval_shadow", None, f"shadow:{content_hash}", None,
                           {"query": query, "scope": scope.as_dict(), "level": level, "candidate_count": len(candidates),
                            "selected_count": len(selected), "omitted_count": len(omitted), "conflict_count": len(conflict_records),
                            "token_budget": token_budget, "token_estimate": used, "budget_overflow": budget_overflow, "latency_ms": elapsed_ms})
    return pack
