from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from pimem.core.models import Scope
from pimem.storage.sqlite import SQLiteStore

from .episodes import build_episode
from .evidence_units import answer_candidates
from .query_plan import QueryPlan, expand_query_tokens


_SESSION_GENERIC_TERMS = {
    "previous", "conversation", "chat", "remind", "checking", "about", "planning", "revisit",
    "looking", "going", "through", "follow", "earlier", "last", "time", "wondering", "could",
    "specific", "certain", "again", "back", "mentioned", "suggested", "recommended",
}

_SESSION_TOKEN_ALIASES = {
    "recommend": {"recommend", "recommended", "recommendation", "suggest", "suggested"},
    "restaurant": {"restaurant", "restaurants", "eatery", "cafe", "cafes"},
    "name": {"name", "named", "nickname", "called"},
    "wearing": {"wear", "wears", "wore", "wearing", "shirt", "dress", "coat", "jacket", "outfit"},
    "designation": {"designation", "designated", "code", "number", "file"},
    "allocated": {"allocated", "allocation", "allocate", "budget", "cost", "amount"},
    "campaign": {"campaign", "campaigns", "plan", "planning"},
    "construction": {"construction", "construct", "constructed", "building", "began", "started"},
}


def _token_family(token: str) -> set[str]:
    normalized = token.lower().strip()
    aliases = _SESSION_TOKEN_ALIASES.get(normalized)
    if aliases:
        return aliases
    variants = {normalized}
    for suffix in ("ing", "ed", "es", "s"):
        if normalized.endswith(suffix) and len(normalized) > len(suffix) + 2:
            variants.add(normalized[:-len(suffix)])
    return variants


def _literal_present(content: str, literal: str) -> bool:
    normalized_content = re.sub(r"\s+", " ", content.lower()).strip()
    normalized_literal = re.sub(r"\s+", " ", literal.lower()).strip()
    return bool(normalized_literal and normalized_literal in normalized_content)


def _slot_matches(slot: Dict[str, Any], item: Dict[str, Any]) -> bool:
    haystack = " ".join(str(item.get(key) or "") for key in ("subject", "predicate", "object", "statement", "text")).lower()
    entity = str(slot.get("entity") or "").lower()
    field = str(slot.get("field") or "").lower()
    if entity and entity != "self" and entity not in haystack:
        return False
    if field and field not in {"", "unknown", "value"} and field not in haystack:
        return entity == "self" and entity in haystack
    return bool(entity or field)


def _focused_closure_rank(item: Dict[str, Any], primary: Dict[str, Any]) -> Tuple[int, ...]:
    turn_distance = abs(int(item.get("turn_index") or 0) - int(primary.get("turn_index") or 0))
    typed_evidence = sum(int(item.get(field, 0) or 0) for field in (
        "requested_entity_hits", "literal_hits", "ordinal_match", "typed_field_match",
        "currency_relevance", "numeric_hits", "relation_hits", "unit_score",
    ))
    same_question = int(bool(item.get("question_turn_id") and item.get("question_turn_id") == primary.get("question_turn_id")))
    adjacent = int(turn_distance <= 1)
    complementary_role = int(item.get("role") != primary.get("role"))
    return (typed_evidence, same_question, adjacent, complementary_role,
            -turn_distance, int(float(item.get("score", 0) or 0)))


def _question_like_turn(item: Dict[str, Any]) -> bool:
    speech_act = str(item.get("speech_act") or "")
    pair_type = str(item.get("pair_type") or "")
    content = str(item.get("content") or "").strip()
    return (
        speech_act in {"user_question", "assistant_question", "question"}
        or pair_type in {"user_without_preceding_question", "missing_source_pair"}
        or content.endswith("?")
    )


def _primary_session_rank(item: Dict[str, Any], plan: QueryPlan | None = None) -> Tuple[int, ...]:
    speech_act = str(item.get("speech_act") or "")
    pair_type = str(item.get("pair_type") or "")
    role = str(item.get("role") or "")
    answer_units = item.get("answer_units") or []
    structured_kind = int(any(unit.get("kind") in {
        "table_coordinate", "table_cell", "table_row", "list_item",
        "quoted_span", "typed_sentence", "attribute_sentence",
    } for unit in answer_units))
    slot_hits = sum(1 for slot in ((plan.requested_slots if plan else []) or []) if _slot_matches(slot, item))
    historical_answer = int(
        bool(plan and getattr(plan, "speech_act_target", None) == "historical_answer")
        and role == "user"
        and speech_act == "user_statement"
        and (bool(item.get("question_turn_id")) or pair_type == "user_historical_statement")
    )
    answer_like = int(
        speech_act in {"assistant_answer", "assistant_recommendation", "assistant_instruction"}
        or historical_answer
        or pair_type == "user_historical_statement"
        or slot_hits > 0
        or structured_kind > 0
    )
    question_like = int(_question_like_turn(item))
    return (
        historical_answer,
        int(role == "assistant"),
        answer_like,
        int(not question_like),
        int(pair_type == "nearest_preceding_user"),
        slot_hits,
        structured_kind,
        -int(item.get("turn_index") or 0),
        int(item.get("typed_relation_score") or 0),
        int(item.get("unit_score") or 0),
        int(item.get("relation_hits") or 0),
        int(item.get("requested_entity_hits") or 0),
        int(item.get("typed_field_match") or 0),
        int(item.get("ordinal_match") or 0),
        int(item.get("numeric_hits") or 0),
        int(item.get("literal_hits") or 0),
        int(item.get("answer_overlap") or 0),
        int(float(item.get("score", 0) or 0) * 100),
    )

def _focused_session_closure(session_items: List[Dict[str, Any]], per_session_limit: int,
                             plan: QueryPlan | None = None) -> List[Dict[str, Any]]:
    if not session_items:
        return []
    answer_items = [item for item in session_items if item.get("speech_act") not in {
        "user_question", "assistant_question", "question"
    }] or session_items
    primary = max(answer_items, key=lambda item: _primary_session_rank(item, plan))
    chosen = [primary]
    chosen_ids = {str(primary.get("record_id") or "")}
    primary_id = str(primary.get("record_id") or "")
    ranked = sorted((item for item in session_items if str(item.get("record_id") or "") != primary_id),
                    key=lambda item: _focused_closure_rank(item, primary), reverse=True)
    for item in ranked:
        if len(chosen) >= per_session_limit:
            break
        record_id = str(item.get("record_id") or "")
        if record_id in chosen_ids:
            continue
        rank = _focused_closure_rank(item, primary)
        if rank[0] <= 0 and rank[2] == 0 and rank[1] == 0:
            continue
        enriched = dict(item)
        reasons = list(enriched.get("reasons") or enriched.get("reason") or [])
        if rank[2]:
            reasons.append("adjacent_turn_closure")
        if rank[1]:
            reasons.append("question_pair_closure")
        if rank[0]:
            reasons.append("typed_evidence_closure")
        enriched["reasons"] = list(dict.fromkeys(reasons))
        enriched["closure_relation"] = {
            "anchor_record_id": primary.get("record_id"),
            "turn_distance": abs(int(item.get("turn_index") or 0) - int(primary.get("turn_index") or 0)),
            "same_question": bool(rank[1]),
        }
        chosen.append(enriched)
        chosen_ids.add(record_id)
    return chosen


def _ordered_session_candidates(session_items: List[Dict[str, Any]],
                                plan: QueryPlan | None = None) -> List[Dict[str, Any]]:
    if not session_items:
        return []
    answer_items = [item for item in session_items if item.get("speech_act") not in {
        "user_question", "assistant_question", "question"
    }] or session_items
    primary = max(answer_items, key=lambda item: _primary_session_rank(item, plan))
    primary_id = str(primary.get("record_id") or "")
    remaining = [item for item in session_items if str(item.get("record_id") or "") != primary_id]
    remaining.sort(key=lambda item: (
        _focused_closure_rank(item, primary),
        int(item.get("turn_index") or 0),
        str(item.get("record_id") or ""),
    ), reverse=True)
    return [primary, *remaining]


def _fallback_session_ids(store: SQLiteStore, scope: Scope, plan: QueryPlan,
                           existing: List[str], limit: int = 24) -> List[str]:
    tokens = {token for token in expand_query_tokens(plan.query)
              if len(token) > 2 and token not in _SESSION_GENERIC_TERMS}
    tokens.update(entity.lower() for entity in plan.entity_candidates if entity)
    field_aliases = {
        "education.degree": {"degree", "graduate", "graduated", "education"},
        "duration": {"duration", "took", "spent", "hours", "minutes"},
        "year": {"year", "began", "started", "constructed"},
        "amount": {"amount", "cost", "price", "budget"},
        "count": {"count", "number", "many"},
    }
    for field_name in plan.fields:
        tokens.update(field_aliases.get(field_name, {field_name.lower()}))
    literal_terms = [literal.lower() for literal in plan.literals if literal]
    relation_terms = [term.lower() for term in plan.relation_terms if term]
    if not tokens:
        return []
    phrase_terms = [term.lower() for term in plan.referential_terms if len(term.split()) > 1]
    query_words = [token.lower() for token in expand_query_tokens(plan.query)
                   if len(token) > 2 and token not in _SESSION_GENERIC_TERMS and token.isalpha()]
    conjunction_words = list(dict.fromkeys(
        token for token in query_words
        if len(token) > 3 and token not in _SESSION_GENERIC_TERMS
    ))
    conjunction_pairs = [
        (left, right)
        for index, left in enumerate(conjunction_words)
        for right in conjunction_words[index + 1:]
    ]
    ordinal_values = [int(item.get("ordinal", 0)) for item in plan.ordinal_constraints
                      if int(item.get("ordinal", 0)) > 0]
    query_ngrams = {
        " ".join(query_words[index:index + size])
        for size in (2, 3)
        for index in range(len(query_words) - size + 1)
    }
    scores: Dict[str, Tuple[int, int, int]] = {}
    session_domain_hits: Dict[str, int] = {}
    session_ordinal_hits: Dict[str, int] = {}
    exact_sessions: Dict[str, int] = {}
    for observation in store.list_observations(scope, limit=50000):
        session_id = str(observation.metadata.get("session_id") or "")
        if not session_id:
            continue
        content = observation.content.lower()
        content_tokens = set(re.findall(r"[a-z0-9]+", content))
        exact_hits = sum(bool(entity) and entity.lower() in content for entity in plan.entity_candidates)
        exact_hits += sum(bool(literal) and _literal_present(content, literal) for literal in literal_terms)
        if exact_hits:
            exact_sessions[session_id] = max(exact_sessions.get(session_id, 0), exact_hits)
        hits = 0
        for token in tokens:
            family = _token_family(token)
            if any(re.search(r"\b" + re.escape(variant) + r"\b", content) for variant in family):
                hits += 1
        literal_hits = sum(_literal_present(content, literal) for literal in literal_terms)
        relation_hits = sum(content.count(term) for term in relation_terms)
        for relation in relation_terms:
            relation_words = [word for word in re.findall(r"[a-z0-9]+", relation) if len(word) > 2]
            if len(relation_words) > 1 and all(word in content for word in relation_words):
                relation_hits += 2
        entity_hits = sum(content.count(entity.lower()) for entity in plan.entity_candidates if entity)
        if not hits and not literal_hits and not relation_hits and not entity_hits:
            continue
        phrase_hits = sum(content.count(phrase) for phrase in phrase_terms)
        phrase_hits += sum(content.count(ngram) for ngram in query_ngrams)
        conjunction_hits = sum(left in content_tokens and right in content_tokens
                               for left, right in conjunction_pairs)
        ordinal_list_hits = sum(bool(re.search(r"(?<!\w)" + str(ordinal) + r"\s*[.):]", content))
                             for ordinal in ordinal_values)
        domain_hits = sum(token in content_tokens for token in conjunction_words
                          if token not in {"list", "provided", "career", "employment", "think",
                                           "discussed", "earlier", "previous", "conversation",
                                           "remind", "remember"})
        session_domain_hits[session_id] = max(session_domain_hits.get(session_id, 0), domain_hits)
        session_ordinal_hits[session_id] = max(session_ordinal_hits.get(session_id, 0), ordinal_list_hits)
        phrase_hits += literal_hits * 40 + exact_hits * 12 + conjunction_hits * 8 + relation_hits * 3 + entity_hits * 2
        previous = scores.get(session_id, (0, 0, 0))
        scores[session_id] = (max(previous[0], hits), previous[1] + hits, previous[2] + phrase_hits)
    for session_id, values in list(scores.items()):
        closure_bonus = session_domain_hits.get(session_id, 0) * session_ordinal_hits.get(session_id, 0) * 80
        if closure_bonus:
            scores[session_id] = (values[0], values[1], values[2] + closure_bonus)
    existing_order = {session_id: index for index, session_id in enumerate(existing)}
    ranked = sorted(scores.items(), key=lambda item: (
        -item[1][2], -item[1][0], -item[1][1], existing_order.get(item[0], len(existing_order) + 1), item[0]
    ))
    ranked_ids = [session_id for session_id, _ in ranked]
    ordered_ids = list(dict.fromkeys(ranked_ids))
    return ordered_ids[:max(limit, 1)]


def retrieve_episodic_evidence(store: SQLiteStore, scope: Scope, plan: QueryPlan,
                               seed_items: List[Dict[str, Any]], limit: int = 50,
                               focused: bool = False,
                               prefer_seed_order: bool = False) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    seed_session_ids = []
    for item in seed_items:
        session_id = item.get("session_id") or (item.get("metadata") or {}).get("session_id")
        if session_id and session_id not in seed_session_ids:
            seed_session_ids.append(str(session_id))
    fallback_session_ids = _fallback_session_ids(store, scope, plan, seed_session_ids, limit=512 if focused else 24)
    session_ids = list(fallback_session_ids) if prefer_seed_order else list(seed_session_ids)
    for session_id in fallback_session_ids:
        if session_id not in session_ids:
            session_ids.append(session_id)
    if prefer_seed_order:
        for session_id in seed_session_ids:
            if session_id not in session_ids:
                session_ids.append(session_id)
    candidates: List[Dict[str, Any]] = []
    episodes: List[Dict[str, Any]] = []
    episode_turns: Dict[str, List[Any]] = {}
    for session_id in session_ids:
        observations = store.list_observations(scope, session_id=session_id, limit=500)
        episode, turns = build_episode(observations, session_id)
        episodes.append(episode.as_dict())
        episode_turns[episode.episode_id] = turns
        candidates.extend(answer_candidates(episode, turns, plan, max_chars=None))
    candidates.sort(key=lambda item: (-float(item.get("score", 0)), str(item.get("record_id", ""))))
    selection_limit = max(min(limit, len(candidates)), 1)
    selected = []
    selected_ids = set()
    session_counts: Dict[str, int] = {}
    historical_answer_query = bool(re.search(r"\b(?:i|we)\s+(?:mentioned|said|used|recommended|suggested|gave|bought|visited|wore|chose)\b", plan.query.lower()))
    if not focused:
        # Unbounded diagnostic mode must retain every generated candidate.
        # Its purpose is to diagnose the complete evidence chain, not to apply
        # the focused per-session cap used by hard/focused retrieval.
        per_session_limit = max(len(candidates), 1)
    elif plan.answer_shape in {
        "entity", "attribute", "list", "list_item", "table_cell", "multi_value"
    }:
        per_session_limit = max(4, int(plan.requested_count or 0) + 2, 8 if historical_answer_query else 0)
    else:
        per_session_limit = 3

    def add_candidate(item: Dict[str, Any]) -> bool:
        if len(selected) >= selection_limit:
            return False
        record_id = str(item.get("record_id") or "")
        if record_id in selected_ids:
            return False
        session_id = str(item.get("session_id") or "")
        if session_counts.get(session_id, 0) >= per_session_limit:
            return False
        selected.append(item)
        selected_ids.add(record_id)
        session_counts[session_id] = session_counts.get(session_id, 0) + 1
        return True

    candidates_by_session: Dict[str, List[Dict[str, Any]]] = {}
    for item in candidates:
        candidates_by_session.setdefault(str(item.get("session_id") or ""), []).append(item)
    if focused:
        def session_quality(session_id: str) -> Tuple[int, ...]:
            return max((int(item.get("requested_entity_hits") or 0),
                        int(item.get("literal_hits") or 0),
                        int(item.get("typed_field_match") or 0),
                        int(item.get("relation_hits") or 0),
                        int(item.get("unit_score") or 0),
                        int(float(item.get("score", 0) or 0)))
                       for item in candidates_by_session[session_id])

        if prefer_seed_order:
            ranked_sessions = sorted(
                candidates_by_session,
                key=lambda session_id: (
                    -(session_ids.index(session_id) if session_id in session_ids else len(session_ids)),
                    session_quality(session_id),
                    session_id,
                ),
                reverse=True,
            )
        else:
            ranked_sessions = sorted(
                candidates_by_session,
                key=lambda session_id: (
                    session_quality(session_id),
                    -(session_ids.index(session_id) if session_id in session_ids else len(session_ids)),
                    session_id,
                ),
                reverse=True,
            )
        for session_id in ranked_sessions:
            session_items = candidates_by_session.get(session_id, [])
            if session_items:
                for item in _focused_session_closure(session_items, per_session_limit, plan):
                    if len(selected) >= selection_limit:
                        break
                    add_candidate(item)
            if len(selected) >= selection_limit:
                break
    else:
        ranked_sessions = [session_id for session_id in session_ids if session_id in candidates_by_session]
        ranked_sessions.extend(session_id for session_id in candidates_by_session if session_id not in ranked_sessions)
        ordered_candidates_by_session = {
            session_id: _ordered_session_candidates(session_items, plan)
            for session_id, session_items in candidates_by_session.items()
        }
        for round_index in range(per_session_limit):
            for session_id in ranked_sessions:
                session_items = ordered_candidates_by_session.get(session_id, [])
                if round_index >= len(session_items):
                    continue
                add_candidate(session_items[round_index])
                if len(selected) >= selection_limit:
                    break
            if len(selected) >= selection_limit:
                break
    selected_ids = {item["record_id"] for item in selected}
    bundles: List[Dict[str, Any]] = []
    for episode in episodes:
        episode_items = [item for item in selected if item.get("episode_id") == episode["episode_id"]]
        if not episode_items:
            continue
        all_turns = episode_turns.get(episode["episode_id"], [])
        anchor_indexes = {int(item.get("turn_index") or 0) for item in episode_items}
        question_turns = []
        for item in episode_items:
            question_turn_id = item.get("question_turn_id")
            question_turns.extend(turn for turn in all_turns if turn.turn_id == question_turn_id)
        question_turns = sorted({turn.turn_id: turn for turn in question_turns}.values(), key=lambda turn: turn.turn_index)[-3:]
        answer_turns = [turn for turn in all_turns if turn.turn_index in anchor_indexes]
        related_turns = [turn for turn in all_turns if any(abs(turn.turn_index - index) <= 1 for index in anchor_indexes)][:8]
        def compact_turn(turn: Any) -> Dict[str, Any]:
            return {"turn_id": turn.turn_id, "observation_id": turn.observation_id, "turn_index": turn.turn_index,
                    "role": turn.role, "content": turn.content, "observed_at": turn.observed_at}
        bundles.append({"bundle_id": "bundle:" + episode["episode_id"], "episode_id": episode["episode_id"],
                        "session_id": episode["root_session_id"], "anchor_candidates": [item["record_id"] for item in episode_items[:3]],
                        "answer_candidates": [dict(item) for item in episode_items[:8]],
                        "question_turn_ids": [turn.turn_id for turn in question_turns],
                        "answer_turn_ids": [turn.turn_id for turn in answer_turns],
                        "related_turn_ids": [turn.turn_id for turn in related_turns],
                        "question_turns": [compact_turn(turn) for turn in question_turns],
                        "answer_turns": [compact_turn(turn) for turn in answer_turns],
                        "related_turns": [compact_turn(turn) for turn in related_turns],
                        "chronological_order": [turn.turn_index for turn in sorted(related_turns, key=lambda x: x.turn_index)],
                        "role_sequence": [turn.role for turn in sorted(related_turns, key=lambda x: x.turn_index)],
                        "structured_facts": [fact for item in episode_items for fact in item.get("structured_facts", [])][:96],
                        "facets": [facet for item in episode_items for facet in item.get("facets", [])][:96],
                        "answer_units": [unit for item in episode_items for unit in item.get("answer_units", [])][:96],
                        "source_spans": [span for item in episode_items for span in item.get("source_span", [])][:96],
                        "source_observations": [item["observation_id"] for item in episode_items],
                        "coverage": {"answer_candidate": bool(answer_turns), "question_turn": bool(question_turns),
                                     "source_span": any(item.get("source_span") for item in episode_items)},
                        "confidence": max(float(item.get("confidence", 0)) for item in episode_items),
                        "reasons": ["episode_seed", "answer_candidate", "evidence_closure"]})
    trace = {"target_episode_count": len(episodes), "candidate_count": len(candidates),
             "selected_candidate_count": len(selected), "target_episode_hit": bool(episodes),
             "answer_candidate_hit": bool(selected), "answer_span_coverage": sum(bool(item.get("source_span")) for item in selected),
             "closure_truncated": len(candidates) > len(selected), "closure_count": len(selected),
             "closure_ids": sorted(selected_ids)}
    return selected, bundles, trace




