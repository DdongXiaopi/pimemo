from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .models import Claim, ClaimRelation, stable_hash


def transition(truth_status: str, lifecycle_status: str, use_policy: str, event: str) -> Tuple[str, str, str]:
    if event == "support":
        return "supported", "active", "context_allowed"
    if event == "propose":
        return "candidate", "proposed", "reference_only"
    if event in {"dispute", "challenge"}:
        return "disputed", "active", "reference_only"
    if event in {"retract", "forget"}:
        return "retracted", "deleted", "blocked"
    if event in {"full_replace", "partial_replace", "temporal_change", "scope_split"}:
        return "supported", "active", "context_allowed"
    raise ValueError(f"unknown claim transition: {event}")


def _scope_overlap(left: Claim, right: Claim) -> bool:
    return left.scope.overlaps(right.scope)


def _family_key(claim: Claim) -> str:
    return claim.assertion_family_key or stable_hash({
        "subject": claim.subject, "predicate": claim.predicate, "object_path": claim.object_path,
        "event_type": claim.event_type, "scope": claim.scope.as_dict(),
    })


def classify_relation(existing: Claim, incoming: Claim) -> Optional[str]:
    if existing.subject != incoming.subject or existing.predicate != incoming.predicate:
        return None
    if not _scope_overlap(existing, incoming):
        return "scope_split"
    if (existing.valid_from, existing.valid_to) != (incoming.valid_from, incoming.valid_to) and any(
        value is not None for value in (existing.valid_from, existing.valid_to, incoming.valid_from, incoming.valid_to)
    ):
        return "temporal_change"
    if existing.object == incoming.object:
        return "duplicate"
    return "full_replace"


def _parse_time(value: Optional[str]) -> datetime:
    if not value:
        return datetime.min.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        for pattern in ("%Y-%m-%d", "%Y/%m/%d", "%Y/%m/%d (%a) %H:%M"):
            try:
                return datetime.strptime(value, pattern).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        return datetime.min.replace(tzinfo=timezone.utc)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _intervals_overlap(left: Claim, right: Claim) -> bool:
    left_start, right_start = _parse_time(left.valid_from), _parse_time(right.valid_from)
    left_end = _parse_time(left.valid_to) if left.valid_to else None
    right_end = _parse_time(right.valid_to) if right.valid_to else None
    return (left_end is None or right_start < left_end) and (right_end is None or left_start < right_end)


def resolve_claim_state(claims: Iterable[Claim], *, as_of: Optional[str] = None,
                        relations: Optional[Iterable[ClaimRelation]] = None) -> Tuple[List[Claim], Dict[str, Any]]:
    anchor = _parse_time(as_of) if as_of else datetime.now(timezone.utc)
    claim_list = list(claims)
    relation_list = [relation for relation in (relations or [])
                     if not relation.effective_at or _parse_time(relation.effective_at) <= anchor]
    relation_by_target: Dict[str, List[ClaimRelation]] = {}
    for relation in relation_list:
        relation_by_target.setdefault(relation.to_claim_id, []).append(relation)
    inactive_by_relation: Dict[str, str] = {}
    for relation in relation_list:
        if relation.relation_type == "supersedes":
            inactive_by_relation[relation.to_claim_id] = "superseded_by_relation"
        elif relation.relation_type == "retracts":
            inactive_by_relation[relation.to_claim_id] = "retracted_by_relation"

    groups: Dict[Tuple[str, str], List[Claim]] = {}
    for claim in claim_list:
        groups.setdefault((_family_key(claim), claim.scope.repository), []).append(claim)
    selected: List[Claim] = []
    trace: Dict[str, Any] = {"as_of": as_of, "groups": [], "relation_count": len(relation_list)}
    for (family_key, _repository), variants in groups.items():
        eligible: List[Claim] = []
        rejected: List[Dict[str, str]] = []
        for claim in variants:
            if claim.claim_id in inactive_by_relation:
                rejected.append({"claim_id": claim.claim_id, "reason": inactive_by_relation[claim.claim_id]})
                continue
            lifecycle_inactive = claim.lifecycle_status in {"deleted", "expired", "archived"} or claim.truth_status == "retracted"
            superseded_without_effective_relation = claim.lifecycle_status == "superseded" and claim.claim_id not in inactive_by_relation
            if lifecycle_inactive or (claim.lifecycle_status == "superseded" and not (as_of and superseded_without_effective_relation)):
                rejected.append({"claim_id": claim.claim_id, "reason": "inactive"})
                continue
            valid_from = _parse_time(claim.valid_from)
            valid_to = _parse_time(claim.valid_to) if claim.valid_to else None
            if valid_from > anchor or (valid_to is not None and anchor >= valid_to):
                rejected.append({"claim_id": claim.claim_id, "reason": "outside_valid_interval"})
                continue
            eligible.append(claim)
        if not eligible:
            continue
        conflicts: List[Tuple[str, str]] = []
        for index, left in enumerate(eligible):
            for right in eligible[index + 1:]:
                if left.object != right.object and _scope_overlap(left, right) and _intervals_overlap(left, right):
                    conflicts.append((left.claim_id, right.claim_id))
        if conflicts:
            selected.extend(eligible)
            selected_ids = [claim.claim_id for claim in eligible]
            group_status = "conflicted"
        else:
            winner = max(eligible, key=lambda claim: (
                1 if claim.truth_status == "supported" else 0,
                1 if claim.use_policy == "context_allowed" else 0,
                _parse_time(claim.valid_from), claim.revision, _parse_time(claim.updated_at), claim.claim_id,
            ))
            selected.append(winner)
            selected_ids = [winner.claim_id]
            group_status = "resolved"
        trace["groups"].append({
            "assertion_family_key": family_key,
            "subject": variants[0].subject,
            "predicate": variants[0].predicate,
            "candidate_claim_ids": [claim.claim_id for claim in variants],
            "selected_claim_ids": selected_ids,
            "status": group_status,
            "conflicts": [list(pair) for pair in conflicts],
            "rejected": rejected,
            "relation_targets": {claim.claim_id: [relation.relation_type for relation in relation_by_target.get(claim.claim_id, [])]
                                 for claim in variants if claim.claim_id in relation_by_target},
        })
    return selected, trace
