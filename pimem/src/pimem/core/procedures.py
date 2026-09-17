from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from .models import Scope, utc_now


@dataclass
class Experience:
    experience_id: str
    task_id: str
    goal: str
    scope: Scope
    actions: List[Dict[str, Any]]
    outcome: Dict[str, Any]
    evidence_ids: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)


@dataclass
class CandidateProcedure:
    procedure_id: str
    trigger: str
    preconditions: List[str]
    actions: List[Dict[str, Any]]
    success_criteria: List[str]
    failure_modes: List[str]
    stop_conditions: List[str]
    rollback: List[str]
    scope: Scope
    provenance: List[str]
    risk_level: str = "S1"
    status: str = "candidate"
    utility_status: str = "unknown"
    created_at: str = field(default_factory=utc_now)


def propose_candidate_procedure(
    experiences: List[Experience],
    *,
    procedure_id: str,
    trigger: str,
    preconditions: List[str],
    actions: List[Dict[str, Any]],
    success_criteria: List[str],
    failure_modes: List[str],
    stop_conditions: List[str],
    rollback: List[str],
    scope: Scope,
    risk_level: str = "S1",
) -> CandidateProcedure:
    if not experiences:
        raise ValueError("at least one experience is required")
    if not trigger or not preconditions or not actions or not success_criteria or not stop_conditions or not rollback:
        raise ValueError("procedure requires trigger, preconditions, actions, success criteria, stop conditions, and rollback")
    if risk_level not in {"S0", "S1", "S2", "S3"}:
        raise ValueError(f"invalid risk_level: {risk_level}")
    if any(experience.scope.repository != scope.repository for experience in experiences):
        raise ValueError("procedure provenance crosses repository scope")
    return CandidateProcedure(
        procedure_id=procedure_id,
        trigger=trigger,
        preconditions=list(preconditions),
        actions=[dict(action) for action in actions],
        success_criteria=list(success_criteria),
        failure_modes=list(failure_modes),
        stop_conditions=list(stop_conditions),
        rollback=list(rollback),
        scope=scope,
        provenance=[experience.experience_id for experience in experiences],
        risk_level=risk_level,
    )


def replay_candidate(procedure: CandidateProcedure, experience: Experience) -> Dict[str, Any]:
    scope_match = procedure.scope.repository == experience.scope.repository
    expected_actions = [action.get("name") or action.get("tool") for action in procedure.actions]
    observed_actions = [action.get("name") or action.get("tool") for action in experience.actions]
    action_match = all(name in observed_actions for name in expected_actions if name)
    outcome_success = experience.outcome.get("status") == "success"
    passed = scope_match and action_match and outcome_success
    return {
        "procedure_id": procedure.procedure_id,
        "experience_id": experience.experience_id,
        "status": "pass" if passed else "review",
        "read_only": True,
        "scope_match": scope_match,
        "action_match": action_match,
        "outcome_success": outcome_success,
        "failure_reasons": [
            reason for condition, reason in (
                (not scope_match, "scope_mismatch"),
                (not action_match, "required_action_missing"),
                (not outcome_success, "experience_not_successful"),
            ) if condition
        ],
    }


def shadow_candidate(procedure: CandidateProcedure, experiences: List[Experience]) -> Dict[str, Any]:
    results = [replay_candidate(procedure, experience) for experience in experiences]
    passed = sum(result["status"] == "pass" for result in results)
    return {
        "procedure_id": procedure.procedure_id,
        "status": "shadow_only",
        "candidate_status": procedure.status,
        "active": False,
        "risk_level": procedure.risk_level,
        "replay_count": len(results),
        "pass_count": passed,
        "utility_estimate": passed / len(results) if results else 0.0,
        "results": results,
        "provenance": procedure.provenance,
    }


def procedure_to_dict(procedure: CandidateProcedure) -> Dict[str, Any]:
    value = asdict(procedure)
    value["scope"] = procedure.scope.as_dict()
    return value


def experience_to_dict(experience: Experience) -> Dict[str, Any]:
    value = asdict(experience)
    value["scope"] = experience.scope.as_dict()
    return value


def procedure_from_dict(value: Dict[str, Any]) -> CandidateProcedure:
    parsed = dict(value)
    parsed["scope"] = Scope(**parsed["scope"])
    return CandidateProcedure(**parsed)


def experience_from_dict(value: Dict[str, Any]) -> Experience:
    parsed = dict(value)
    parsed["scope"] = Scope(**parsed["scope"])
    return Experience(**parsed)
