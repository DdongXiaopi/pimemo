from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


class UsePolicy:
    BLOCKED = "blocked"
    REFERENCE_ONLY = "reference_only"
    CONTEXT_ALLOWED = "context_allowed"
    VERIFICATION_REQUIRED = "verification_required"


ALLOWED_PREDICATES = {
    "project_test_command",
    "project_run_command",
    "project_runtime",
    "project_directory",
    "user_code_preference",
    "tool_failure_experience",
    "conversation_fact",
}


@dataclass(frozen=True)
class CaptureDecision:
    allowed: bool
    content: str
    reason: str = "allowed"
    sensitivity: str = "internal"


def validate_scope(scope: Dict[str, Any]) -> None:
    if not scope or not scope.get("repository"):
        raise ValueError("scope.repository is required")
    if not isinstance(scope["repository"], str):
        raise ValueError("scope.repository must be a string")


def validate_claim_policy(predicate: str, truth_status: str, use_policy: str) -> None:
    if predicate not in ALLOWED_PREDICATES:
        raise ValueError(f"unsupported V0 predicate: {predicate}")
    if truth_status not in {"candidate", "supported", "disputed", "retracted"}:
        raise ValueError(f"invalid truth_status: {truth_status}")
    if use_policy not in {
        UsePolicy.BLOCKED,
        UsePolicy.REFERENCE_ONLY,
        UsePolicy.CONTEXT_ALLOWED,
        UsePolicy.VERIFICATION_REQUIRED,
    }:
        raise ValueError(f"invalid use_policy: {use_policy}")
    if truth_status == "retracted" and use_policy == UsePolicy.CONTEXT_ALLOWED:
        raise ValueError("retracted claims cannot be context_allowed")
