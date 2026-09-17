from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class Scope:
    repository: str
    branch: Optional[str] = None
    service: Optional[str] = None
    module: Optional[str] = None
    file: Optional[str] = None
    session: Optional[str] = None
    task: Optional[str] = None

    def as_dict(self) -> Dict[str, Optional[str]]:
        return asdict(self)

    def matches(self, query: "Scope") -> bool:
        if self.repository != query.repository:
            return False
        for field_name in ("branch", "service", "module", "file", "session", "task"):
            value = getattr(self, field_name)
            query_value = getattr(query, field_name)
            if value is not None and query_value is not None and value != query_value:
                return False
        return True

    def overlaps(self, other: "Scope") -> bool:
        return self.matches(other) and other.matches(self)


@dataclass
class Observation:
    id: str
    content: str
    source_type: str
    source_ref: str
    scope: Scope
    observed_at: str = field(default_factory=utc_now)
    recorded_at: str = field(default_factory=utc_now)
    content_hash: str = ""
    sensitivity: str = "internal"
    idempotency_key: str = ""
    trace_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.content_hash:
            self.content_hash = hashlib.sha256(self.content.encode("utf-8")).hexdigest()
        if not self.idempotency_key:
            self.idempotency_key = f"observation:{self.id}"


@dataclass
class Evidence:
    id: str
    claim_id: str
    observation_id: str
    relation: str = "supports"
    created_at: str = field(default_factory=utc_now)


@dataclass
class ClaimRelation:
    relation_id: str
    from_claim_id: str
    to_claim_id: str
    relation_type: str
    assertion_family_key: str
    covered_paths: List[str] = field(default_factory=list)
    effective_at: Optional[str] = None
    source_observation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)
    trace_id: Optional[str] = None
    idempotency_key: str = ""

    def __post_init__(self) -> None:
        allowed = {"duplicate", "supersedes", "refines", "contradicts", "retracts"}
        if self.relation_type not in allowed:
            raise ValueError(f"invalid claim relation type: {self.relation_type}")
        if self.from_claim_id == self.to_claim_id:
            raise ValueError("claim relation cannot reference itself")
        if not isinstance(self.covered_paths, list) or not all(isinstance(path, str) for path in self.covered_paths):
            raise ValueError("covered_paths must be a list of strings")
        if not self.idempotency_key:
            self.idempotency_key = stable_hash({
                "from_claim_id": self.from_claim_id,
                "to_claim_id": self.to_claim_id,
                "relation_type": self.relation_type,
                "covered_paths": self.covered_paths,
            })


@dataclass
class Facet:
    facet_id: str
    kind: str
    raw_value: Any
    normalized_value: Any
    source_observation_id: str
    span_start: Optional[int] = None
    span_end: Optional[int] = None
    confidence: float = 0.0
    polarity: Optional[str] = None
    modality: Optional[str] = None
    event_time: Optional[str] = None
    extractor_name: str = "builtin"
    extractor_version: str = "1"


@dataclass
class Claim:
    claim_id: str
    subject: str
    predicate: str
    object: Any
    scope: Scope
    source_observation_ids: List[str]
    valid_from: Optional[str] = None
    valid_to: Optional[str] = None
    time_precision: Optional[str] = None
    truth_status: str = "candidate"
    lifecycle_status: str = "proposed"
    use_policy: str = "reference_only"
    utility_status: str = "unknown"
    revision: int = 1
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    idempotency_key: str = ""
    object_path: Optional[str] = None
    event_type: str = "state"
    event_time: Optional[str] = None
    assertion_family_key: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.idempotency_key:
            self.idempotency_key = self.fingerprint()
        if not self.assertion_family_key:
            self.assertion_family_key = stable_hash({
                "subject": self.subject,
                "predicate": self.predicate,
                "object_path": self.object_path,
                "event_type": self.event_type,
            })

    def fingerprint(self) -> str:
        return stable_hash({
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "scope": self.scope.as_dict(),
            "object_path": self.object_path,
            "event_type": self.event_type,
        })

    def statement(self) -> str:
        value = self.object if isinstance(self.object, str) else json.dumps(self.object, ensure_ascii=False, sort_keys=True)
        return f"{self.subject} {self.predicate}: {value}"
