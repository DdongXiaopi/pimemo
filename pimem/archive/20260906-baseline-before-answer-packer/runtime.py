from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pimem.adapters.pi import PiAdapter
from pimem.core.models import Claim, ClaimRelation, Observation, Scope
from pimem.core.policy import UsePolicy, validate_claim_policy, validate_scope
from pimem.privacy.sanitizer import sanitize
from pimem.retrieval.fts import recall
from pimem.retrieval.compiler import compile_context_pack, compile_context_pack_v06
from pimem.storage.sqlite import SQLiteStore
from pimem.core.state_machine import classify_relation


class MemoryRuntime:
    def __init__(self, db_path: str) -> None:
        self.store = SQLiteStore(db_path)
        self.pi = PiAdapter(self)

    def init_repository(self, path: str) -> str:
        return self.store.register_repository(path)

    def observe(self, *, content: str, source_type: str, source_ref: str, scope: Scope,
                observed_at: Optional[str] = None, idempotency_key: Optional[str] = None,
                trace_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> Observation:
        validate_scope(scope.as_dict())
        decision = sanitize(content)
        if not decision.allowed:
            self.store.record_operation("capture_rejected_sensitive", source_ref, f"reject:{idempotency_key or source_ref}", trace_id,
                                        {"reason": decision.reason, "source_type": source_type})
            raise ValueError(decision.reason)
        observation = Observation(id="obs_" + uuid.uuid4().hex, content=decision.content, source_type=source_type,
                                  source_ref=source_ref, scope=scope, observed_at=observed_at or datetime.now(timezone.utc).isoformat(),
                                  sensitivity=decision.sensitivity, idempotency_key=idempotency_key or f"{source_type}:{source_ref}",
                                  trace_id=trace_id, metadata=metadata or {})
        saved, _ = self.store.insert_observation(observation)
        return saved

    def ingest_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        return self.pi.capture(event)

    def propose_claims(self, observation_id: str) -> List[Dict[str, Any]]:
        observation = self.store.get_observation(observation_id)
        candidates = self._extract_candidates(observation)
        result = []
        for claim in candidates:
            candidate_id = "cand_" + uuid.uuid4().hex
            self.store.save_candidate(candidate_id, claim, observation_id)
            result.append({"candidate_id": candidate_id, "claim": self.store.claim_to_dict(claim)})
        self.store.record_operation("claim_propose", observation_id, f"propose:{observation_id}", observation.trace_id,
                                    {"count": len(result)})
        return result

    def commit_claim(self, candidate_id: str, *, truth_status: str = "supported",
                     use_policy: str = UsePolicy.CONTEXT_ALLOWED) -> Claim:
        claim = self.store.get_candidate(candidate_id)
        validate_claim_policy(claim.predicate, truth_status, use_policy)
        claim.truth_status = truth_status
        claim.use_policy = use_policy
        claim.lifecycle_status = "active"
        claim.updated_at = datetime.now(timezone.utc).isoformat()
        committed, _ = self.store.commit_claim(claim)
        return committed

    def patch_claim(self, claim_id: str, *, base_revision: int, covered_scope: Dict[str, Any],
                    changes: Dict[str, Any], expected_old: Dict[str, Any],
                    idempotency_key: Optional[str] = None) -> Claim:
        return self.store.patch_claim(claim_id, base_revision=base_revision, covered_scope=covered_scope,
                                      changes=changes, expected_old=expected_old, idempotency_key=idempotency_key)

    def classify_claim_relation(self, claim_id: str, incoming: Claim) -> Optional[str]:
        return classify_relation(self.store.get_claim(claim_id), incoming)

    def recall(self, query: str, scope: Scope, limit: int = 10) -> Dict[str, Any]:
        return recall(self.store, query, scope, limit)

    def shadow(self, query: str, scope: Scope, *, token_budget: int = 256,
               level: str = "L0", limit: int = 50) -> Dict[str, Any]:
        return compile_context_pack(self.store, query, scope, token_budget=token_budget, level=level, limit=limit)

    def prepare_context(self, query: str, scope: Scope, *, as_of: Optional[str] = None,
                        token_budget: int = 256, limit: int = 50,
                        options: Optional[Dict[str, Any]] = None,
                        semantic_mode: str = "legacy_claim_text") -> Dict[str, Any]:
        active_options = dict(options or {})
        active_options.setdefault("semantic_mode", semantic_mode)
        return compile_context_pack_v06(self.store, query, scope, as_of=as_of,
                                        token_budget=token_budget, limit=limit, options=active_options)

    def add_claim_relation(self, relation: ClaimRelation) -> ClaimRelation:
        saved, _ = self.store.add_claim_relation(relation)
        return saved

    def rebuild_state(self) -> Dict[str, Any]:
        return self.store.rebuild_state()

    def rebuild_all_derived(self) -> Dict[str, Any]:
        return self.store.rebuild_all_derived()

    def explain(self, claim_id: str) -> Dict[str, Any]:
        result = self.store.explain(claim_id)
        source_ids = [item["observation_id"] for item in result["evidence"]]
        result["claim"]["source_observation_ids"] = source_ids
        return result

    def rebuild_indexes(self) -> int:
        return self.store.rebuild_indexes()

    def forget(self, target: str, policy: str = "local_runtime") -> None:
        self.store.forget_claim(target, policy)

    def record_task_outcome(self, outcome: Dict[str, Any]) -> None:
        self.store.record_operation("task_outcome", outcome.get("task_id"),
                                    "task_outcome:" + str(outcome.get("run_id") or uuid.uuid4().hex),
                                    outcome.get("trace_id"), outcome)

    @staticmethod
    def _extract_candidates(observation: Observation) -> List[Claim]:
        text = observation.content
        metadata = observation.metadata.get("payload", {}) if observation.metadata else {}
        lower = text.lower()
        found: List[Claim] = []

        def add(predicate: str, value: Any, subject: str = "project") -> None:
            found.append(Claim(claim_id="claim_" + uuid.uuid4().hex, subject=subject, predicate=predicate, object=value,
                               scope=observation.scope, source_observation_ids=[observation.id]))

        command = metadata.get("command")
        if not command:
            command_match = re.search(r"\b(?:python|pytest|npm|pnpm|yarn|cargo|go|make)\b[^\n|]*", text, re.IGNORECASE)
            command = command_match.group(0).strip() if command_match else None
        if command and any(token in lower for token in ("test", "pytest", "run", "命令", "测试")):
            add("project_test_command" if "test" in lower or "pytest" in lower or "测试" in text else "project_run_command", command)
        if any(token in lower for token in ("prefer", "style", "test first", "测试优先", "风格", "偏好")):
            add("user_code_preference", text, subject="user")
        if any(token in lower for token in ("fail", "failed", "error", "失败", "错误")):
            add("tool_failure_experience", text, subject="tool")
        if any(token in lower for token in ("python", "node", "java", "rust", "runtime", "运行时")):
            runtime = next((name for name in ("python", "node", "java", "rust") if name in lower), text)
            add("project_runtime", runtime)
        return found
