from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from pimem.core.models import Observation, Scope
from pimem.privacy.sanitizer import sanitize
from pimem.runtime import MemoryRuntime


@dataclass(frozen=True)
class LongMemEvalCase:
    question_id: str
    question_type: str
    question: str
    answer: Any
    question_date: str
    haystack_dates: List[str]
    haystack_session_ids: List[str]
    haystack_sessions: List[List[Dict[str, Any]]]
    answer_session_ids: List[str]


def _timestamp(value: str) -> str:
    if not value:
        return datetime.now(timezone.utc).isoformat()
    normalized = value.strip()
    for pattern in ("%Y/%m/%d (%a) %H:%M", "%Y/%m/%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(normalized, pattern).replace(tzinfo=timezone.utc).isoformat()
        except ValueError:
            continue
    try:
        parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except ValueError:
        return normalized
    return (parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)).isoformat()


def _parse_case(value: Dict[str, Any]) -> LongMemEvalCase:
    required = (
        "question_id", "question_type", "question", "answer", "question_date",
        "haystack_dates", "haystack_session_ids", "haystack_sessions", "answer_session_ids",
    )
    missing = [name for name in required if name not in value]
    if missing:
        raise ValueError(f"LongMemEval case missing fields: {', '.join(missing)}")
    dates = value["haystack_dates"]
    session_ids = value["haystack_session_ids"]
    sessions = value["haystack_sessions"]
    if not isinstance(dates, list) or not isinstance(session_ids, list) or not isinstance(sessions, list):
        raise ValueError("LongMemEval haystack fields must be lists")
    if not len(session_ids) == len(sessions) == len(dates):
        raise ValueError("LongMemEval haystack lists must have equal lengths")
    if not isinstance(value["answer_session_ids"], list):
        raise ValueError("LongMemEval answer_session_ids must be a list")
    return LongMemEvalCase(
        question_id=str(value["question_id"]),
        question_type=str(value["question_type"]),
        question=str(value["question"]),
        answer=value["answer"],
        question_date=str(value["question_date"]),
        haystack_dates=[str(item) for item in dates],
        haystack_session_ids=[str(item) for item in session_ids],
        haystack_sessions=sessions,
        answer_session_ids=[str(item) for item in value["answer_session_ids"]],
    )


def iter_cases(path: str) -> Iterable[LongMemEvalCase]:
    decoder = json.JSONDecoder()
    with Path(path).open(encoding="utf-8") as source:
        first = source.read(1)
        while first and first.isspace():
            first = source.read(1)
        if first != "[":
            payload = json.loads(first + source.read())
            if isinstance(payload, dict):
                payload = payload.get("data", payload.get("cases"))
            if not isinstance(payload, list):
                raise ValueError("LongMemEval input must be a JSON list")
            for value in payload:
                yield _parse_case(value)
            return
        buffer = ""
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            buffer += chunk
            cursor = 0
            while True:
                while cursor < len(buffer) and (buffer[cursor].isspace() or buffer[cursor] == ","):
                    cursor += 1
                if cursor >= len(buffer):
                    buffer = ""
                    break
                if buffer[cursor] == "]":
                    return
                try:
                    value, end = decoder.raw_decode(buffer, cursor)
                except json.JSONDecodeError:
                    buffer = buffer[cursor:]
                    break
                yield _parse_case(value)
                cursor = end


def load_cases(path: str, limit: Optional[int] = None) -> List[LongMemEvalCase]:
    cases = []
    for case in iter_cases(path):
        cases.append(case)
        if limit is not None and len(cases) >= limit:
            break
    return cases


def ingest_case(runtime: MemoryRuntime, case: LongMemEvalCase, repository_path: str) -> Dict[str, Any]:
    repository = runtime.init_repository(repository_path)
    scope = Scope(repository, branch=case.question_id)
    observation_to_session: Dict[str, str] = {}
    indexed_turns = 0
    observed_turns = 0
    rejected_turns: List[Dict[str, Any]] = []
    observations: List[Observation] = []
    claims: List[Claim] = []
    for session_index, session in enumerate(case.haystack_sessions):
        session_id = case.haystack_session_ids[session_index]
        observed_at = _timestamp(case.haystack_dates[session_index])
        if not isinstance(session, list):
            raise ValueError(f"LongMemEval session must be a list: {session_id}")
        for turn_index, turn in enumerate(session):
            if not isinstance(turn, dict) or turn.get("role") not in {"user", "assistant"}:
                raise ValueError(f"invalid LongMemEval turn in {session_id}")
            content = turn.get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            observed_turns += 1
            decision = sanitize(content)
            if not decision.allowed:
                reason = decision.reason
                runtime.store.record_operation(
                    "capture_rejected_sensitive", f"{case.question_id}:{session_id}:{turn_index}",
                    f"reject:longmemeval:observation:{case.question_id}:{session_id}:{turn_index}", None,
                    {"reason": reason, "source_type": "longmemeval_turn"},
                )
                if not reason.startswith("capture_rejected_sensitive:"):
                    raise ValueError(reason)
                rejected_turns.append({"session_id": session_id, "turn_index": turn_index, "reason": reason})
                continue
            observation = Observation(
                id="obs_" + uuid.uuid4().hex, content=decision.content, source_type="longmemeval_turn",
                source_ref=f"{case.question_id}:{session_id}:{turn_index}", scope=scope, observed_at=observed_at,
                idempotency_key=f"longmemeval:observation:{case.question_id}:{session_id}:{turn_index}",
                sensitivity=decision.sensitivity,
                metadata={
                    "benchmark": "LongMemEval", "question_id": case.question_id,
                    "question_type": case.question_type, "session_id": session_id,
                    "turn_index": turn_index, "role": turn["role"],
                },
            )
            observations.append(observation)
            observation_to_session[observation.id] = session_id
    indexed_turns, _ = runtime.store.bulk_insert_observations_claims(observations, [])
    return {
        "repository": repository, "scope": scope, "observation_to_session": observation_to_session,
        "observed_turns": observed_turns, "indexed_turns": indexed_turns, "rejected_turns": rejected_turns,
    }


def _session_ids(items: Iterable[Dict[str, Any]], observation_to_session: Dict[str, str]) -> Set[str]:
    result: Set[str] = set()
    for item in items:
        result.update(observation_to_session[source_id] for source_id in item.get("source_ids", []) if source_id in observation_to_session)
    return result


def run_longmemeval(input_path: str, artifact_root: Optional[str] = None, limit: Optional[int] = None,
                    offset: int = 0) -> Dict[str, Any]:
    all_cases = load_cases(input_path)
    start = max(offset, 0)
    end = len(all_cases) if limit is None else start + max(limit, 0)
    cases = all_cases[start:end]
    root = Path(artifact_root or ".pimem/eval-runs") / ("longmemeval-" + uuid.uuid4().hex[:12])
    root.mkdir(parents=True, exist_ok=True)
    runtime = MemoryRuntime(str(root / "memory.sqlite3"))
    repository_path = str(root / "repository")
    rows: List[Dict[str, Any]] = []
    observed_turns = 0
    indexed_turns = 0
    checkpoint_path = root / "progress.jsonl"
    for case in cases:
        ingested = ingest_case(runtime, case, repository_path)
        observed_turns += ingested["observed_turns"]
        indexed_turns += ingested["indexed_turns"]
        context_pack = runtime.prepare_context(case.question, ingested["scope"], as_of=case.question_date,
                                               token_budget=4096, limit=50, semantic_mode="episode_evidence")
        episodic_trace = context_pack.get("episodic_trace", {})
        requested_slots = context_pack.get("unknowns", [])
        retrieved_sessions = _session_ids(context_pack["selected"], ingested["observation_to_session"])
        abstention = case.question_id.endswith("_abs")
        target_sessions = set(case.answer_session_ids)
        rows.append({
            "question_id": case.question_id, "question_type": case.question_type, "abstention": abstention,
            "target_session_ids": sorted(target_sessions), "retrieved_session_ids": sorted(retrieved_sessions),
            "evidence_hit": bool(target_sessions & retrieved_sessions) if not abstention else None,
            "retrieval_count": context_pack["retrieval_count"], "memory_tokens": context_pack["token_estimate"],
            "target_episode_hit": episodic_trace.get("target_episode_hit"),
            "answer_candidate_hit": episodic_trace.get("answer_candidate_hit"),
            "answer_span_coverage": episodic_trace.get("answer_span_coverage", 0),
            "requested_slot_coverage": sum(slot.get("status") == "known" for slot in requested_slots),
            "latency_ms": context_pack["latency_ms"], "context_pack": context_pack,
        })
        with checkpoint_path.open("a", encoding="utf-8") as checkpoint:
            checkpoint.write(json.dumps(rows[-1], ensure_ascii=False) + "\n")
        print(json.dumps({
            "completed": len(rows), "total": len(cases), "offset": start,
            "question_id": case.question_id, "evidence_hit": rows[-1]["evidence_hit"],
        }, ensure_ascii=False), flush=True)
    non_abstention = [row for row in rows if not row["abstention"]]
    evidence_hits = sum(bool(row["evidence_hit"]) for row in non_abstention)
    report = {
        "suite": "longmemeval", "status": "pass" if indexed_turns == observed_turns else "review", "evaluation_scope": "canonical_claim_evidence_retrieval",
        "input_path": str(Path(input_path).resolve()), "artifact_dir": str(root),
        "offset": start, "requested_limit": limit,
        "database_path": str(root / "memory.sqlite3"), "case_count": len(cases),
        "abstention_count": sum(row["abstention"] for row in rows), "observed_turns": observed_turns,
        "indexed_turns": indexed_turns,
        "progress_path": str(checkpoint_path),
        "active_claim_traceability": 1.0 if indexed_turns == observed_turns else indexed_turns / observed_turns if observed_turns else 1.0,
        "evidence_hit_rate": evidence_hits / len(non_abstention) if non_abstention else None, "rows": rows,
    }
    (root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
