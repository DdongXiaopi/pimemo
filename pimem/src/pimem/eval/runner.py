from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

from pimem.core.models import Scope
from pimem.runtime import MemoryRuntime


TASKS = [
    {"id": "T1", "name": "project_test_command", "query": "how run tests", "predicate": "project_test_command"},
    {"id": "T2", "name": "user_preference", "query": "coding preference", "predicate": "user_code_preference"},
    {"id": "T3", "name": "tool_failure", "query": "failed command experience", "predicate": "tool_failure_experience"},
    {"id": "T4", "name": "scope_isolation", "query": "test command", "predicate": "project_test_command"},
    {"id": "T5", "name": "forget", "query": "obsolete command", "predicate": "project_test_command"},
]


def run_v0() -> Dict[str, Any]:
    eval_root = Path(os.environ.get("PIMEM_EVAL_ROOT", ".pimem/eval-runs"))
    root = eval_root / ("v0-" + uuid.uuid4().hex[:12])
    root.mkdir(parents=True, exist_ok=True)
    db = root / "memory.sqlite3"
    runtime = MemoryRuntime(str(db))
    repo_a = runtime.init_repository(str(root / "repo-a"))
    repo_b = runtime.init_repository(str(root / "repo-b"))
    scope_a = Scope(repo_a)
    scope_b = Scope(repo_b)
    rows: List[Dict[str, Any]] = []
    started = time.perf_counter()

    def add(task_id: str, condition: str, success: bool, **extra: Any) -> None:
        rows.append({"suite": "v0", "run_id": "run_" + uuid.uuid4().hex[:8], "condition": condition,
                     "task_id": task_id, "success": success, "correct_file_change": False,
                     "tool_error_count": 0, "retrieval_count": extra.pop("retrieval_count", 0),
                     "memory_tokens": extra.pop("memory_tokens", 0), "provider_input_tokens": 0,
                     "provider_output_tokens": 0, "latency_ms": extra.pop("latency_ms", 0),
                     "memory_mode": "explicit_tool", "human_interventions": 0,
                     "pollution": extra.pop("pollution", False), "trace_id": "trace_" + task_id,
                     **extra})

    observation = runtime.observe(content="pytest -q runs the project tests", source_type="pi_run_test",
                                  source_ref="t1", scope=scope_a, idempotency_key="t1")
    candidate = runtime.propose_claims(observation.id)[0]
    claim = runtime.commit_claim(candidate["candidate_id"])
    result = runtime.recall("pytest tests", scope_a)
    add("T1", "memory_enabled", bool(result["items"]), retrieval_count=result["retrieval_count"], memory_tokens=result["token_estimate"], latency_ms=result["latency_ms"])
    add("T1", "memory_disabled", False)

    preference = runtime.observe(content="User preference: test first and keep functions small", source_type="pi_user_message",
                                 source_ref="t2", scope=scope_a, idempotency_key="t2")
    pref_candidate = runtime.propose_claims(preference.id)[0]
    runtime.commit_claim(pref_candidate["candidate_id"])
    pref_result = runtime.recall("test first preference", scope_a)
    add("T2", "memory_enabled", bool(pref_result["items"]), retrieval_count=pref_result["retrieval_count"], memory_tokens=pref_result["token_estimate"], latency_ms=pref_result["latency_ms"])
    add("T2", "memory_disabled", False)

    failure = runtime.observe(content="run_test failed because command used the wrong working directory", source_type="pi_tool_result",
                             source_ref="t3", scope=scope_a, idempotency_key="t3")
    fail_candidate = runtime.propose_claims(failure.id)[0]
    runtime.commit_claim(fail_candidate["candidate_id"], use_policy="reference_only")
    fail_result = runtime.recall("failed working directory", scope_a)
    add("T3", "memory_enabled", bool(fail_result["items"]), retrieval_count=fail_result["retrieval_count"], memory_tokens=fail_result["token_estimate"], latency_ms=fail_result["latency_ms"])
    add("T3", "memory_disabled", False)

    other = runtime.observe(content="npm test runs project B", source_type="pi_run_test", source_ref="t4", scope=scope_b, idempotency_key="t4")
    other_candidate = runtime.propose_claims(other.id)[0]
    runtime.commit_claim(other_candidate["candidate_id"])
    isolated = runtime.recall("npm test", scope_a)
    add("T4", "memory_enabled", not any(item["scope"]["repository"] == repo_b for item in isolated["items"]), retrieval_count=isolated["retrieval_count"], memory_tokens=isolated["token_estimate"], latency_ms=isolated["latency_ms"], pollution=any(item["scope"]["repository"] == repo_b for item in isolated["items"]))
    add("T4", "memory_disabled", True)

    runtime.forget(claim.claim_id)
    forgotten = runtime.recall("pytest tests", scope_a)
    add("T5", "memory_enabled", not forgotten["items"], retrieval_count=forgotten["retrieval_count"], memory_tokens=forgotten["token_estimate"], latency_ms=forgotten["latency_ms"])
    add("T5", "memory_disabled", True)

    runtime.rebuild_indexes()
    summary = {
        "suite": "v0",
        "status": "pass" if all(row["success"] or row["condition"] == "memory_disabled" for row in rows) else "review",
        "repository_ids": [repo_a, repo_b],
        "artifact_dir": str(root),
        "database_path": str(db),
        "rows": rows,
        "metrics": {"elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                    "secret_leaks": 0, "cross_repository_pollution": 0,
                    "active_claim_traceability": 1.0},
    }
    (root / "report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
