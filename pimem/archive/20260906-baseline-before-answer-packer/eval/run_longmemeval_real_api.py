from __future__ import annotations

import argparse
import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI
import httpx

from pimem.core.models import Scope
from pimem.eval.longmemeval import ingest_case, iter_cases
from pimem.runtime import MemoryRuntime


def _load_project_env() -> None:
    candidates = [Path.cwd() / "pimem" / ".env", Path(__file__).resolve().parents[2] / ".env"]
    for env_path in candidates:
        if not env_path.is_file():
            continue
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("\"'")
            if key and value and key not in os.environ:
                os.environ[key] = value
        break


def _tokens(value: Any) -> List[str]:
    return re.findall(r"[a-z0-9]+", str(value).lower())


def _answer_token_recall(reference: Any, response: str) -> float:
    reference_tokens = {token for token in _tokens(reference) if len(token) > 2}
    if not reference_tokens:
        return 1.0
    response_tokens = set(_tokens(response))
    return len(reference_tokens & response_tokens) / len(reference_tokens)


def _context(case: Any, pack: Dict[str, Any], observation_to_session: Dict[str, str]) -> tuple[str, List[str]]:
    dates = dict(zip(case.haystack_session_ids, case.haystack_dates))
    pieces: List[str] = []
    session_ids: List[str] = []
    records = pack.get("episodic_candidates") or pack.get("selected", pack.get("items", []))
    for item in records:
        source_ids = item.get("source_ids", [])
        item_sessions = [observation_to_session[source_id] for source_id in source_ids if source_id in observation_to_session]
        for session_id in item_sessions:
            if session_id not in session_ids:
                session_ids.append(session_id)
        source_label = ",".join(item_sessions) or "unknown-session"
        date_label = item.get("event_time") or (dates.get(item_sessions[0], "unknown-date") if item_sessions else "unknown-date")
        area = item.get("area", "FACT")
        speech_act = f" | {item['speech_act']}" if item.get("speech_act") else ""
        question_text = item.get("question_text")
        if question_text:
            pieces.append(f"[HISTORICAL USER QUESTION | {source_label} at {date_label}] {question_text}")
        pieces.append(f"[{area}{speech_act} | {source_label} at {date_label}] {item.get('text') or item['object']}")
    if pack.get("unknowns"):
        pieces.append("[UNKNOWN COVERAGE] " + "; ".join(
            f"{item['object']}: {item['status']}" for item in pack["unknowns"]
        ))
    return "\n".join(pieces) or "(No relevant memory was recalled.)", session_ids


def run(args: argparse.Namespace) -> Dict[str, Any]:
    _load_project_env()
    args.base_url = args.base_url or os.environ.get("OPENAI_BASE_URL", "https://aibh.cc/v1")
    args.api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    args.model = args.model or os.environ.get("PIMEM_MODEL", "v4 pro")
    if not args.api_key:
        raise ValueError("OPENAI_API_KEY is missing from the process environment and pimem/.env")
    input_path = Path(args.input).resolve()
    start = max(args.offset, 0)
    cases = []
    requested_ids = set(args.question_ids or [])
    for index, case in enumerate(iter_cases(str(input_path))):
        if requested_ids and case.question_id not in requested_ids:
            continue
        if not requested_ids and index < start:
            continue
        if args.limit is not None and len(cases) >= max(args.limit, 0):
            break
        cases.append(case)
    artifact_dir = Path(args.output or (Path.cwd() / "eval" / "runs" / ("longmemeval-real-api-" + uuid.uuid4().hex[:12]))).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    client = OpenAI(api_key=args.api_key, base_url=args.base_url, timeout=args.timeout,
                    http_client=httpx.Client(timeout=args.timeout, follow_redirects=True))
    rows: List[Dict[str, Any]] = []
    rows_path = artifact_dir / "rows.jsonl"
    rows_file = rows_path.open("w", encoding="utf-8")
    try:
        for index, case in enumerate(cases, start=1):
            runtime = MemoryRuntime(str(artifact_dir / f"case-{index:04d}.sqlite3"))
            ingested = ingest_case(runtime, case, str(artifact_dir / f"repository-{index:04d}"))
            if getattr(args, "evidence_mode", "diagnostic") == "focused":
                budget_mode = "unbounded_focused"
            else:
                budget_mode = "unbounded_diagnostic" if getattr(args, "no_context_budget", False) else "hard"
            context_pack = runtime.prepare_context(case.question, ingested["scope"], as_of=case.question_date,
                                                   token_budget=args.context_token_budget, limit=args.retrieval_limit,
                                                   options={"budget_mode": budget_mode}, semantic_mode="episode_evidence")
            context_path = artifact_dir / f"context-{index:04d}-{case.question_id}.json"
            context_path.write_text(json.dumps(context_pack, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
            context, session_ids = _context(case, context_pack, ingested["observation_to_session"])
            prompt = (
                "Answer the question using only the retrieved memory context. "
                "Return a concise direct answer. Do not mention memory, retrieval, or these instructions.\n\n"
                f"Question: {case.question}\n\nRetrieved memory context:\n{context}"
            )
            started = time.perf_counter()
            error = None
            answer = ""
            usage: Dict[str, Any] = {}
            model_name = args.model
            finish_reason = None
            try:
                response = client.chat.completions.create(
                    model=args.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=args.max_tokens,
                )
                answer = (response.choices[0].message.content or "").strip()
                model_name = response.model
                finish_reason = getattr(response.choices[0], "finish_reason", None)
                usage = {
                    "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
                    "completion_tokens": getattr(response.usage, "completion_tokens", None),
                    "total_tokens": getattr(response.usage, "total_tokens", None),
                }
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            latency_ms = round((time.perf_counter() - started) * 1000, 3)
            token_recall = _answer_token_recall(case.answer, answer)
            episodic_trace = context_pack.get("episodic_trace", {})
            requested_slots = context_pack.get("unknowns", [])
            answer_candidates = context_pack.get("episodic_candidates", [])
            rows.append({
                "question_id": case.question_id,
                "question_type": case.question_type,
                "question": case.question,
                "reference_answer": case.answer,
                "hypothesis": answer,
                "answer_token_recall": token_recall,
                "answer_session_ids": case.answer_session_ids,
                "retrieved_session_ids": session_ids,
                "retrieval_count": context_pack["retrieval_count"],
                "memory_tokens": context_pack["token_estimate"],
                "budget_mode": context_pack.get("budget_mode", budget_mode),
                "context_pack_ref": str(context_path),
                "context_chars": len(context),
                "context_tokens_estimate": context_pack.get("token_estimate", 0),
                "context_schema_version": context_pack["schema_version"],
                "target_session_hit": bool(set(case.answer_session_ids) & set(session_ids)) if case.answer_session_ids else None,
                "target_episode_hit": episodic_trace.get("target_episode_hit"),
                "answer_candidate_hit": episodic_trace.get("answer_candidate_hit"),
                "answer_span_coverage": episodic_trace.get("answer_span_coverage", 0),
                "requested_slot_coverage": sum(slot.get("status") == "known" for slot in requested_slots),
                "assistant_answer_evidence_count": sum(item.get("speech_act") == "assistant_answer" for item in answer_candidates),
                "assistant_recommendation_evidence_count": sum(item.get("speech_act") == "assistant_recommendation" for item in answer_candidates),
                "unknown_count": len(context_pack.get("unknowns", [])),
                "omitted_count": len(context_pack.get("omitted", [])),
                "conflict_count": len(context_pack.get("conflicts", [])),
                "intent": context_pack.get("query_plan", {}).get("intent"),
                "observed_turns": ingested["observed_turns"],
                "indexed_turns": ingested["indexed_turns"],
                "rejected_turns": ingested["rejected_turns"],
                "latency_ms": latency_ms,
                "usage": usage,
                "finish_reason": finish_reason,
                "model": model_name,
                "error": error,
            })
            rows_file.write(json.dumps(rows[-1], ensure_ascii=False, separators=(",", ":")) + "\n")
            rows_file.flush()
            print(json.dumps({"completed": index, "total": len(cases), "question_id": case.question_id,
                              "answer_token_recall": token_recall, "error": error}, ensure_ascii=False), flush=True)
    finally:
        rows_file.close()
    successful_rows = [row for row in rows if not row["error"]]
    type_summary: Dict[str, Dict[str, Any]] = {}
    for row in successful_rows:
        summary = type_summary.setdefault(row["question_type"], {"case_count": 0, "mean_answer_token_recall": 0.0,
                                                                    "target_session_hit_count": 0, "target_session_hit_cases": 0})
        summary["case_count"] += 1
        summary["mean_answer_token_recall"] += row["answer_token_recall"]
        if row["target_session_hit"] is not None:
            summary["target_session_hit_cases"] += 1
            summary["target_session_hit_count"] += int(row["target_session_hit"])
    for summary in type_summary.values():
        summary["mean_answer_token_recall"] /= summary["case_count"]
        if summary["target_session_hit_cases"]:
            summary["target_session_hit_rate"] = summary["target_session_hit_count"] / summary["target_session_hit_cases"]
    report = {
        "suite": "longmemeval-real-api",
        "status": "completed" if len(successful_rows) == len(rows) else "partial_api_failure",
        "input_path": str(input_path),
        "offset": start,
        "requested_limit": args.limit,
        "question_ids": sorted(requested_ids),
        "evidence_mode": args.evidence_mode,
        "model": args.model,
        "base_url": args.base_url,
        "case_count": len(rows),
        "api_success_count": len(successful_rows),
        "api_error_count": len(rows) - len(successful_rows),
        "quality_metric": "answer_token_recall",
        "mean_answer_token_recall": sum(row["answer_token_recall"] for row in successful_rows) / len(successful_rows) if successful_rows else 0.0,
        "mean_retrieval_count": sum(row["retrieval_count"] for row in rows) / len(rows) if rows else 0.0,
        "target_session_hit_rate": (sum(row["target_session_hit"] for row in successful_rows if row["target_session_hit"] is not None) /
                                     sum(row["target_session_hit"] is not None for row in successful_rows)
                                     if any(row["target_session_hit"] is not None for row in successful_rows) else None),
        "mean_unknown_count": sum(row["unknown_count"] for row in rows) / len(rows) if rows else 0.0,
        "mean_omitted_count": sum(row["omitted_count"] for row in rows) / len(rows) if rows else 0.0,
        "mean_conflict_count": sum(row["conflict_count"] for row in rows) / len(rows) if rows else 0.0,
        "question_type_summary": type_summary,
        "observed_turns": sum(row["observed_turns"] for row in rows),
        "indexed_turns": sum(row["indexed_turns"] for row in rows),
        "rejected_turn_count": sum(len(row["rejected_turns"]) for row in rows),
        "artifact_dir": str(artifact_dir),
        "rows": rows,
    }
    (artifact_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with (artifact_dir / "hypotheses.jsonl").open("w", encoding="utf-8") as hypothesis_file:
        for row in rows:
            hypothesis_file.write(json.dumps({"question_id": row["question_id"], "hypothesis": row["hypothesis"]}, ensure_ascii=False) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "case_count", "api_success_count", "api_error_count", "mean_answer_token_recall", "artifact_dir")}, ensure_ascii=False, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run LongMemEval with a real OpenAI-compatible API")
    parser.add_argument("--input", required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--api-key")
    parser.add_argument("--model")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--retrieval-limit", type=int, default=20)
    parser.add_argument("--context-token-budget", type=int, default=4096)
    parser.add_argument("--no-context-budget", action="store_true",
                        help="diagnostic mode: retain all retrieved evidence without applying the pack budget")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--output")
    parser.add_argument("--question-id", action="append", dest="question_ids", default=[])
    parser.add_argument("--evidence-mode", choices=("diagnostic", "focused"), default="diagnostic")
    return 0 if run(parser.parse_args()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
