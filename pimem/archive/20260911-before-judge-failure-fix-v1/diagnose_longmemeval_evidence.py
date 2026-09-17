from __future__ import annotations

import argparse
import gc
import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

from pimem.eval.longmemeval import LongMemEvalCase, ingest_case, iter_cases
from pimem.runtime import MemoryRuntime


def _tokens(value: Any) -> Set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", str(value).lower()) if len(token) > 1 or token.isdigit()}


def _coverage(reference: Any, value: Any) -> float:
    expected = _tokens(reference)
    return 1.0 if not expected else len(expected & _tokens(value)) / len(expected)


def _normalized_atom(value: Any) -> str:
    text = str(value).lower().replace("’", "'")
    text = re.sub(r"[*_`<>]", "", text)
    text = re.sub(r"\s+", " ", text).strip(" \t\r\n.,;:()[]{}")
    return text


def _typed_reference_atoms(answer: Any) -> Dict[str, List[str]]:
    text = str(answer)
    special_literals = re.findall(r"(?<!\w)\d+\.\s*[A-Za-z][A-Za-z0-9+/#-]*(?:\s+[A-Za-z][A-Za-z0-9+/#-]*){0,5}", text)
    return {
        "numeric_value": list(dict.fromkeys(re.findall(r"(?:[$€£]\s?\d[\d,.]*(?:\.\d+)?|\b\d+(?:\.\d+)?\s?%|\b(?:19|20)\d{2}\b)", text, flags=re.IGNORECASE))),
        "phone_number": list(dict.fromkeys(re.findall(r"(?<!\w)\+?\d[\d()\s./-]{6,}\d(?!\w)", text))),
        "url": list(dict.fromkeys(re.findall(r"https?://[^\s>]+", text, flags=re.IGNORECASE))),
        "handle": list(dict.fromkeys(re.findall(r"@[a-z0-9_]+", text, flags=re.IGNORECASE))),
        "quoted": list(dict.fromkeys(match.group(1) for match in re.finditer(r"['\"]([^'\"\n]{1,120})['\"]", text))),
        "special_literal": list(dict.fromkeys(special_literals)),
    }


def _evidence_text(value: Any) -> str:
    if isinstance(value, dict):
        parts = [str(value.get(key) or "") for key in ("content", "candidate_text", "text", "statement", "object", "value", "header", "row_label")]
        for key in ("answer_units", "structured_facts", "facets"):
            nested = value.get(key) or []
            if isinstance(nested, list):
                parts.extend(_evidence_text(item) for item in nested)
        source_span = value.get("source_span")
        if isinstance(source_span, dict):
            parts.append(str(source_span.get("text") or ""))
        return " ".join(part for part in parts if part)
    if isinstance(value, (list, tuple, set)):
        return " ".join(_evidence_text(item) for item in value)
    return str(value)


def _evidence_atoms(case: LongMemEvalCase, plan: Dict[str, Any]) -> Dict[str, List[str]]:
    answer = str(case.answer)
    plan_entities = [str(entity).strip() for entity in plan.get("entity_candidates", []) if str(entity).strip()]
    named_entities = [entity for entity in plan_entities if _tokens(entity) and entity.lower() in answer.lower()]
    if not named_entities:
        ignored = {"the", "a", "an", "this", "that", "these", "those", "according", "he", "she", "it", "i"}
        named_entities = [value for value in re.findall(r"\b[A-Z][A-Za-z0-9'-]*(?:\s+[A-Z][A-Za-z0-9'-]*)*\b", answer)
                          if value.split()[0].lower() not in ignored]
    typed = _typed_reference_atoms(answer)
    atoms: Dict[str, List[str]] = {"named_entity": named_entities, "numeric_value": typed["numeric_value"]}
    answer_shape = plan.get("answer_shape")
    if answer_shape not in {"table_cell", "list", "list_item", "multi_value"} and answer.strip():
        atoms["quote_sentence"] = [answer.strip()]
    if isinstance(case.answer, list):
        atoms["list_item"] = [str(item).strip() for item in case.answer if str(item).strip()]
    elif plan.get("answer_shape") in {"list", "list_item", "multi_value"} and any(mark in answer for mark in (",", ";", " and ")):
        if typed["quoted"]:
            atoms["list_item"] = typed["quoted"]
        else:
            numbered = [match.group(1).strip() for match in re.finditer(
                r"(?:^|\s)\d+\s*[.):]\s*(.*?)(?=(?:\s+\d+\s*[.):])|$)", answer)]
            list_text = numbered or [re.sub(r"^.*?\b(?:were|was|are|is|:)[ ]*", "", answer, count=1, flags=re.IGNORECASE)]
            if len(named_entities) >= 2:
                atoms["list_item"] = named_entities
            elif numbered:
                atoms["list_item"] = [re.sub(r"\s+(?:and|or)\s*$", "", value).strip(" .,;")
                                       for value in numbered if value.strip()]
            else:
                atoms["list_item"] = [part.strip(" .") for value in list_text for part in re.split(r"\s*(?:,|;|\band\b|\bor\b)\s*", value) if part.strip()]
    if typed["special_literal"]:
        atoms["special_literal"] = typed["special_literal"]
    if typed["phone_number"]:
        atoms["phone_number"] = typed["phone_number"]
    if plan.get("answer_shape") == "table_cell":
        coordinate_parts = []
        coordinate_parts.extend(plan_entities)
        coordinate_parts.extend(plan.get("temporal_mentions", []))
        coordinate_parts.extend(re.findall(r"\b\d+\s*(?:am|pm)\s*-\s*\d+\s*(?:am|pm)(?:\s*\([^)]*\))?", answer, flags=re.IGNORECASE))
        atoms["table_coordinate"] = list(dict.fromkeys(part.strip() for part in coordinate_parts if part.strip()))
    if plan.get("fields") or plan.get("relation_terms"):
        attribute_parts = []
        mentioned = re.search(r"\b(?:mentioned|include|including)\b\s+(.+?)(?=\s+as\s+examples?|\s+as\s+an?\s+example|[.!?]|$)", answer, flags=re.IGNORECASE)
        if mentioned:
            attribute_parts.extend(part.strip() for part in re.split(r"\s+(?:or|and)\s+", mentioned.group(1)) if part.strip())
        clothing = re.search(r"\b(?:wearing|wears|wore)\b\s+(?:an?\s+|the\s+)?([^.!?]+)", answer, flags=re.IGNORECASE)
        if clothing:
            attribute_parts.append(clothing.group(1).strip())
        if typed["quoted"]:
            attribute_parts.extend(typed["quoted"])
        if plan.get("fields") == ["year"]:
            attribute_parts.extend(typed["numeric_value"])
        if answer_shape not in {"table_cell", "list", "list_item", "multi_value"}:
            atoms["attribute"] = attribute_parts or ([answer.strip()] if answer.strip() else [])
    return {kind: list(dict.fromkeys(values)) for kind, values in atoms.items() if values}


def _atom_coverage(atoms: Dict[str, List[str]], value: Any) -> Dict[str, float]:
    text = _evidence_text(value)
    normalized_text = _normalized_atom(text)
    fuzzy_kinds = {"quote_sentence", "attribute", "table_coordinate", "list_item"}
    def covered(atom: str, kind: str) -> float:
        normalized = _normalized_atom(atom)
        if not normalized:
            return 1.0
        if normalized in normalized_text:
            return 1.0
        return _coverage(atom, text) if kind in fuzzy_kinds else 0.0
    return {kind: (sum(covered(atom, kind) for atom in values) / len(values) if values else 1.0)
            for kind, values in atoms.items()}


def _best_atom_coverage(atom_maps: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not atom_maps:
        return {}
    kinds = set().union(*(mapping for mapping in atom_maps))
    return {kind: max(mapping.get(kind, 0.0) for mapping in atom_maps) for kind in kinds}


def _required_atom_kinds(plan: Dict[str, Any], atoms: Dict[str, List[str]]) -> Set[str]:
    shape = plan.get("answer_shape")
    fields = set(plan.get("fields") or [])
    required: Set[str] = set()
    if shape == "table_cell":
        required.add("table_coordinate")
    if shape in {"list", "list_item", "multi_value"}:
        required.add("list_item")
    if fields & {"amount", "percentage", "year", "date", "duration"}:
        required.add("numeric_value")
    if re.search(r"\b(?:phone|telephone)\b", plan.get("query", "").lower()) and "phone_number" in atoms:
        required.add("phone_number")
    if "attribute" in fields or plan.get("relation_terms"):
        required.add("attribute")
    if "special_literal" in atoms:
        required.add("special_literal")
    if shape not in {"table_cell", "list", "list_item", "multi_value"} and not required:
        required.add("quote_sentence")
    return {kind for kind in required if kind in atoms}


def _effective_coverage(atom_coverage: Dict[str, float], plan: Dict[str, Any], atoms: Dict[str, List[str]]) -> float:
    required = _required_atom_kinds(plan, atoms)
    return min((atom_coverage.get(kind, 0.0) for kind in required), default=1.0)


def _load_observations(db_path: Path) -> List[Dict[str, Any]]:
    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(
            "SELECT observation_id, metadata_json, content, observed_at, source_ref FROM observations "
            "ORDER BY CAST(json_extract(metadata_json, '$.turn_index') AS INTEGER), observation_id"
        ).fetchall()
    result = []
    for observation_id, metadata_json, content, observed_at, source_ref in rows:
        metadata = json.loads(metadata_json or "{}")
        result.append({
            "observation_id": str(observation_id), "session_id": metadata.get("session_id"),
            "turn_index": int(metadata.get("turn_index", 0)), "role": metadata.get("role", "unknown"),
            "content": str(content), "observed_at": observed_at, "source_ref": source_ref,
        })
    return result


def _pair_turns(turns: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    pairs = []
    last_user_by_session = {}
    for turn in sorted(turns, key=lambda item: (item["turn_index"], item["observation_id"])):
        session_id = turn.get("session_id")
        if turn["role"] == "user":
            if turn.get("content", "").strip().endswith("?"):
                last_user_by_session[session_id] = turn
                continue
            last_user = last_user_by_session.get(session_id)
            pairs.append({
                "question_turn": last_user, "answer_turn": turn,
                "pair_type": "user_historical_statement" if last_user else "user_without_preceding_question",
                "turn_gap": turn["turn_index"] - last_user["turn_index"] if last_user else None,
            })
            last_user_by_session[session_id] = turn
        elif turn["role"] == "assistant":
            last_user = last_user_by_session.get(session_id)
            pairs.append({
                "question_turn": last_user, "answer_turn": turn,
                "pair_type": "nearest_preceding_user" if last_user else "assistant_without_preceding_user",
                "turn_gap": turn["turn_index"] - last_user["turn_index"] if last_user else None,
            })
    return pairs


def _gold_answer_turns(case: LongMemEvalCase, observations: Sequence[Dict[str, Any]], plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    target = [item for item in observations if item["session_id"] in set(case.answer_session_ids)]
    answer_turns = [item for item in target if item["role"] in {"assistant", "user"}]
    if not answer_turns:
        return []
    typed = _typed_reference_atoms(case.answer)
    query_terms = {token for token in _tokens(case.question) if token not in {"previous", "conversation", "chat", "remind", "mentioned", "what", "using", "used"}}
    scored = []
    for item in answer_turns:
        content = item.get("content", "")
        normalized = _normalized_atom(content)
        typed_hits = sum(_normalized_atom(atom) in normalized for values in typed.values() for atom in values)
        lexical = _coverage(case.answer, content)
        query_relevance = sum(token in content.lower() for token in query_terms)
        question_like = item["role"] == "user" and content.strip().endswith("?")
        if question_like and not typed_hits:
            continue
        score = typed_hits * 100 + lexical * 100 + min(query_relevance, 10)
        # 允许 assistant 和 user role，并根据角色权重进行微调
        if item["role"] == "assistant":
            score += 0.02
        scored.append((score, typed_hits, lexical, item))
    if not scored:
        return []
    best_item = max(scored, key=lambda item: (item[1], item[0], item[3].get("role") == "assistant", -len(item[3].get("content", ""))))
    best = best_item[0]
    return [item for score, _, _, item in scored if score == best and best > 0]


def _candidate_chain(case: LongMemEvalCase, pack: Dict[str, Any], observations: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_id = {item["observation_id"]: item for item in observations}
    pair_by_answer = {pair["answer_turn"]["observation_id"]: pair for pair in _pair_turns(observations)}
    result = []
    for rank, candidate in enumerate(pack.get("episodic_candidates", []), 1):
        source_ids = list(candidate.get("source_ids") or [])
        sources = [by_id[source_id] for source_id in source_ids if source_id in by_id]
        source = sources[0] if sources else None
        pair = pair_by_answer.get(source["observation_id"]) if source else None
        result.append({
            "rank": rank, "candidate_id": candidate.get("candidate_id"), "session_id": candidate.get("session_id"),
            "turn_id": candidate.get("turn_id"), "turn_index": candidate.get("turn_index"),
            "role": candidate.get("role"), "speech_act": candidate.get("speech_act"), "score": candidate.get("score"),
            "answer_shape": candidate.get("answer_shape"), "candidate_text": candidate.get("evidence_text") or candidate.get("text", ""),
            "candidate_source_span": candidate.get("source_span", []), "answer_span": candidate.get("answer_span", {}),
            "answer_units": candidate.get("answer_units", []), "facets": candidate.get("facets", []),
            "candidate_evidence": candidate,
            "source_observations": sources,
            "paired_question_turn": pair.get("question_turn") if pair else None,
            "pair_type": pair.get("pair_type") if pair else "missing_source_pair",
            "reference_coverage_in_candidate": _coverage(case.answer, candidate.get("evidence_text") or candidate.get("text", "")),
            "reference_coverage_in_source": _coverage(case.answer, source.get("content", "") if source else ""),
        })
    return result


def _model_context(case: LongMemEvalCase, pack: Dict[str, Any], observation_to_session: Dict[str, str]) -> str:
    dates = dict(zip(case.haystack_session_ids, case.haystack_dates))
    records = pack.get("episodic_candidates") or pack.get("selected", pack.get("items", []))
    pieces = []
    for item in records:
        source_ids = item.get("source_ids", [])
        sessions = [observation_to_session[source_id] for source_id in source_ids if source_id in observation_to_session]
        source_label = ",".join(dict.fromkeys(sessions)) or "unknown-session"
        date_label = item.get("event_time") or (dates.get(sessions[0], "unknown-date") if sessions else "unknown-date")
        speech_act = f" | {item['speech_act']}" if item.get("speech_act") else ""
        pieces.append(f"[{item.get('area', 'FACT')}{speech_act} | {source_label} at {date_label}] {item.get('evidence_text') or item.get('text') or item.get('object', '')}")
    if pack.get("unknowns"):
        pieces.append("[UNKNOWN COVERAGE] " + "; ".join(
            f"{item.get('object')}: {item.get('status')}" for item in pack["unknowns"]
        ))
    return "\n".join(pieces) or "(No relevant memory was recalled.)"


def _diagnose_case(case: LongMemEvalCase, artifact_dir: Path, index: int, budget_mode: str, token_budget: int) -> Dict[str, Any]:
    runtime = MemoryRuntime(str(artifact_dir / f"case-{index:04d}.sqlite3"))
    ingested = ingest_case(runtime, case, str(artifact_dir / f"repository-{index:04d}"))
    pack = runtime.prepare_context(
        case.question, ingested["scope"], as_of=case.question_date, token_budget=token_budget, limit=50,
        options={"budget_mode": budget_mode}, semantic_mode="episode_evidence",
    )
    observations = _load_observations(artifact_dir / f"case-{index:04d}.sqlite3")
    target_observations = [item for item in observations if item["session_id"] in set(case.answer_session_ids)]
    query_plan = pack.get("query_plan", {})
    gold_turns = _gold_answer_turns(case, observations, query_plan)
    candidates = _candidate_chain(case, pack, observations)
    context = _model_context(case, pack, ingested["observation_to_session"])
    gold_ids = {item["observation_id"] for item in gold_turns}
    answer_turn_hit = any(
        item.get("source_observations", [{}])[0].get("observation_id") in gold_ids
        for item in candidates if item.get("source_observations")
    )
    best_candidate = max((item["reference_coverage_in_candidate"] for item in candidates), default=0.0)
    best_source = max((item["reference_coverage_in_source"] for item in candidates), default=0.0)
    target_ids = set(case.answer_session_ids)
    atoms = _evidence_atoms(case, query_plan)
    source_atom_coverage = _best_atom_coverage([_atom_coverage(atoms, item.get("source_observations", [{}])[0].get("content", "")) for item in candidates])
    candidate_atom_coverage = _atom_coverage(atoms, candidates)
    context_atom_coverage = _atom_coverage(atoms, context)
    source_effective_coverage = max((_effective_coverage(_atom_coverage(atoms, item.get("source_observations", [{}])[0].get("content", "")), query_plan, atoms) for item in candidates), default=0.0)
    candidate_effective_coverage = _effective_coverage(candidate_atom_coverage, query_plan, atoms)
    context_effective_coverage = _effective_coverage(context_atom_coverage, query_plan, atoms)
    bundle_records = list(pack.get("evidence_bundles") or pack.get("episode_bundles") or [])
    if pack.get("evidence_bundles") and pack.get("episode_bundles"):
        bundle_records = list(pack.get("evidence_bundles") or [])
        bundle_records.extend(bundle for bundle in pack.get("episode_bundles") or []
                              if bundle not in bundle_records)
    target_episode_ids = {bundle.get("episode_id") for bundle in bundle_records
                          if bundle.get("session_id") in target_ids}
    candidate_hit = best_candidate > 0
    bundle_hit = bool(target_episode_ids and any(item.get("session_id") in target_ids for item in candidates))
    if not (target_ids & {item.get("session_id") for item in candidates}):
        failure_stage = "session_seed"
    elif not target_episode_ids:
        failure_stage = "episode"
    elif not answer_turn_hit:
        failure_stage = "turn"
    elif source_effective_coverage < 0.8:
        failure_stage = "span"
    elif candidate_effective_coverage < 0.8:
        failure_stage = "candidate"
    elif not bundle_hit:
        failure_stage = "bundle"
    elif context_effective_coverage < 0.8:
        failure_stage = "context"
    else:
        failure_stage = None
    return {
        "question_id": case.question_id, "question_type": case.question_type,
        "question": case.question, "reference_answer": case.answer, "question_date": case.question_date,
        "budget_mode": budget_mode, "token_budget": token_budget,
        "answer_session_ids": case.answer_session_ids, "query_plan": pack.get("query_plan", {}),
        "target_session": {
            "ids": case.answer_session_ids, "observation_count": len(target_observations),
            "turns": target_observations, "pairs": _pair_turns(target_observations),
        },
        "target_episode": {"ids": sorted(target_episode_ids),
                           "bundles": [bundle for bundle in bundle_records if bundle.get("episode_id") in target_episode_ids]},
        "answer_turn": {
            "gold_audit_only": True,
            "selection_rule": "max reference token coverage among non-question user/assistant turns in target session",
            "turns": gold_turns, "hit_in_candidates": answer_turn_hit,
        },
        "answer_span": {
            "best_candidate_reference_coverage": best_candidate,
            "best_source_reference_coverage": best_source,
            "candidate_spans": [
                {"candidate_id": item.get("candidate_id"), "turn_index": item.get("turn_index"),
                 "source_span": item.get("candidate_source_span"),
                 "candidate_coverage": item.get("reference_coverage_in_candidate"),
                 "source_coverage": item.get("reference_coverage_in_source")}
                 for item in candidates
            ],
            "evidence_atoms": atoms,
            "source_atom_coverage": source_atom_coverage,
            "candidate_atom_coverage": candidate_atom_coverage,
            "source_effective_coverage": source_effective_coverage,
            "candidate_effective_coverage": candidate_effective_coverage,
            "context_effective_coverage": context_effective_coverage,
        },
        "candidates": candidates,
        "evidence_bundle": {"count": len(pack.get("evidence_bundles", [])), "target_hit": bundle_hit,
                            "bundles": pack.get("evidence_bundles", [])},
        "model_context": {
            "text": context, "reference_coverage": _coverage(case.answer, context),
            "candidate_count": len(candidates), "context_chars": len(context),
            "context_tokens": len(_tokens(context)),
        },
        "metrics": {
            "target_session_hit": bool(target_ids & {item.get("session_id") for item in candidates}),
            "target_candidate_count": sum(1 for item in candidates if item.get("session_id") in target_ids),
            "answer_turn_hit": answer_turn_hit, "answer_candidate_hit": candidate_hit,
            "answer_span_source_coverage": best_source, "answer_span_candidate_coverage": best_candidate,
            "model_context_coverage": _coverage(case.answer, context),
            "source_atom_coverage": source_atom_coverage, "candidate_atom_coverage": candidate_atom_coverage,
            "model_context_atom_coverage": context_atom_coverage,
            "source_effective_coverage": source_effective_coverage,
            "candidate_effective_coverage": candidate_effective_coverage,
            "context_effective_coverage": context_effective_coverage,
            "failure_stage": failure_stage,
            "evaluator_role_mismatch": False,
            "requested_slot_coverage": sum(slot.get("status") == "known" for slot in pack.get("unknowns", [])),
            "candidate_count": len(candidates), "retrieval_count": pack.get("retrieval_count", 0),
        },
        "context_pack": pack,
        "ingestion": {key: ingested[key] for key in ("observed_turns", "indexed_turns", "rejected_turns")},
    }


def _compact_report_row(row: Dict[str, Any], artifact_path: Path) -> Dict[str, Any]:
    metrics = row["metrics"]
    context_pack = row["context_pack"]
    return {
        "question_id": row["question_id"], "question_type": row["question_type"],
        "question": row["question"], "reference_answer": row["reference_answer"],
        "question_date": row["question_date"], "answer_session_ids": row["answer_session_ids"],
        "query_plan": row["query_plan"], "chain_artifact": str(artifact_path),
        "target_session": {"ids": row["target_session"]["ids"], "observation_count": row["target_session"]["observation_count"]},
        "target_episode": {"ids": row["target_episode"]["ids"]},
        "answer_turn": {"selection_rule": row["answer_turn"]["selection_rule"],
                         "turn_ids": [turn["observation_id"] for turn in row["answer_turn"]["turns"]],
                         "hit_in_candidates": row["answer_turn"]["hit_in_candidates"]},
        "answer_span": {"best_candidate_reference_coverage": row["answer_span"]["best_candidate_reference_coverage"],
                        "best_source_reference_coverage": row["answer_span"]["best_source_reference_coverage"],
                        "source_atom_coverage": row["answer_span"]["source_atom_coverage"],
                        "candidate_atom_coverage": row["answer_span"]["candidate_atom_coverage"]},
        "evidence_bundle": {"count": row["evidence_bundle"]["count"], "target_hit": row["evidence_bundle"]["target_hit"]},
        "model_context": {"reference_coverage": row["model_context"]["reference_coverage"],
                          "candidate_count": row["model_context"]["candidate_count"],
                          "context_chars": row["model_context"]["context_chars"],
                          "context_tokens": row["model_context"]["context_tokens"]},
        "metrics": metrics, "budget_overflow": context_pack.get("budget_overflow", False),
        "context_token_estimate": context_pack.get("token_estimate", 0), "ingestion": row["ingestion"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline LongMemEval episodic evidence chain diagnostic")
    parser.add_argument("--input", required=True)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--question-id", action="append", dest="question_ids", default=[])
    parser.add_argument("--budget-mode", choices=("unbounded_diagnostic", "unbounded_focused", "hard"), default="unbounded_diagnostic")
    parser.add_argument("--token-budget", type=int, default=4096)
    parser.add_argument("--chain-artifacts", choices=("failures", "all", "none"), default="failures",
                        help="control large per-case chain JSON artifacts; default preserves failure chains")
    args = parser.parse_args()
    artifact_dir = Path(args.output).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    requested_ids = set(args.question_ids)
    for index, case in enumerate(iter_cases(args.input)):
        if requested_ids and case.question_id not in requested_ids:
            continue
        if not requested_ids and index < max(args.offset, 0):
            continue
        if args.limit is not None and len(cases) >= max(args.limit, 0):
            break
        cases.append(case)
    rows = []
    for index, case in enumerate(cases, 1):
        full_row = _diagnose_case(case, artifact_dir, index, args.budget_mode, args.token_budget)
        is_failure = full_row["metrics"].get("failure_stage") is not None or full_row["metrics"].get("answer_span_source_coverage", 1.0) < 0.99
        if args.chain_artifacts == "all" or (args.chain_artifacts == "failures" and is_failure):
            chain_path = artifact_dir / f"chain-{index:04d}-{case.question_id}.json"
            chain_path.write_text(json.dumps(full_row, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
            row = _compact_report_row(full_row, chain_path)
        else:
            row = _compact_report_row(full_row, Path("N/A"))
        rows.append(row)
        print(json.dumps({"completed": index, "total": len(cases), "question_id": case.question_id,
                          "metrics": row["metrics"]}, ensure_ascii=False), flush=True)
        del full_row
        gc.collect()
    report = {
        "diagnostic_schema_version": "evidence-chain-v1", "suite": "longmemeval-evidence-diagnostic", "status": "completed",
        "input_path": str(Path(args.input).resolve()), "offset": max(args.offset, 0),
        "budget_mode": args.budget_mode, "token_budget": args.token_budget,
        "case_count": len(rows), "artifact_dir": str(artifact_dir),
        "metric_means": {
            "target_session_hit_rate": sum(row["metrics"]["target_session_hit"] for row in rows) / len(rows) if rows else 0.0,
            "answer_turn_hit_rate": sum(row["metrics"]["answer_turn_hit"] for row in rows) / len(rows) if rows else 0.0,
            "answer_candidate_hit_rate": sum(row["metrics"]["answer_candidate_hit"] for row in rows) / len(rows) if rows else 0.0,
            "mean_answer_span_source_coverage": sum(row["metrics"]["answer_span_source_coverage"] for row in rows) / len(rows) if rows else 0.0,
            "mean_answer_span_candidate_coverage": sum(row["metrics"]["answer_span_candidate_coverage"] for row in rows) / len(rows) if rows else 0.0,
            "mean_model_context_coverage": sum(row["metrics"]["model_context_coverage"] for row in rows) / len(rows) if rows else 0.0,
            "mean_source_atom_coverage": {kind: sum(row["metrics"].get("source_atom_coverage", {}).get(kind, 0.0) for row in rows) / len(rows) for kind in set().union(*(row["metrics"].get("source_atom_coverage", {}) for row in rows))} if rows else {},
            "mean_candidate_atom_coverage": {kind: sum(row["metrics"].get("candidate_atom_coverage", {}).get(kind, 0.0) for row in rows) / len(rows) for kind in set().union(*(row["metrics"].get("candidate_atom_coverage", {}) for row in rows))} if rows else {},
            "mean_model_context_atom_coverage": {kind: sum(row["metrics"].get("model_context_atom_coverage", {}).get(kind, 0.0) for row in rows) / len(rows) for kind in set().union(*(row["metrics"].get("model_context_atom_coverage", {}) for row in rows))} if rows else {},
            "failure_stage_counts": {stage: sum(row["metrics"].get("failure_stage") == stage for row in rows) for stage in sorted({row["metrics"].get("failure_stage") for row in rows if row["metrics"].get("failure_stage")})},
            "mean_candidate_count": sum(row["metrics"]["candidate_count"] for row in rows) / len(rows) if rows else 0.0,
            "budget_overflow_rate": sum(bool(row.get("budget_overflow")) for row in rows) / len(rows) if rows else 0.0,
            "mean_context_token_estimate": sum(row.get("context_token_estimate", 0) for row in rows) / len(rows) if rows else 0.0,
        },
        "rows": rows,
    }
    (artifact_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



