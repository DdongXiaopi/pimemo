            include_answer_candidates=False,
            include_turn_text=False,
        ) for bundle in episode_bundle_records
           if str(bundle.get("session_id") or "") in selected_session_ids]
    base = {"schema_version": "semantic-context-pack.v0.6", "semantic_mode": semantic_mode,
            "budget_mode": budget_mode,
            "capabilities": ["query_plan", "observation_fallback", "state_closure", "requested_slots", "hard_budget"],
            "query": query, "scope": scope.as_dict(), "as_of": as_of, "level": "L1", "mode": "shadow",
            "memory_mode": "retrieval_shadow", "injected": False, "query_plan": plan.as_dict(),
            "token_budget": token_budget, "budget_unit": "pimem_internal_estimate", "estimator_type": type(estimator).__name__,
            "estimator_version": getattr(estimator, "version", "custom"), "retrieval_trace": _compact_trace(recalled.get("retrieval_trace", {}), query),
            "state_resolution_trace": {**state_trace, "closure": closure_trace}, "retrieval_count": len(selected),
            "candidate_count": len(candidates), "closure_truncated": bool(closure_trace.get("closure_truncated")),
            "evidence_trust": "untrusted_historical_data", "episodic_candidates": episodic_candidates,
            "episode_bundles": episode_bundles,
            "episodic_trace": episodic_trace if budget_mode == "unbounded_diagnostic" else _compact_episodic_trace(episodic_trace),
            "content_hash": "0" * 64, "payload_token_estimate": 0, "token_estimate": 0,
            "budget_overflow": False}
    if semantic_mode == "episode_evidence":
        base["capabilities"].extend(["episodes", "turns", "answer_candidates", "evidence_bundles", "source_spans"])
    base["aggregation_hints"] = _aggregation_hints(plan, ordered) if plan.memory_mode == "semantic_state" else {
        "operator": (plan.aggregation or {}).get("operator"), "dedupe_key": (plan.aggregation or {}).get("dedupe_key"),
        "groups": [], "aggregation_incomplete": True, "requires_external_calculation": False}
    base["budget_compressed"] = False
    slots = _build_slots(plan, selected, [])
    for _ in range(len(selected) + 1):
        pack = _pack_candidate(base, selected, omitted, conflict_records, slots, estimator)
        if base.get("semantic_mode") == "legacy_claim_text" or budget_mode in {"unbounded_diagnostic", "unbounded_focused"} or pack["token_estimate"] <= token_budget or not selected:
            break
        if budget_mode == "hard" and semantic_mode == "episode_evidence" and all(
                item.get("record_type") == "answer_candidate" for item in selected):
            break
        removable = [item for item in selected if item.get("area") != "VERIFY" and item.get("preference_source") != "explicit_user_preference"]
        removable = _budget_removable(removable)
        if not removable:
            removable = [item for item in selected if item.get("record_type") != "answer_candidate"]
        if not removable:
            break
        victim = _choose_budget_victim(removable)
        selected.remove(victim)
        omitted.append({"record_id": victim.get("record_id"), "claim_id": victim.get("claim_id"), "observation_id": victim.get("observation_id"),
                        "record_type": victim.get("record_type", "claim"), "reason": "budget", "token_estimate": _item_cost(victim, "L1")})
        slots = _build_slots(plan, selected, omitted if budget_mode == "unbounded_diagnostic" else [])
    pack = _pack_candidate(base, selected, omitted, conflict_records, slots, estimator)
    pack["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
    while base.get("semantic_mode") != "legacy_claim_text" and budget_mode == "hard" and pack["token_estimate"] > token_budget and selected:
        if semantic_mode == "episode_evidence" and all(item.get("record_type") == "answer_candidate" for item in selected):
            break
        removable = [item for item in selected if item.get("area") != "VERIFY" and item.get("preference_source") != "explicit_user_preference"]
        removable = _budget_removable(removable)
        if not removable:
            break
        victim = _choose_budget_victim(removable)
        selected.remove(victim)
        omitted.append({"record_id": victim.get("record_id"), "claim_id": victim.get("claim_id"), "observation_id": victim.get("observation_id"),
                        "record_type": victim.get("record_type", "claim"), "reason": "budget", "token_estimate": _item_cost(victim, "L1")})
        slots = _build_slots(plan, selected, omitted if budget_mode == "unbounded_diagnostic" else [])
        pack = _pack_candidate(base, selected, omitted, conflict_records, slots, estimator)
        pack["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
    if base.get("semantic_mode") != "legacy_claim_text" and budget_mode == "hard" and pack["token_estimate"] > token_budget:
        pack = _minimal_budget_pack(base, omitted, selected, conflict_records, slots, estimator)
        pack["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
    pack["budget_policy"] = "evidence_anchor_priority_v1" if budget_mode == "hard" else budget_mode
    if budget_mode == "hard":
        # Final metadata (latency/budget_policy/hash placeholders) can push a
        # minimal pack slightly over tiny budgets. Trim only empty optional
        # containers here; never remove non-empty answer anchors or provenance.
        for key in ("omitted", "omitted_summary", "unknowns", "conflicts", "memory_view", "evidence_view",
                    "episode_bundles", "evidence_bundles", "selected", "items", "episodic_candidates",
                    "required_verification", "aggregation_hints"):
            final_serialized = json.dumps(pack, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            if estimator.estimate(final_serialized) <= token_budget:
                break
            if key in pack and not pack.get(key):
                pack.pop(key, None)
    final_serialized = json.dumps(pack, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    pack["payload_token_estimate"] = estimator.estimate(final_serialized)
    pack["token_estimate"] = pack["payload_token_estimate"]
    pack["budget_overflow"] = False if budget_mode in {"unbounded_diagnostic", "unbounded_focused"} else pack["token_estimate"] > token_budget
    pack["content_hash"] = hashlib.sha256(final_serialized.encode("utf-8")).hexdigest()
    store.record_operation("context_prepare", None, f"context:{pack['content_hash']}", None,
                           {"query": query, "scope": scope.as_dict(), "intent": plan.primary_intent,
                            "candidate_count": len(candidates), "selected_count": len(selected), "omitted_count": len(omitted),
                            "token_budget": token_budget, "token_estimate": pack["token_estimate"],
                            "budget_overflow": pack["budget_overflow"], "closure_count": closure_trace.get("closure_count", 0)})
    return pack


def compile_context_pack(store: SQLiteStore, query: str, scope: Scope, *, token_budget: int = 256,
                         level: str = "L0", limit: int = 50) -> Dict[str, Any]:
