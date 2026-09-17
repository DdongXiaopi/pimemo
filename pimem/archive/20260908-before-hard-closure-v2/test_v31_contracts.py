import json
import tempfile
import unittest
from pathlib import Path

from pimem.adapters.generic import GenericMemoryAdapter
from pimem.core.models import Claim, ClaimRelation, Scope
from pimem.core.state_machine import resolve_claim_state
from pimem.retrieval.budget import ConservativeTokenEstimator
from pimem.retrieval.compiler import _minimal_budget_pack
from pimem.retrieval.facets import BuiltinFacetExtractor
from pimem.retrieval.query_plan import build_query_plan
from pimem.runtime import MemoryRuntime
from pimem.retrieval.compiler import _compact_hard_episodic_candidate, _select_hard_episode_anchors


class V31ContractTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.runtime = MemoryRuntime(str(Path(self.tempdir.name) / "memory.sqlite3"))
        self.repository = self.runtime.init_repository(str(Path(self.tempdir.name) / "repo"))
        self.scope = Scope(self.repository)

    def observation(self, content, *, session="session-1", turn=1, role="user", observed_at=None):
        return self.runtime.observe(
            content=content,
            source_type="generic",
            source_ref=f"{session}:{turn}:{content[:12]}",
            scope=self.scope,
            observed_at=observed_at,
            metadata={"session_id": session, "turn_index": turn, "role": role},
        )

    def claim(self, claim_id, value, *, predicate="project_status", source_ids=None,
              subject="project", family=None, valid_from=None, valid_to=None,
              event_time=None, truth_status="supported", lifecycle_status="active",
              use_policy="context_allowed"):
        item = Claim(
            claim_id=claim_id, subject=subject, predicate=predicate, object=value,
            scope=self.scope, source_observation_ids=list(source_ids or []),
            assertion_family_key=family, valid_from=valid_from, valid_to=valid_to,
            event_time=event_time, truth_status=truth_status,
            lifecycle_status=lifecycle_status, use_policy=use_policy,
        )
        self.runtime.store.commit_claim(item)
        return item

    def test_query_plan_keeps_self_and_secondary_intent(self):
        plan = build_query_plan("What is my latest preferred camera?", as_of="2026-08-31")
        self.assertEqual(plan.subject_ref, "self")
        self.assertEqual(plan.primary_intent, "latest")
        self.assertIn("preference", plan.secondary_intents)
        self.assertEqual(plan.intent_scores["latest"], 1.0)

    def test_query_plan_does_not_treat_phone_number_as_count(self):
        plan = build_query_plan("What is the phone number of the Speyer tourism board?")
        self.assertIn("phone_number", plan.fields)
        self.assertNotEqual(plan.primary_intent, "count")
        self.assertEqual(plan.requested_slots[0]["field"], "phone_number")

    def test_query_plan_types_historical_question_without_grammar_entities(self):
        plan = build_query_plan(
            "I'm checking our previous chat. Can you remind me what was the rotation for Admon on a Sunday?"
        )
        self.assertEqual(plan.entities, ["Admon"])
        self.assertNotIn("I'm", plan.entities)
        self.assertNotIn("Can", plan.entities)
        self.assertNotIn("Sunday", plan.entities)
        self.assertEqual(plan.answer_shape, "table_cell")
        self.assertEqual(plan.memory_mode, "episodic_recall")
        self.assertIn("previous chat", plan.referential_terms)
        self.assertEqual(len(plan.requested_slots), 1)

    def test_query_plan_personal_fact_questions_use_episodic_recall(self):
        cases = (
            "Where did I redeem a $5 coupon on coffee creamer?",
            "What type of rice is my favorite?",
            "Where did I complete my Bachelor's degree in Computer Science?",
            "Where did I go on a week-long trip with my family?",
        )
        for query in cases:
            with self.subTest(query=query):
                plan = build_query_plan(query)
                self.assertEqual(plan.memory_mode, "episodic_recall")
                self.assertNotIn("Where", plan.entities)
                self.assertNotIn("Bachelor", plan.entities)

    def test_builtin_facets_have_source_spans_and_proposals(self):
        observation = self.observation("I prefer Fujifilm and stayed 4 weeks.")
        result = BuiltinFacetExtractor().extract(observation)
        self.assertTrue(result.facets)
        self.assertTrue(all(f.source_observation_id == observation.id for f in result.facets))
        self.assertTrue(all(f.span_start is not None and f.span_end is not None for f in result.facets))
        self.assertTrue(result.claim_proposals)

    def test_builtin_facets_are_stable_and_typed(self):
        observation = self.observation("Sunday: Admon works 8 am - 4 pm.\n1. SQLite")
        first = BuiltinFacetExtractor().extract(observation)
        second = BuiltinFacetExtractor().extract(observation)
        self.assertEqual([facet.facet_id for facet in first.facets], [facet.facet_id for facet in second.facets])
        self.assertIn("date", {facet.kind for facet in first.facets})
        self.assertIn("named_entity", {facet.kind for facet in first.facets})
        self.assertIn("ordinal", {facet.kind for facet in first.facets})

    def test_episode_bundle_has_answer_candidate_and_source_span(self):
        self.observation("What shift is Admon assigned on Sunday?", turn=1, role="user")
        answer = self.observation("Admon is assigned to the 8 am - 4 pm (Day Shift) on Sunday.", turn=2, role="assistant")
        pack = self.runtime.prepare_context(
            "Can you remind me what was the rotation for Admon on a Sunday?", self.scope,
            semantic_mode="episode_evidence", token_budget=4096,
        )
        self.assertTrue(pack["episodic_candidates"])
        candidate = pack["episodic_candidates"][0]
        self.assertEqual(candidate["record_type"], "answer_candidate")
        self.assertEqual(candidate["observation_id"], answer.id)
        self.assertEqual(candidate["speech_act"], "assistant_answer")
        self.assertTrue(candidate["source_span"])
        self.assertIn("8 am - 4 pm", candidate["text"])
        self.assertTrue(pack["evidence_bundles"])
        self.assertIn(candidate["turn_id"], pack["evidence_bundles"][0]["answer_turn_ids"])

    def test_episode_evidence_extracts_table_value_and_user_historical_answer(self):
        self.observation("What was Admon's rotation on Sunday?", turn=1, role="user")
        answer = self.observation(
            "Schedule:\n| Day | 8 am - 4 pm (Day Shift) |\n| --- | --- |\n| Sunday | Admon |",
            turn=2, role="assistant",
        )
        plan = build_query_plan("Can you remind me what was Admon's rotation on Sunday?")
        pack = self.runtime.prepare_context(
            "Can you remind me what was Admon's rotation on Sunday?", self.scope,
            semantic_mode="episode_evidence", token_budget=4096,
        )
        self.assertTrue(pack["episodic_candidates"])
        self.assertTrue(any("8 am - 4 pm" in item["text"] and "Admon" in item["text"] for item in pack["episodic_candidates"]))
        self.assertEqual(plan.answer_shape, "table_cell")

        self.observation("What show did you use as an example?", session="session-2", turn=1, role="assistant")
        self.observation("The show I used as an example was Doc Martin.", session="session-2", turn=2, role="user")
        user_pack = self.runtime.prepare_context(
            "I mentioned a show as an example. What was it?", self.scope,
            semantic_mode="episode_evidence", token_budget=4096,
        )
        self.assertTrue(any("Doc Martin" in item["text"] for item in user_pack["episodic_candidates"]))

    def test_episode_evidence_preserves_table_coordinates_and_named_list_items(self):
        self.observation("What was Admon's rotation on Sunday?", turn=1, role="user")
        self.observation(
            "|  | 8 am - 4 pm (Day Shift) | 12 pm - 8 pm |\n"
            "| --- | --- | --- |\n"
            "| Sunday | Admon | Magdy |",
            turn=2, role="assistant",
        )
        pack = self.runtime.prepare_context(
            "Can you remind me what was Admon's rotation on Sunday?", self.scope,
            semantic_mode="episode_evidence", token_budget=4096,
        )
        text = pack["episodic_candidates"][0]["text"]
        self.assertIn("Sunday", text)
        self.assertIn("8 am - 4 pm", text)
        self.assertIn("Admon", text)

    def test_episode_evidence_keeps_numeric_and_named_list_spans(self):
        self.observation("Can you recommend a restaurant?", turn=1, role="user")
        self.observation(
            "Here are options:\n1. First Place\n2. Miss Bee Providore\n3. Third Place",
            turn=2, role="assistant",
        )
        pack = self.runtime.prepare_context(
            "Can you remind me what was the name of the restaurant?", self.scope,
            semantic_mode="episode_evidence", token_budget=4096,
        )
        self.assertIn("Miss Bee Providore", pack["episodic_candidates"][0]["text"])

    def test_episode_evidence_does_not_penalize_answer_after_preamble(self):
        self.observation("Create an influencer marketing plan.", turn=1, role="user")
        self.observation(
            "Sure, here's a detailed plan. Budget:\n* Influencer marketing: $2,000",
            turn=2, role="assistant",
        )
        pack = self.runtime.prepare_context(
            "Can you remind me how much was allocated for influencer marketing?", self.scope,
            semantic_mode="episode_evidence", token_budget=4096,
        )
        self.assertIn("2,000", pack["episodic_candidates"][0]["text"])

    def test_hard_budget_limits_context_candidates_after_selection(self):
        for index in range(12):
            self.observation(
                f"Answer {index}: the project uses SQLite and has a detailed historical explanation.",
                session=f"s{index}", turn=1, role="assistant",
            )
        pack = self.runtime.prepare_context(
            "What did you mention about the project?", self.scope,
            semantic_mode="episode_evidence", token_budget=4096,
            options={"budget_mode": "hard"}, limit=50,
        )
        self.assertLessEqual(pack["token_estimate"], 4096)
        self.assertLessEqual(len(pack.get("episodic_candidates", [])), len(pack["selected"]))

    def test_hard_budget_prefers_typed_answer_unit_over_numeric_distractor(self):
        plan = build_query_plan("What is the phone number of the Speyer tourism board?")
        target = {"record_type": "answer_candidate", "record_id": "phone", "session_id": "target",
                  "speech_act": "assistant_answer", "score": 100, "unit_score": 20,
                  "answer_units": [{"kind": "phone_number", "value": "+49 (0) 62 32 / 14 23 - 0"}],
                  "facets": [{"kind": "phone_number", "value": "+49 (0) 62 32 / 14 23 - 0"}]}
        distractor = {"record_type": "answer_candidate", "record_id": "years", "session_id": "distractor",
                      "speech_act": "assistant_recommendation", "score": 9000, "unit_score": 9000,
                      "answer_units": [{"kind": "numeric_value", "value": "2014"}],
                      "facets": [{"kind": "numeric_value", "value": "2014"}]}
        selected = _select_hard_episode_anchors([distractor, target], 1, plan)
        self.assertEqual(selected[0]["record_id"], "phone")

    def test_hard_budget_uses_answer_unit_text_for_table_cell(self):
        item = {"record_type": "answer_candidate", "record_id": "table", "text": "Admon | Sunday",
                "answer_units": [{"kind": "table_coordinate", "value": "8 am - 4 pm (Day Shift)"}]}
        compact = _compact_hard_episodic_candidate(item)
        self.assertIn("8 am - 4 pm", compact["text"])
        self.assertIn("Admon", compact["text"])

    def test_observation_fallback_is_reference_only(self):
        observation = self.observation("The deployment uses SQLite for durable memory.")
        pack = self.runtime.prepare_context(
            "Which database does the deployment use?", self.scope,
            semantic_mode="episode_evidence", token_budget=4096,
        )
        fallback = next(item for item in pack["selected"] if item["record_type"] == "observation")
        self.assertEqual(fallback["area"], "REFERENCE")
        self.assertFalse(fallback["state_eligible"])
        self.assertIn(observation.id, fallback["source_ids"])

    def test_observation_only_candidates_do_not_form_claim_conflicts(self):
        for index in range(8):
            self.observation(
                f"The deployment uses SQLite for durable memory and test note {index}.",
                session=f"observation-session-{index}",
                turn=index + 1,
            )
        pack = self.runtime.prepare_context(
            "Which database does the deployment use?", self.scope,
            semantic_mode="episode_evidence", token_budget=4096,
        )
        self.assertGreater(pack["retrieval_count"], 0)
        self.assertTrue(pack["selected"])
        self.assertTrue(all(item["record_type"] == "observation" for item in pack["selected"]))
        self.assertEqual(pack["conflict_count"], 0)
        self.assertEqual(pack["conflicts"], [])

    def test_unbounded_diagnostic_mode_keeps_retrieved_evidence(self):
        for index in range(8):
            self.observation(
                f"The deployment uses SQLite for durable memory and diagnostic note {index}.",
                session=f"diagnostic-session-{index}",
                turn=index + 1,
            )
        pack = self.runtime.prepare_context(
            "Which database does the deployment use?", self.scope,
            semantic_mode="episode_evidence", token_budget=1,
            options={"budget_mode": "unbounded_diagnostic"},
        )
        self.assertEqual(pack["budget_mode"], "unbounded_diagnostic")
        self.assertFalse(pack["budget_overflow"])
        self.assertGreater(pack["retrieval_count"], 0)
        self.assertGreater(pack["token_estimate"], 1)

    def test_claim_and_observation_hits_do_not_duplicate_source_record(self):
        observation = self.observation("The project uses pytest for tests.")
        self.claim("claim-pytest", "pytest", predicate="project_test_runner", source_ids=[observation.id])
        pack = self.runtime.prepare_context(
            "What test runner does the project use?", self.scope,
            semantic_mode="episode_evidence", token_budget=4096,
        )
        source_hits = [item for item in pack["selected"] if observation.id in item.get("source_ids", [])]
        self.assertEqual(len(source_hits), 1)

    def test_claim_relation_is_idempotent_and_validated(self):
        old = self.claim("old", "MongoDB")
        new = self.claim("new", "SQLite", family=old.assertion_family_key)
        relation = ClaimRelation("r1", new.claim_id, old.claim_id, "supersedes", old.assertion_family_key)
        self.runtime.add_claim_relation(relation)
        self.runtime.add_claim_relation(relation)
        self.assertEqual(len(self.runtime.store.list_claim_relations()), 1)
        with self.assertRaises(ValueError):
            ClaimRelation("bad", "old", "old", "supersedes", old.assertion_family_key)
        with self.assertRaises(ValueError):
            ClaimRelation("bad-type", "new", "old", "supports", old.assertion_family_key)

    def test_claim_relation_rejects_missing_cross_repository_and_cycle(self):
        old = self.claim("old", "one")
        new = self.claim("new", "two", family=old.assertion_family_key)
        with self.assertRaises(KeyError):
            self.runtime.add_claim_relation(ClaimRelation("missing", "new", "absent", "supersedes", old.assertion_family_key))
        other_repo = "repo-other"
        other_claim = Claim("other", "project", "project_status", "three", Scope(other_repo), [], assertion_family_key=old.assertion_family_key)
        self.runtime.store.commit_claim(other_claim)
        with self.assertRaises(ValueError):
            self.runtime.add_claim_relation(ClaimRelation("cross", "new", "other", "supersedes", old.assertion_family_key))
        self.runtime.add_claim_relation(ClaimRelation("r-old-new", "new", "old", "supersedes", old.assertion_family_key))
        with self.assertRaises(ValueError):
            self.runtime.add_claim_relation(ClaimRelation("r-new-old", "old", "new", "supersedes", old.assertion_family_key))

    def test_relation_effective_at_and_state_closure(self):
        old = self.claim("old", "MongoDB")
        new = self.claim("new", "SQLite", family=old.assertion_family_key)
        self.runtime.add_claim_relation(ClaimRelation(
            "future-relation", new.claim_id, old.claim_id, "supersedes", old.assertion_family_key,
            effective_at="2026-01-01T00:00:00+00:00",
        ))
        closure, trace = self.runtime.store.claim_state_closure([old.claim_id], as_of="2025-01-01T00:00:00+00:00")
        self.assertEqual({item.claim_id for item in closure}, {old.claim_id, new.claim_id})
        self.assertFalse(trace["closure_truncated"])
        historical, _ = resolve_claim_state(closure, as_of="2025-01-01T00:00:00+00:00", relations=[])
        self.assertEqual({item.claim_id for item in historical}, {old.claim_id, new.claim_id})
        current, _ = resolve_claim_state(
            closure, as_of="2026-02-01T00:00:00+00:00",
            relations=self.runtime.store.list_claim_relations(as_of="2026-02-01T00:00:00+00:00"),
        )
        self.assertEqual([item.claim_id for item in current], [new.claim_id])

    def test_retract_relation_removes_target_from_current_context(self):
        observation = self.observation("The project uses SQLite.")
        claim = self.claim("sqlite", "SQLite", predicate="database", source_ids=[observation.id])
        retractor = self.claim("retractor", "withdrawal", predicate="database", family=claim.assertion_family_key)
        self.runtime.add_claim_relation(ClaimRelation(
            "retract-sqlite", retractor.claim_id, claim.claim_id, "retracts", claim.assertion_family_key,
        ))
        pack = self.runtime.prepare_context("Which database does the project use?", self.scope, semantic_mode="episode_evidence", token_budget=4096)
        self.assertFalse(any(item.get("claim_id") == claim.claim_id for item in pack["selected"]))
        self.assertEqual(self.runtime.store.get_claim(claim.claim_id).lifecycle_status, "deleted")

    def test_relation_round_trip_preserves_paths_and_rebuild_state(self):
        old = self.claim("old", {"city": "Paris"}, predicate="address")
        new = self.claim("new", {"city": "Berlin", "country": "DE"}, predicate="address", family=old.assertion_family_key)
        relation = ClaimRelation(
            "refine-address", new.claim_id, old.claim_id, "refines", old.assertion_family_key,
            covered_paths=["address.city", "address.country"], metadata={"reason": "partial_update"},
        )
        self.runtime.add_claim_relation(relation)
        saved = self.runtime.store.list_claim_relations()[0]
        self.assertEqual(saved.covered_paths, ["address.city", "address.country"])
        self.assertEqual(saved.metadata["reason"], "partial_update")
        result = self.runtime.rebuild_state()
        self.assertEqual(result["relation_count"], 1)

    def test_small_episode_budgets_use_compressed_pack_without_overflow(self):
        self.observation("A long historical observation about project memory and testing.")
        for budget in (256, 512, 1024):
            pack = self.runtime.prepare_context("project memory testing", self.scope,
                                                semantic_mode="episode_evidence", token_budget=budget)
            self.assertLessEqual(pack["token_estimate"], budget)
            self.assertFalse(pack["budget_overflow"])
            self.assertIn("budget_compressed", pack)

    def test_hard_budget_keeps_answer_evidence_unit(self):
        candidate = {
            "record_type": "answer_candidate", "record_id": "candidate:phone",
            "candidate_id": "candidate:phone", "observation_id": "obs:phone",
            "source_ids": ["obs:phone"], "episode_id": "episode:phone",
            "session_id": "session:phone", "turn_id": "turn:phone", "turn_index": 1,
            "role": "assistant", "speech_act": "assistant_answer", "answer_shape": "entity",
            "score": 100, "answer_span": {"start": 0, "end": 14, "text": "+1 555 0100"},
            "answer_units": [{"kind": "attribute_value", "value": "+1 555 0100",
                              "text": "+1 555 0100", "source_span": {"text": "+1 555 0100"}}],
            "source_span": [{"kind": "attribute_value", "text": "+1 555 0100"}],
            "provenance": {"episode_id": "episode:phone", "turn_id": "turn:phone"},
        }
        pack = _minimal_budget_pack(
            {"schema_version": "semantic-context-pack.v0.6", "semantic_mode": "episode_evidence",
             "query": "What is the phone number?", "token_budget": 512,
             "query_plan": {"answer_shape": "entity", "fields": ["phone_number"],
                             "requested_slots": [{"slot_id": "phone", "field": "phone_number"}],
                             "requested_count": None}, "episode_bundles": []},
            [], [candidate], [], [], ConservativeTokenEstimator(),
        )
        text = "\n".join(item.get("text", "") for item in pack.get("episodic_candidates", []))
        self.assertIn("+1 555 0100", text)

    def test_temporal_context_orders_by_event_time(self):
        first = self.observation("The first event happened.", observed_at="2026-08-30T00:00:00+00:00")
        second = self.observation("The second event happened.", observed_at="2026-08-01T00:00:00+00:00", session="s2")
        self.claim("event-1", "first", predicate="event", source_ids=[first.id], event_time="2026-08-30T00:00:00+00:00")
        self.claim("event-2", "second", predicate="event", source_ids=[second.id], event_time="2026-08-01T00:00:00+00:00")
        pack = self.runtime.prepare_context("event when", self.scope, as_of="2026-08-31", semantic_mode="episode_evidence", token_budget=4096)
        event_ids = [item["claim_id"] for item in pack["selected"] if item.get("claim_id") in {"event-1", "event-2"}]
        self.assertEqual(event_ids, ["event-2", "event-1"])

    def test_state_closure_reports_truncation(self):
        seed = self.claim("seed", "value-0", family="family-large")
        for index in range(1, 5):
            self.claim(f"value-{index}", f"value-{index}", family=seed.assertion_family_key)
        closure, trace = self.runtime.store.claim_state_closure([seed.claim_id], max_claims=2)
        self.assertEqual(len(closure), 2)
        self.assertTrue(trace["closure_truncated"])

    def test_different_assertion_families_are_not_conflicts(self):
        self.claim("first", "Alice", predicate="owner", subject="project")
        self.claim("second", "Bob", predicate="owner", subject="repository")
        pack = self.runtime.prepare_context("owner", self.scope, semantic_mode="episode_evidence", token_budget=4096)
        self.assertFalse(pack["conflicts"])

    def test_aggregation_hints_keep_object_sources(self):
        first = self.observation("Bought 2 cameras.", session="s1")
        second = self.observation("Bought 3 cameras.", session="s2")
        self.claim("buy-2", 2, predicate="purchase_count", source_ids=[first.id], subject="self")
        self.claim("buy-3", 3, predicate="purchase_count", source_ids=[second.id], subject="self")
        pack = self.runtime.prepare_context("How many cameras did I buy in total?", self.scope, semantic_mode="episode_evidence", token_budget=4096)
        hints = pack["aggregation_hints"]
        self.assertEqual(hints["operator"], "sum")
        self.assertTrue(hints["requires_external_calculation"])
        self.assertTrue(any(group["source_values"] for group in hints["groups"]))

    def test_preference_source_separates_assistant_suggestion(self):
        user = self.observation("I prefer Fujifilm cameras.", role="user")
        assistant = self.observation("You may prefer Seattle cameras.", role="assistant", session="s2")
        self.claim("user-pref", "Fujifilm", predicate="user_code_preference", source_ids=[user.id], subject="user")
        self.claim("assistant-pref", "Seattle", predicate="user_code_preference", source_ids=[assistant.id], subject="assistant")
        pack = self.runtime.prepare_context("What camera do I prefer?", self.scope, semantic_mode="episode_evidence", token_budget=4096)
        sources = {item.get("object"): item.get("preference_source") for item in pack["selected"] if item.get("claim_id")}
        self.assertEqual(sources.get("Fujifilm"), "explicit_user_preference")
        self.assertNotEqual(sources.get("Seattle"), "explicit_user_preference")

    def test_requested_object_coverage_does_not_substitute_similar_object(self):
        observation = self.observation("Hawaii took 7 days.")
        self.claim("hawaii", "7 days", predicate="trip_duration", source_ids=[observation.id], subject="Hawaii")
        pack = self.runtime.prepare_context("How long was my trip to Seattle?", self.scope, semantic_mode="episode_evidence", token_budget=4096)
        slots = [slot for slot in pack["unknowns"] if slot.get("entity") == "Seattle"]
        self.assertTrue(slots)
        self.assertIn(slots[0]["status"], {"unknown", "partially_known"})

    def test_episode_pack_has_hard_budget_and_consistent_views(self):
        for index in range(8):
            self.observation("A long historical observation about project memory and testing.", session=f"s{index}", turn=index)
        pack = self.runtime.prepare_context("project memory testing", self.scope, semantic_mode="episode_evidence", token_budget=1024)
        self.assertLessEqual(pack["token_estimate"], 1024)
        self.assertFalse(pack["budget_overflow"])
        self.assertEqual(pack["omitted_summary"]["decision_completeness"], 1.0)
        self.assertEqual(len(pack["selected"]), len(pack["memory_view"]))
        self.assertEqual(len(pack["selected"]), len(pack["evidence_view"]))

    def test_legacy_items_and_shadow_contract_remain(self):
        observation = self.observation("The project uses pytest.")
        self.claim("pytest", "pytest", predicate="project_test_runner", source_ids=[observation.id])
        legacy = self.runtime.prepare_context("project test runner", self.scope, semantic_mode="legacy_claim_text", token_budget=4096)
        self.assertEqual(legacy["schema_version"], "semantic-context-pack.v0.6")
        self.assertTrue(all("statement" in item and "text" in item for item in legacy["items"]))
        self.assertEqual(self.runtime.shadow("project test runner", self.scope, token_budget=4096)["schema_version"], "semantic-context-pack.v0.5")

    def test_generic_adapter_query_is_agent_neutral(self):
        adapter = GenericMemoryAdapter(self.runtime)
        result = adapter.ingest({
            "type": "observation", "content": "The service runs on Python.", "role": "user",
            "session_id": "generic-1", "turn_index": 3,
            "observed_at": "2026-08-31T00:00:00+00:00",
            "scope": {"repository": self.repository}, "metadata": {"source": "test"},
        })
        self.assertTrue(result["accepted"])
        pack = adapter.query({
            "type": "query", "query": "What runtime does the service use?",
            "scope": {"repository": self.repository}, "semantic_mode": "episode_evidence",
            "token_budget": 4096, "limit": 10,
        })
        self.assertEqual(pack["schema_version"], "semantic-context-pack.v0.6")

    def test_rebuild_all_derived_preserves_result(self):
        observation = self.observation("The project uses pytest.")
        self.claim("pytest", "pytest", predicate="project_test_runner", source_ids=[observation.id])
        before = self.runtime.prepare_context("project test runner", self.scope, semantic_mode="episode_evidence", token_budget=4096)
        self.runtime.rebuild_all_derived()
        after = self.runtime.prepare_context("project test runner", self.scope, semantic_mode="episode_evidence", token_budget=4096)
        self.assertEqual(before["selected"], after["selected"])
        self.assertEqual(before["unknowns"], after["unknowns"])
        self.assertEqual(before["conflicts"], after["conflicts"])

    def test_budget_estimator_counts_unicode_more_conservatively(self):
        estimator = ConservativeTokenEstimator()
        ascii_cost = estimator.estimate(json.dumps({"text": "hello"}, ensure_ascii=False))
        unicode_cost = estimator.estimate(json.dumps({"text": "你好，记忆"}, ensure_ascii=False))
        self.assertGreater(unicode_cost, ascii_cost)


if __name__ == "__main__":
    unittest.main()
