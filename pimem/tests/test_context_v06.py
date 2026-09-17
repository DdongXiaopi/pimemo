import tempfile
import unittest
from pathlib import Path

from pimem.adapters.generic import GenericMemoryAdapter
from pimem.core.models import Claim, Scope
from pimem.retrieval.query_plan import build_query_plan
from pimem.runtime import MemoryRuntime


class ContextPackV06Tests(unittest.TestCase):
    def test_query_plan_uses_question_date_for_relative_time(self):
        plan = build_query_plan("What happened four weeks ago?", as_of="2026-08-31")
        self.assertEqual(plan.intent, "temporal")
        self.assertEqual(plan.temporal_anchor, "2026-08-31")
        self.assertEqual(plan.temporal_constraints[0]["target"], "2026-08-03T00:00:00+00:00")

    def test_historical_count_query_keeps_episodic_memory_mode(self):
        plan = build_query_plan(
            "Can you remind me how many subjects were in the study from our previous conversation?"
        )
        self.assertEqual(plan.primary_intent, "count")
        self.assertEqual(plan.memory_mode, "episodic_recall")

    def test_context_pack_exposes_preference_and_unknown_coverage(self):
        with tempfile.TemporaryDirectory() as tempdir:
            runtime = MemoryRuntime(str(Path(tempdir) / "memory.sqlite3"))
            repository = runtime.init_repository(str(Path(tempdir) / "repo"))
            scope = Scope(repository)
            observation = runtime.observe(
                content="I prefer Fujifilm cameras.", source_type="generic", source_ref="preference", scope=scope,
                metadata={"session_id": "session-1", "turn_index": 2, "role": "user"},
            )
            runtime.store.commit_claim(Claim(
                claim_id="preference-1", subject="user", predicate="user_code_preference",
                object="I prefer Fujifilm cameras.", scope=scope, source_observation_ids=[observation.id],
                truth_status="supported", lifecycle_status="active", use_policy="context_allowed",
            ))
            pack = runtime.prepare_context("What camera do I prefer, Fujifilm or Seattle?", scope,
                                           as_of="2026-08-31", token_budget=100)
            self.assertEqual(pack["schema_version"], "semantic-context-pack.v0.6")
            self.assertEqual(pack["query_plan"]["intent"], "preference")
            self.assertTrue(any(item["object"] == "Seattle" and item["status"] == "unknown"
                                for item in pack["unknowns"]))
            self.assertEqual(pack["evidence_bundles"][0]["session_id"], "session-1")

    def test_generic_adapter_does_not_require_pi_event_schema(self):
        with tempfile.TemporaryDirectory() as tempdir:
            runtime = MemoryRuntime(str(Path(tempdir) / "memory.sqlite3"))
            repository = runtime.init_repository(str(Path(tempdir) / "repo"))
            adapter = GenericMemoryAdapter(runtime)
            result = adapter.ingest({
                "type": "observation", "content": "The project uses pytest -q.", "source_ref": "turn-1",
                "scope": {"repository": repository}, "session_id": "s1", "turn_index": 1, "role": "user",
            })
            self.assertTrue(result["accepted"])
            self.assertEqual(result["observation"]["metadata"]["session_id"], "s1")


if __name__ == "__main__":
    unittest.main()
