import tempfile
import unittest
from pathlib import Path

from pimem.core.models import Claim, Scope
from pimem.core.state_machine import classify_relation, transition
from pimem.runtime import MemoryRuntime


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.runtime = MemoryRuntime(str(Path(self.tempdir.name) / "memory.sqlite3"))
        self.repository = self.runtime.init_repository(str(Path(self.tempdir.name) / "repo"))
        self.scope = Scope(self.repository)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_observation_to_claim_to_explain(self):
        observation = self.runtime.observe(content="pytest -q runs tests", source_type="pi_run_test", source_ref="t1", scope=self.scope, idempotency_key="t1")
        candidate = self.runtime.propose_claims(observation.id)[0]
        claim = self.runtime.commit_claim(candidate["candidate_id"])
        explanation = self.runtime.explain(claim.claim_id)
        self.assertEqual(explanation["claim"]["source_observation_ids"], [observation.id])
        self.assertEqual(explanation["evidence"][0]["observation_id"], observation.id)

    def test_conflict_relations_are_classified_without_auto_replacement(self):
        existing = Claim("existing", "project", "project_test_command", "pytest -q", self.scope, [])
        duplicate = Claim("duplicate", "project", "project_test_command", "pytest -q", self.scope, [])
        replacement = Claim("replacement", "project", "project_test_command", "node --test tests", self.scope, [])
        future = Claim("future", "project", "project_test_command", "node --test tests", self.scope, [], valid_from="2027-01-01")
        scoped_existing = Claim("scoped-existing", "project", "project_test_command", "pytest -q", Scope(self.repository, module="api"), [])
        scoped_incoming = Claim("scoped-incoming", "project", "project_test_command", "node --test tests", Scope(self.repository, module="worker"), [])

        self.assertEqual(classify_relation(existing, duplicate), "duplicate")
        self.assertEqual(classify_relation(existing, replacement), "full_replace")
        self.assertEqual(classify_relation(existing, future), "temporal_change")
        self.assertEqual(classify_relation(scoped_existing, scoped_incoming), "scope_split")
        self.assertEqual(transition("supported", "active", "context_allowed", "challenge"), ("disputed", "active", "reference_only"))
        self.assertEqual(transition("supported", "active", "context_allowed", "partial_replace"), ("supported", "active", "context_allowed"))
