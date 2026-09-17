import tempfile
import unittest
from pathlib import Path

from pimem.core.models import Claim, Scope
from pimem.runtime import MemoryRuntime


class ContextCompilerTests(unittest.TestCase):
    def _runtime_with_claims(self):
        tempdir = tempfile.TemporaryDirectory()
        runtime = MemoryRuntime(str(Path(tempdir.name) / "memory.sqlite3"))
        repository = runtime.init_repository(str(Path(tempdir.name) / "repo"))
        scope = Scope(repository)
        first_observation = runtime.observe(content="pytest command", source_type="pi_run_test", source_ref="t1", scope=scope)
        second_observation = runtime.observe(content="npm command", source_type="pi_run_test", source_ref="t2", scope=scope)
        runtime.store.commit_claim(Claim(
            claim_id="claim_pytest", subject="project", predicate="project_test_command", object="pytest -q",
            scope=scope, source_observation_ids=[first_observation.id], truth_status="supported",
            lifecycle_status="active", use_policy="context_allowed",
        ))
        runtime.store.commit_claim(Claim(
            claim_id="claim_npm", subject="project", predicate="project_test_command", object="npm test",
            scope=scope, source_observation_ids=[second_observation.id], truth_status="supported",
            lifecycle_status="active", use_policy="verification_required",
        ))
        return tempdir, runtime, scope

    def test_shadow_preserves_conflicts_and_traceability(self):
        tempdir, runtime, scope = self._runtime_with_claims()
        self.addCleanup(tempdir.cleanup)

        pack = runtime.shadow("project test command", scope, token_budget=1, level="L1")

        self.assertEqual(pack["schema_version"], "semantic-context-pack.v0.5")
        self.assertFalse(pack["injected"])
        self.assertEqual(pack["memory_mode"], "retrieval_shadow")
        self.assertEqual(pack["selected_count"], 2)
        self.assertEqual(pack["required_verification"], ["claim_npm", "claim_pytest"])
        self.assertEqual(len(pack["conflicts"]), 1)
        self.assertIn("budget_overflow", pack["warnings"])
        self.assertTrue(all(item["area"] == "VERIFY" for item in pack["selected"]))
        self.assertTrue(all(item["evidence"] for item in pack["selected"]))

    def test_shadow_omits_lower_priority_items_when_budget_is_exhausted(self):
        tempdir, runtime, scope = self._runtime_with_claims()
        self.addCleanup(tempdir.cleanup)
        runtime.store.commit_claim(Claim(
            claim_id="claim_runtime", subject="project", predicate="project_runtime", object="python",
            scope=scope, source_observation_ids=[], truth_status="supported",
            lifecycle_status="active", use_policy="context_allowed",
        ))

        pack = runtime.shadow("project python test command", scope, token_budget=1, level="L0")

        self.assertTrue(any(item["claim_id"] == "claim_runtime" for item in pack["omitted"]))
        self.assertTrue(all(item["claim_id"] != "claim_runtime" for item in pack["selected"]))


if __name__ == "__main__":
    unittest.main()
