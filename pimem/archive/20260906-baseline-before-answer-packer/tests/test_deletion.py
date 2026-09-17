import tempfile
import unittest
from pathlib import Path

from pimem.core.models import Scope
from pimem.runtime import MemoryRuntime


class DeletionTests(unittest.TestCase):
    def test_forget_removes_recall(self):
        with tempfile.TemporaryDirectory() as tempdir:
            runtime = MemoryRuntime(str(Path(tempdir) / "memory.sqlite3"))
            repo = runtime.init_repository(str(Path(tempdir) / "repo"))
            scope = Scope(repo)
            obs = runtime.observe(content="pytest -q tests", source_type="pi_run_test", source_ref="t", scope=scope, idempotency_key="t")
            candidate = runtime.propose_claims(obs.id)[0]
            claim = runtime.commit_claim(candidate["candidate_id"])
            runtime.forget(claim.claim_id)
            self.assertEqual(runtime.recall("pytest", scope)["items"], [])
            self.assertEqual(runtime.explain(claim.claim_id)["claim"]["use_policy"], "blocked")

