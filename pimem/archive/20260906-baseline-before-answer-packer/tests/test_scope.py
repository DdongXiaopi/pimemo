import tempfile
import unittest
from pathlib import Path

from pimem.core.models import Scope
from pimem.runtime import MemoryRuntime


class ScopeTests(unittest.TestCase):
    def test_repository_scope_isolation(self):
        with tempfile.TemporaryDirectory() as tempdir:
            runtime = MemoryRuntime(str(Path(tempdir) / "memory.sqlite3"))
            repo_a = runtime.init_repository(str(Path(tempdir) / "a"))
            repo_b = runtime.init_repository(str(Path(tempdir) / "b"))
            obs = runtime.observe(content="pytest -q tests", source_type="pi_run_test", source_ref="a", scope=Scope(repo_a), idempotency_key="a")
            candidate = runtime.propose_claims(obs.id)[0]
            runtime.commit_claim(candidate["candidate_id"])
            result = runtime.recall("pytest tests", Scope(repo_b))
            self.assertEqual(result["items"], [])

