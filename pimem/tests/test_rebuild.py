import tempfile
import unittest
from pathlib import Path

from pimem.core.models import Scope
from pimem.runtime import MemoryRuntime


class RebuildTests(unittest.TestCase):
    def test_fts_rebuild(self):
        with tempfile.TemporaryDirectory() as tempdir:
            runtime = MemoryRuntime(str(Path(tempdir) / "memory.sqlite3"))
            repo = runtime.init_repository(str(Path(tempdir) / "repo"))
            obs = runtime.observe(content="pytest -q tests", source_type="pi_run_test", source_ref="t", scope=Scope(repo), idempotency_key="t")
            candidate = runtime.propose_claims(obs.id)[0]
            runtime.commit_claim(candidate["candidate_id"])
            runtime.store.rebuild_indexes()
            result = runtime.recall("pytest", Scope(repo))
            self.assertEqual(len(result["items"]), 1)

    def test_fts_survives_runtime_reopen(self):
        with tempfile.TemporaryDirectory() as tempdir:
            db = str(Path(tempdir) / "memory.sqlite3")
            first = MemoryRuntime(db)
            repo = first.init_repository(str(Path(tempdir) / "repo"))
            obs = first.observe(content="pytest -q tests", source_type="pi_run_test", source_ref="t", scope=Scope(repo), idempotency_key="t")
            candidate = first.propose_claims(obs.id)[0]
            first.commit_claim(candidate["candidate_id"])

            reopened = MemoryRuntime(db)
            result = reopened.recall("pytest", Scope(repo))
            self.assertEqual(len(result["items"]), 1)
