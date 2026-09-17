import tempfile
import unittest
from pathlib import Path

from pimem.core.models import Scope
from pimem.runtime import MemoryRuntime


class IdempotencyTests(unittest.TestCase):
    def test_duplicate_event_and_claim(self):
        with tempfile.TemporaryDirectory() as tempdir:
            runtime = MemoryRuntime(str(Path(tempdir) / "memory.sqlite3"))
            repo = runtime.init_repository(str(Path(tempdir) / "repo"))
            first = runtime.ingest_event({"event_type": "run_test", "scope": {"repository": repo}, "payload": {"command": "pytest -q"}, "idempotency_key": "same"})
            second = runtime.ingest_event({"event_type": "run_test", "scope": {"repository": repo}, "payload": {"command": "pytest -q"}, "idempotency_key": "same"})
            self.assertEqual(first["observation"].id, second["observation"].id)
            candidate = runtime.propose_claims(first["observation"].id)[0]
            runtime.commit_claim(candidate["candidate_id"])
            duplicate_candidate = runtime.propose_claims(first["observation"].id)[0]
            duplicate = runtime.commit_claim(duplicate_candidate["candidate_id"])
            self.assertEqual(len(runtime.store.list_claims()), 1)
            self.assertEqual(duplicate.predicate, "project_test_command")

