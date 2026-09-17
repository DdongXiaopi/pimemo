import tempfile
import unittest
from pathlib import Path

from pimem.core.models import Scope
from pimem.runtime import MemoryRuntime


class PartialPatchTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.runtime = MemoryRuntime(str(Path(self.tempdir.name) / "memory.sqlite3"))
        self.repository = self.runtime.init_repository(str(Path(self.tempdir.name) / "repo"))
        self.scope = Scope(self.repository)
        observation = self.runtime.observe(
            content="pytest -q runs the project tests",
            source_type="pi_run_test",
            source_ref="patch-test",
            scope=self.scope,
            idempotency_key="patch-test",
            metadata={"payload": {"command": "pytest -q"}},
        )
        candidate = self.runtime.propose_claims(observation.id)[0]
        self.claim = self.runtime.commit_claim(candidate["candidate_id"])
        self.observation_id = observation.id

    def tearDown(self):
        self.tempdir.cleanup()

    def test_patch_requires_explicit_scope_old_value_and_revision(self):
        updated = self.runtime.patch_claim(
            self.claim.claim_id,
            base_revision=1,
            covered_scope={"repository": self.repository},
            changes={"object": "node --test tests"},
            expected_old={"object": "pytest -q"},
            idempotency_key="patch-once",
        )

        self.assertEqual(updated.revision, 2)
        self.assertEqual(updated.object, "node --test tests")
        self.assertEqual(updated.source_observation_ids, [self.observation_id])
        self.assertEqual(self.runtime.recall("node test tests", self.scope)["items"][0]["claim_id"], self.claim.claim_id)
        self.assertEqual(self.runtime.recall("pytest", self.scope)["items"], [])

        retry = self.runtime.patch_claim(
            self.claim.claim_id,
            base_revision=1,
            covered_scope={"repository": self.repository},
            changes={"object": "node --test tests"},
            expected_old={"object": "pytest -q"},
            idempotency_key="patch-once",
        )
        self.assertEqual(retry.revision, 2)

    def test_patch_rejects_stale_revision_old_value_and_scope(self):
        kwargs = {
            "base_revision": 1,
            "covered_scope": {"repository": self.repository},
            "changes": {"object": "node --test tests"},
            "expected_old": {"object": "wrong"},
        }
        with self.assertRaisesRegex(ValueError, "expected_old_mismatch"):
            self.runtime.patch_claim(self.claim.claim_id, **kwargs)
        with self.assertRaisesRegex(ValueError, "covered_scope_mismatch"):
            self.runtime.patch_claim(self.claim.claim_id, **{**kwargs, "expected_old": {"object": "pytest -q"}, "covered_scope": {"repository": "repo-other"}})
        self.runtime.patch_claim(self.claim.claim_id, **{**kwargs, "expected_old": {"object": "pytest -q"}})
        with self.assertRaisesRegex(ValueError, "base_revision_mismatch"):
            self.runtime.patch_claim(self.claim.claim_id, **{**kwargs, "expected_old": {"object": "node --test tests"}})

    def test_patch_then_forget_and_rebuild_removes_derived_recall(self):
        self.runtime.patch_claim(
            self.claim.claim_id,
            base_revision=1,
            covered_scope={"repository": self.repository},
            changes={"object": "node --test tests"},
            expected_old={"object": "pytest -q"},
        )
        self.runtime.forget(self.claim.claim_id)
        self.runtime.rebuild_indexes()
        self.assertEqual(self.runtime.recall("node test tests", self.scope)["items"], [])


if __name__ == "__main__":
    unittest.main()
