import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pimem.core.models import Claim, Scope
from pimem.runtime import MemoryRuntime


class TemporalClaimTests(unittest.TestCase):
    def test_future_and_expired_claims_are_not_recalled(self):
        with tempfile.TemporaryDirectory() as tempdir:
            runtime = MemoryRuntime(str(Path(tempdir) / "memory.sqlite3"))
            repository = runtime.init_repository(str(Path(tempdir) / "repo"))
            scope = Scope(repository)
            observation = runtime.observe(content="project command evidence", source_type="pi_run_test", source_ref="time", scope=scope)
            now = datetime.now(timezone.utc)
            future = (now + timedelta(days=1)).isoformat()
            expired = (now - timedelta(days=1)).isoformat()
            runtime.store.commit_claim(Claim(
                claim_id="claim_future", subject="project", predicate="project_test_command", object="future-test",
                scope=scope, source_observation_ids=[observation.id], valid_from=future,
                truth_status="supported", lifecycle_status="active", use_policy="context_allowed",
            ))
            runtime.store.commit_claim(Claim(
                claim_id="claim_expired", subject="project", predicate="project_test_command", object="expired-test",
                scope=scope, source_observation_ids=[observation.id], valid_to=expired,
                truth_status="supported", lifecycle_status="active", use_policy="context_allowed",
            ))
            self.assertEqual(runtime.recall("future-test", scope)["items"], [])
            self.assertEqual(runtime.recall("expired-test", scope)["items"], [])

    def test_patch_validates_time_interval_and_persists_time_fields(self):
        with tempfile.TemporaryDirectory() as tempdir:
            runtime = MemoryRuntime(str(Path(tempdir) / "memory.sqlite3"))
            repository = runtime.init_repository(str(Path(tempdir) / "repo"))
            scope = Scope(repository)
            observation = runtime.observe(content="pytest -q", source_type="pi_run_test", source_ref="time-patch", scope=scope)
            claim = Claim(
                claim_id="claim_time", subject="project", predicate="project_test_command", object="pytest -q",
                scope=scope, source_observation_ids=[observation.id], truth_status="supported",
                lifecycle_status="active", use_policy="context_allowed",
            )
            runtime.store.commit_claim(claim)
            updated = runtime.patch_claim(
                claim.claim_id, base_revision=1, covered_scope={"repository": repository},
                changes={"valid_from": "2026-08-28T00:00:00+00:00", "time_precision": "day"},
                expected_old={"valid_from": None, "time_precision": None},
            )
            self.assertEqual(updated.valid_from, "2026-08-28T00:00:00+00:00")
            self.assertEqual(updated.time_precision, "day")
            with self.assertRaisesRegex(ValueError, "valid_from must not be after valid_to"):
                runtime.patch_claim(
                    claim.claim_id, base_revision=2, covered_scope={"repository": repository},
                    changes={"valid_to": "2026-08-27T00:00:00+00:00"},
                    expected_old={"valid_to": None},
                )


if __name__ == "__main__":
    unittest.main()
