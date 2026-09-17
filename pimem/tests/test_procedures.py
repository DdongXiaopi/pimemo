import tempfile
import unittest
from pathlib import Path

from pimem.core.models import Scope
from pimem.core.procedures import Experience, propose_candidate_procedure, replay_candidate, shadow_candidate
from pimem.runtime import MemoryRuntime


class ProcedureTests(unittest.TestCase):
    def test_candidate_procedure_requires_provenance_and_stays_shadow_only(self):
        with tempfile.TemporaryDirectory() as tempdir:
            runtime = MemoryRuntime(str(Path(tempdir) / "memory.sqlite3"))
            repository = runtime.init_repository(str(Path(tempdir) / "repo"))
            scope = Scope(repository)
            experiences = [
                Experience("exp-1", "T1", "find test command", scope, [{"name": "memory_search"}], {"status": "success"}, ["obs-1"]),
                Experience("exp-2", "T1", "find test command", scope, [{"name": "memory_search"}], {"status": "success"}, ["obs-2"]),
            ]
            procedure = propose_candidate_procedure(
                experiences,
                procedure_id="proc-test-command",
                trigger="user asks for project test command",
                preconditions=["repository scope is known"],
                actions=[{"name": "memory_search", "query": "project test command"}],
                success_criteria=["retrieved command is returned"],
                failure_modes=["memory unavailable"],
                stop_conditions=["retrieval is empty"],
                rollback=["do not change files"],
                scope=scope,
            )

            shadow = shadow_candidate(procedure, experiences)

            self.assertEqual(procedure.status, "candidate")
            self.assertFalse(shadow["active"])
            self.assertEqual(shadow["pass_count"], 2)
            self.assertEqual(shadow["utility_estimate"], 1.0)
            self.assertEqual(shadow["provenance"], ["exp-1", "exp-2"])

    def test_replay_is_read_only_and_rejects_scope_or_failed_outcome(self):
        scope = Scope("repo-a")
        procedure = propose_candidate_procedure(
            [Experience("exp-1", "T1", "goal", scope, [{"tool": "read_file"}], {"status": "success"})],
            procedure_id="proc-read",
            trigger="file inspection",
            preconditions=["repository scope is known"],
            actions=[{"tool": "read_file"}],
            success_criteria=["file read"],
            failure_modes=["file missing"],
            stop_conditions=["path is outside scope"],
            rollback=["no file changes"],
            scope=scope,
            risk_level="S0",
        )
        failed = Experience("exp-2", "T1", "goal", Scope("repo-b"), [{"tool": "read_file"}], {"status": "failed"})

        result = replay_candidate(procedure, failed)

        self.assertEqual(result["status"], "review")
        self.assertTrue(result["read_only"])
        self.assertIn("scope_mismatch", result["failure_reasons"])
        self.assertIn("experience_not_successful", result["failure_reasons"])


if __name__ == "__main__":
    unittest.main()
