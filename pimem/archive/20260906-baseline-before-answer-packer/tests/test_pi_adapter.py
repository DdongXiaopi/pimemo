import tempfile
import unittest
from pathlib import Path

from pimem.runtime import MemoryRuntime


class PiAdapterTests(unittest.TestCase):
    def test_allowlist_and_non_blocking_failure(self):
        with tempfile.TemporaryDirectory() as tempdir:
            runtime = MemoryRuntime(str(Path(tempdir) / "memory.sqlite3"))
            repository = runtime.init_repository(str(Path(tempdir) / "repo"))
            skipped = runtime.ingest_event({"event_type": "agent_start", "scope": {"repository": repository}})
            self.assertFalse(skipped["accepted"])
            rejected = runtime.ingest_event({"event_type": "user_message", "scope": {"repository": repository}, "content": "api_key=abcdefghijklmnop"})
            self.assertFalse(rejected["accepted"])
            self.assertEqual(rejected["reason"], "runtime_unavailable")
            self.assertEqual(runtime.pi.context_mode(), "none")
            self.assertFalse(runtime.pi.report_status()["memory_available"] is False)

    def test_runtime_failure_degrades_to_no_memory(self):
        with tempfile.TemporaryDirectory() as tempdir:
            runtime = MemoryRuntime(str(Path(tempdir) / "memory.sqlite3"))
            repository = runtime.init_repository(str(Path(tempdir) / "repo"))

            def fail_observe(**_kwargs):
                raise OSError("simulated storage outage")

            runtime.observe = fail_observe
            result = runtime.ingest_event({
                "event_type": "run_test",
                "scope": {"repository": repository},
                "payload": {"command": "pytest -q"},
                "idempotency_key": "outage-1",
            })
            self.assertFalse(result["accepted"])
            self.assertEqual(result["reason"], "runtime_unavailable")
