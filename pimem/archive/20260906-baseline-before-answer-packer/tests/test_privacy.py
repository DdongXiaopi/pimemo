import tempfile
import unittest
from pathlib import Path

from pimem.core.models import Scope
from pimem.runtime import MemoryRuntime


class PrivacyTests(unittest.TestCase):
    def test_secret_is_rejected_before_storage(self):
        with tempfile.TemporaryDirectory() as tempdir:
            runtime = MemoryRuntime(str(Path(tempdir) / "memory.sqlite3"))
            repo = runtime.init_repository(str(Path(tempdir) / "repo"))
            with self.assertRaises(ValueError):
                runtime.observe(content="api_key=abcdefghijklmnop", source_type="pi_user_message", source_ref="secret", scope=Scope(repo), idempotency_key="secret")
            self.assertEqual(runtime.store.list_claims(), [])

