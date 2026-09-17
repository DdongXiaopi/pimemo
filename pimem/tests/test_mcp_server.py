import io
import json
import os
import tempfile
import unittest
from pathlib import Path

from pimem.adapters.generic import GenericMemoryAdapter
from pimem.mcp_server import (
    PROTOCOL_VERSION,
    TOOLS,
    handle_request,
    resolve_db_path,
    serve,
    tool_ingest,
    tool_query,
)
from pimem.runtime import MemoryRuntime

OBSERVATIONS = [
    {"content": "we run tests with pytest -q", "session_id": "s1", "turn_index": 0, "role": "user"},
    {"content": "the release branch is cut on Fridays", "session_id": "s1", "turn_index": 1, "role": "assistant"},
    {"content": "deploys go through the ops pipeline", "session_id": "s2", "turn_index": 0, "role": "user"},
]


def _seed(runtime, repository="repo-mcp"):
    for item in OBSERVATIONS:
        tool_ingest(runtime, dict(item, repository=repository))
    return runtime


class McpServerTests(unittest.TestCase):
    def test_initialize_and_tools_list(self):
        with tempfile.TemporaryDirectory() as tempdir:
            runtime = MemoryRuntime(str(Path(tempdir) / "memory.sqlite3"))
            response = handle_request(runtime, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
            self.assertEqual(response["result"]["protocolVersion"], PROTOCOL_VERSION)
            self.assertIn("tools", response["result"]["capabilities"])

            listed = handle_request(runtime, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
            names = [tool["name"] for tool in listed["result"]["tools"]]
            self.assertEqual(names, ["pimem_ingest", "pimem_query", "pimem_forget"])
            for tool in listed["result"]["tools"]:
                self.assertEqual(tool["inputSchema"]["type"], "object")
                for required in tool["inputSchema"]["required"]:
                    self.assertIn(required, tool["inputSchema"]["properties"])

    def test_tools_call_ingest_and_query(self):
        with tempfile.TemporaryDirectory() as tempdir:
            runtime = MemoryRuntime(str(Path(tempdir) / "memory.sqlite3"))
            ingested = handle_request(runtime, {
                "jsonrpc": "2.0", "id": 3, "method": "tools/call",
                "params": {"name": "pimem_ingest",
                           "arguments": {"content": "we run tests with pytest -q", "repository": "repo-mcp"}},
            })
            self.assertFalse(ingested["result"]["isError"])
            payload = json.loads(ingested["result"]["content"][0]["text"])
            self.assertTrue(payload["accepted"])
            self.assertEqual(payload["observation"]["content"], "we run tests with pytest -q")

            queried = handle_request(runtime, {
                "jsonrpc": "2.0", "id": 4, "method": "tools/call",
                "params": {"name": "pimem_query",
                           "arguments": {"query": "how do we run tests", "repository": "repo-mcp", "token_budget": 512}},
            })
            pack = json.loads(queried["result"]["content"][0]["text"])
            self.assertEqual(pack["schema_version"], "semantic-context-pack.v0.6")
            self.assertEqual(pack["token_budget"], 512)

    def test_missing_required_argument_is_tool_error(self):
        with tempfile.TemporaryDirectory() as tempdir:
            runtime = MemoryRuntime(str(Path(tempdir) / "memory.sqlite3"))
            response = handle_request(runtime, {
                "jsonrpc": "2.0", "id": 5, "method": "tools/call",
                "params": {"name": "pimem_ingest", "arguments": {"content": "orphan"}},
            })
            self.assertTrue(response["result"]["isError"])
            self.assertIn("repository is required", response["result"]["content"][0]["text"])

    def test_unknown_tool_is_protocol_error(self):
        with tempfile.TemporaryDirectory() as tempdir:
            runtime = MemoryRuntime(str(Path(tempdir) / "memory.sqlite3"))
            response = handle_request(runtime, {
                "jsonrpc": "2.0", "id": 6, "method": "tools/call",
                "params": {"name": "pimem_nope", "arguments": {}},
            })
            self.assertEqual(response["error"]["code"], -32602)

    def test_unknown_method_and_notifications(self):
        with tempfile.TemporaryDirectory() as tempdir:
            runtime = MemoryRuntime(str(Path(tempdir) / "memory.sqlite3"))
            unknown = handle_request(runtime, {"jsonrpc": "2.0", "id": 7, "method": "does/not/exist"})
            self.assertEqual(unknown["error"]["code"], -32601)
            self.assertIsNone(handle_request(runtime, {"jsonrpc": "2.0", "method": "notifications/initialized"}))

    def test_serve_handles_stream_and_parse_errors(self):
        with tempfile.TemporaryDirectory() as tempdir:
            runtime = MemoryRuntime(str(Path(tempdir) / "memory.sqlite3"))
            stdin = io.StringIO(
                '{"jsonrpc":"2.0","id":1,"method":"tools/list"}\n'
                "\n"
                "{not json}\n"
            )
            stdout = io.StringIO()
            serve(runtime, stdin, stdout)
            lines = [line for line in stdout.getvalue().splitlines() if line.strip()]
            self.assertEqual(len(lines), 2)
            self.assertEqual(json.loads(lines[0])["id"], 1)
            self.assertEqual(json.loads(lines[1])["error"]["code"], -32700)

    def test_resolve_db_path_requires_explicit_path(self):
        previous = os.environ.pop("PIMEM_DB", None)
        try:
            with self.assertRaises(ValueError):
                resolve_db_path(None)
            self.assertEqual(resolve_db_path("/tmp/explicit.sqlite3"), "/tmp/explicit.sqlite3")
            os.environ["PIMEM_DB"] = "/tmp/from-env.sqlite3"
            self.assertEqual(resolve_db_path(None), "/tmp/from-env.sqlite3")
            self.assertEqual(resolve_db_path("/tmp/override.sqlite3"), "/tmp/override.sqlite3")
        finally:
            if previous is None:
                os.environ.pop("PIMEM_DB", None)
            else:
                os.environ["PIMEM_DB"] = previous


class McpAdapterEquivalenceTests(unittest.TestCase):
    """The MCP surface must stay byte-identical to the library surface."""

    def test_query_matches_generic_adapter(self):
        with tempfile.TemporaryDirectory() as tempdir:
            runtime = _seed(MemoryRuntime(str(Path(tempdir) / "memory.sqlite3")))
            adapter = GenericMemoryAdapter(runtime)
            via_adapter = adapter.query({"type": "query", "query": "how do we run tests",
                                         "scope": {"repository": "repo-mcp"},
                                         "token_budget": 1024, "limit": 50})
            via_mcp = tool_query(runtime, {"query": "how do we run tests", "repository": "repo-mcp",
                                           "token_budget": 1024, "limit": 50})
            self.assertEqual(_stable(via_adapter), _stable(via_mcp))

    def test_ingest_matches_generic_adapter(self):
        payload = {"content": "the canary command is make check", "session_id": "s9",
                   "turn_index": 3, "role": "assistant", "observed_at": "2026-09-13T00:00:00+00:00"}
        with tempfile.TemporaryDirectory() as tempdir:
            left = MemoryRuntime(str(Path(tempdir) / "left.sqlite3"))
            right = MemoryRuntime(str(Path(tempdir) / "right.sqlite3"))
            via_adapter = GenericMemoryAdapter(left).ingest(
                {"type": "observation", "scope": {"repository": "repo-mcp"}, **payload})
            via_mcp = tool_ingest(right, dict(payload, repository="repo-mcp"))
            self.assertEqual(_observation_facts(via_adapter["observation"]),
                             _observation_facts(via_mcp["observation"]))


# content_hash is hashed over the whole pack including latency_ms, so it is not
# reproducible across calls; both are excluded from the equivalence comparison.
NON_DETERMINISTIC_KEYS = ("latency_ms", "content_hash")


def _stable(pack):
    return json.dumps({key: value for key, value in pack.items() if key not in NON_DETERMINISTIC_KEYS},
                      sort_keys=True, default=str)


def _observation_facts(observation):
    return {key: observation[key] for key in
            ("content", "source_type", "source_ref", "metadata", "observed_at", "scope", "sensitivity")}


if __name__ == "__main__":
    unittest.main()
