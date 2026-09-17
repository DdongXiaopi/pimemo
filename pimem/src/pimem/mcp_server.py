"""Zero-dependency MCP server exposing pimem over stdio JSON-RPC 2.0.

The server is intentionally dependency-free: pimem ships with
``dependencies = []`` and this module keeps that property. It implements just
enough of the Model Context Protocol to be usable from any MCP client:

    initialize / notifications/* / tools/list / tools/call / ping

All protocol traffic uses newline-delimited JSON on stdin/stdout, so nothing
else may write to stdout. Diagnostics go to stderr only.
"""

from __future__ import annotations

import io
import json
import os
import sys
from typing import Any, Callable, Dict, List, Optional

from pimem.adapters.generic import GenericMemoryAdapter
from pimem.runtime import MemoryRuntime

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "pimem"
SERVER_VERSION = "0.1.0"

DEFAULT_QUERY_TOKEN_BUDGET = 2048
DEFAULT_QUERY_LIMIT = 50

JSONRPC_PARSE_ERROR = -32700
JSONRPC_INVALID_REQUEST = -32600
JSONRPC_METHOD_NOT_FOUND = -32601
JSONRPC_INVALID_PARAMS = -32602
JSONRPC_INTERNAL_ERROR = -32603


# --------------------------------------------------------------------------
# tool implementations
# --------------------------------------------------------------------------

def _require(args: Dict[str, Any], key: str) -> Any:
    value = args.get(key)
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError("%s is required" % key)
    return value


def _scope_args(args: Dict[str, Any]) -> Dict[str, Any]:
    scope: Dict[str, Any] = {"repository": _require(args, "repository")}
    for key in ("branch", "service", "module", "file", "session", "task"):
        if args.get(key) is not None:
            scope[key] = args[key]
    return scope


def tool_ingest(runtime: MemoryRuntime, args: Dict[str, Any]) -> Dict[str, Any]:
    """Store one observation. Mirrors ``GenericMemoryAdapter.ingest``."""
    adapter = GenericMemoryAdapter(runtime)
    payload: Dict[str, Any] = {
        "type": "observation",
        "content": _require(args, "content"),
        "scope": _scope_args(args),
    }
    for key in ("session_id", "turn_index", "role", "source_type", "source_ref",
                "observed_at", "idempotency_key", "trace_id", "metadata"):
        if args.get(key) is not None:
            payload[key] = args[key]
    return adapter.ingest(payload)


def tool_query(runtime: MemoryRuntime, args: Dict[str, Any]) -> Dict[str, Any]:
    """Build a context pack. Mirrors ``GenericMemoryAdapter.query``."""
    adapter = GenericMemoryAdapter(runtime)
    payload: Dict[str, Any] = {
        "type": "query",
        "query": _require(args, "query"),
        "scope": _scope_args(args),
        "token_budget": int(args.get("token_budget", DEFAULT_QUERY_TOKEN_BUDGET)),
        "limit": int(args.get("limit", DEFAULT_QUERY_LIMIT)),
    }
    for key in ("as_of", "semantic_mode", "options"):
        if args.get(key) is not None:
            payload[key] = args[key]
    return adapter.query(payload)


def tool_forget(runtime: MemoryRuntime, args: Dict[str, Any]) -> Dict[str, Any]:
    policy = str(args.get("policy") or "local_runtime")
    target = _require(args, "target")
    runtime.forget(target, policy)
    return {"forgotten": target, "policy": policy}


TOOLS: List[Dict[str, Any]] = [
    {
        "name": "pimem_ingest",
        "description": "Store a memory observation (a turn, an event, a fact) for later recall.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Observation text to remember."},
                "repository": {"type": "string", "description": "Repository / project identifier."},
                "branch": {"type": "string"},
                "service": {"type": "string"},
                "module": {"type": "string"},
                "file": {"type": "string"},
                "session": {"type": "string"},
                "task": {"type": "string"},
                "session_id": {"type": "string", "description": "Conversation or session identifier."},
                "turn_index": {"type": "integer"},
                "role": {"type": "string", "description": "user / assistant / tool."},
                "source_type": {"type": "string"},
                "source_ref": {"type": "string"},
                "observed_at": {"type": "string", "description": "ISO-8601 timestamp."},
                "idempotency_key": {"type": "string"},
                "trace_id": {"type": "string"},
                "metadata": {"type": "object"},
            },
            "required": ["content", "repository"],
            "additionalProperties": False,
        },
    },
    {
        "name": "pimem_query",
        "description": "Build a budgeted, provenance-tracked context pack for a query.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "repository": {"type": "string"},
                "branch": {"type": "string"},
                "service": {"type": "string"},
                "module": {"type": "string"},
                "file": {"type": "string"},
                "session": {"type": "string"},
                "task": {"type": "string"},
                "token_budget": {"type": "integer", "description": "Context token budget.", "default": DEFAULT_QUERY_TOKEN_BUDGET},
                "limit": {"type": "integer", "default": DEFAULT_QUERY_LIMIT},
                "as_of": {"type": "string", "description": "ISO-8601 point-in-time cutoff."},
                "semantic_mode": {"type": "string", "default": "legacy_claim_text"},
                "options": {"type": "object"},
            },
            "required": ["query", "repository"],
            "additionalProperties": False,
        },
    },
    {
        "name": "pimem_forget",
        "description": "Forget a claim by id under an explicit deletion policy.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "target": {"type": "string", "description": "Claim id to forget."},
                "policy": {"type": "string", "default": "local_runtime"},
            },
            "required": ["target"],
            "additionalProperties": False,
        },
    },
]

_HANDLERS: Dict[str, Callable[[MemoryRuntime, Dict[str, Any]], Dict[str, Any]]] = {
    "pimem_ingest": tool_ingest,
    "pimem_query": tool_query,
    "pimem_forget": tool_forget,
}


def call_tool(runtime: MemoryRuntime, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    handler = _HANDLERS.get(name)
    if handler is None:
        raise KeyError(name)
    return handler(runtime, args)


# --------------------------------------------------------------------------
# JSON-RPC plumbing
# --------------------------------------------------------------------------

def _error(request_id: Any, code: int, message: str) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def _result(request_id: Any, result: Any) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _tool_result(payload: Any) -> Dict[str, Any]:
    return {
        "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False, sort_keys=True)}],
        "isError": False,
    }


def handle_request(runtime: MemoryRuntime, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Handle one JSON-RPC message. Returns None for notifications."""
    if not isinstance(request, dict):
        return _error(None, JSONRPC_INVALID_REQUEST, "request must be an object")
    method = request.get("method")
    request_id = request.get("id")
    if not isinstance(method, str):
        return _error(request_id, JSONRPC_INVALID_REQUEST, "method is required")
    is_notification = "id" not in request

    if method == "initialize":
        return _result(request_id, {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
        })
    if method == "ping":
        return _result(request_id, {})
    if method == "tools/list":
        return _result(request_id, {"tools": TOOLS})
    if method == "tools/call":
        params = request.get("params") or {}
        name = params.get("name")
        args = params.get("arguments") or {}
        if not isinstance(name, str) or not isinstance(args, dict):
            return _error(request_id, JSONRPC_INVALID_PARAMS, "tools/call requires name and arguments")
        try:
            payload = call_tool(runtime, name, args)
        except KeyError:
            return _error(request_id, JSONRPC_INVALID_PARAMS, "unknown tool: %s" % name)
        except ValueError as exc:
            # Expected domain errors (bad scope, sensitive content, unknown claim).
            return _result(request_id, {
                "content": [{"type": "text", "text": str(exc)}],
                "isError": True,
            })
        except Exception as exc:  # pragma: no cover - defensive
            return _error(request_id, JSONRPC_INTERNAL_ERROR, "%s: %s" % (type(exc).__name__, exc))
        return _result(request_id, _tool_result(payload))

    if is_notification:
        return None
    return _error(request_id, JSONRPC_METHOD_NOT_FOUND, "method not found: %s" % method)


def resolve_db_path(cli_value: Optional[str] = None) -> str:
    """Resolve the SQLite path.

    MCP clients control the working directory, so a CWD-relative default would
    silently create empty databases everywhere. The path must be explicit.
    """
    if cli_value:
        return cli_value
    env_value = os.environ.get("PIMEM_DB")
    if env_value:
        return env_value
    raise ValueError("PIMEM_DB is required for the MCP server (or pass --db)")


def serve(runtime: MemoryRuntime, stdin: io.TextIOBase, stdout: io.TextIOBase) -> int:
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            stdout.write(json.dumps(_error(None, JSONRPC_PARSE_ERROR, str(exc))) + "\n")
            stdout.flush()
            continue
        response = handle_request(runtime, request)
        if response is None:
            continue
        stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        stdout.flush()
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    db_path = None
    if "--db" in args:
        index = args.index("--db")
        if index + 1 >= len(args):
            sys.stderr.write("--db requires a value\n")
            return 2
        db_path = args[index + 1]
    try:
        resolved = resolve_db_path(db_path)
    except ValueError as exc:
        sys.stderr.write("%s\n" % exc)
        return 2
    runtime = MemoryRuntime(resolved)
    stdin = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8", newline="\n")
    stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", newline="\n", write_through=True)
    return serve(runtime, stdin, stdout)


if __name__ == "__main__":
    raise SystemExit(main())
