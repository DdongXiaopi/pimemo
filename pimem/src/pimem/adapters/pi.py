from __future__ import annotations

import json
import os
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from pimem.core.models import Scope


ALLOWLIST = {"user_message", "read_file", "search_text", "run_test", "tool_execution_end"}


class PiAdapter:
    def __init__(self, runtime: Any, *, pi_command: str = "pi") -> None:
        self.runtime = runtime
        self.pi_command = pi_command

    def capabilities(self) -> Dict[str, Any]:
        executable = shutil.which(self.pi_command)
        version = os.environ.get("PI_VERSION")
        if executable and not version:
            try:
                completed = subprocess.run([executable, "--version"], capture_output=True, text=True, timeout=2)
                version = (completed.stdout or completed.stderr).strip().splitlines()[0] if completed.returncode == 0 else None
            except (OSError, subprocess.SubprocessError):
                version = None
        status = "supported" if executable else "unknown"
        return {
            "adapter": "pi",
            "pi_command": self.pi_command,
            "pi_executable": executable,
            "pi_version": version,
            "checks": {
                "extension_load": status,
                "memory_probe_tool": "unknown",
                "session_events": "unknown",
                "tool_execution_events": "unknown",
                "before_agent_start": "unknown",
                "context_event": "unknown",
                "compact_events": "unknown",
                "failure_non_blocking": "unknown",
                "shared_extension_process": "unknown",
            },
            "note": "Only locally observable executable/version facts are reported; unknown is not an architecture assumption.",
        }

    def current_scope(self, event: Dict[str, Any]) -> Scope:
        raw = event.get("scope") or {}
        repository = raw.get("repository") or event.get("repository")
        if not repository:
            raise ValueError("Pi event requires scope.repository")
        return Scope(repository=repository, branch=raw.get("branch"), service=raw.get("service"),
                     module=raw.get("module"), file=raw.get("file"), session=raw.get("session"), task=raw.get("task"))

    def capture(self, event: Dict[str, Any]) -> Dict[str, Any]:
        event_type = event.get("event_type") or event.get("type")
        if event_type not in ALLOWLIST:
            return {"accepted": False, "reason": "event_not_allowlisted", "event_type": event_type}
        scope = self.current_scope(event)
        payload = event.get("payload") or {}
        content = self._summary(event_type, payload, event.get("content"))
        if not content:
            return {"accepted": False, "reason": "capture_rejected_empty", "event_type": event_type}
        try:
            observation = self.runtime.observe(
                content=content,
                source_type="pi_" + event_type,
                source_ref=str(event.get("tool_call_id") or event.get("event_id") or uuid.uuid4().hex),
                scope=scope,
                observed_at=event.get("timestamp") or datetime.now(timezone.utc).isoformat(),
                idempotency_key=event.get("idempotency_key") or event.get("event_id") or uuid.uuid4().hex,
                trace_id=event.get("trace_id"),
                metadata={"session_id": event.get("session_id"), "turn_id": event.get("turn_id"),
                          "tool_call_id": event.get("tool_call_id"), "event_type": event_type,
                          "payload": {key: value for key, value in payload.items() if key in {"command", "exit_code", "path", "query", "summary", "error"}}},
            )
        except Exception as error:
            return {"accepted": False, "reason": "runtime_unavailable", "warning": str(error), "event_type": event_type}
        return {"accepted": True, "observation": observation}

    def current_turn(self, event: Dict[str, Any]) -> Optional[str]:
        return event.get("turn_id")

    def context_mode(self) -> str:
        return "none"

    def report_status(self, *, pack_id: Optional[str] = None, retrieval_count: int = 0,
                      injected_token_estimate: int = 0, warnings: Optional[list[str]] = None,
                      required_verification: Optional[list[str]] = None) -> Dict[str, Any]:
        return {
            "mode": self.context_mode(),
            "pack_id": pack_id,
            "memory_available": True,
            "retrieval_count": retrieval_count,
            "injected_token_estimate": injected_token_estimate,
            "warnings": warnings or [],
            "required_verification": required_verification or [],
        }

    @staticmethod
    def _summary(event_type: str, payload: Dict[str, Any], content: Optional[str]) -> str:
        if content:
            return str(content)
        if event_type == "user_message":
            return str(payload.get("text") or payload.get("message") or "")
        if event_type in {"read_file", "search_text", "run_test", "tool_execution_end"}:
            pieces = [str(payload.get(key)) for key in ("summary", "command", "query", "path", "error", "exit_code") if payload.get(key) is not None]
            return " | ".join(pieces)
        return ""
