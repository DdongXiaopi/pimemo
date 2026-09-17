from __future__ import annotations

import re

from pimem.core.policy import CaptureDecision
from .secret_scan import scan_secrets


def sanitize(content: str, *, allow_email: bool = False) -> CaptureDecision:
    if not isinstance(content, str) or not content.strip():
        return CaptureDecision(False, "", "capture_rejected_empty", "unknown")
    findings = scan_secrets(content)
    if findings:
        return CaptureDecision(False, "", f"capture_rejected_sensitive:{','.join(findings)}", "secret")
    sanitized = content.strip()
    if not allow_email:
        sanitized = re.sub(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", "[email]", sanitized)
    sanitized = re.sub(r"(?i)(/Users/|/home/|[A-Z]:\\Users\\)[^\s]+", "[user-path]", sanitized)
    return CaptureDecision(True, sanitized, "allowed", "internal")

