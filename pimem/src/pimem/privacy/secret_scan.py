from __future__ import annotations

import re
from typing import List


PATTERNS = {
    "private_key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "bearer_token": re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{16,}", re.IGNORECASE),
    "api_key": re.compile(r"\b(?:api[_-]?key|secret[_-]?key|access[_-]?token)\s*[:=]\s*['\"]?[A-Za-z0-9._~+/=-]{12,}", re.IGNORECASE),
    "password": re.compile(r"\b(?:password|passwd|pwd)\s*[:=]\s*\S+", re.IGNORECASE),
    "connection_string": re.compile(r"\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis)://[^\s]+", re.IGNORECASE),
    "cloud_key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
}


def scan_secrets(content: str) -> List[str]:
    return [name for name, pattern in PATTERNS.items() if pattern.search(content)]

