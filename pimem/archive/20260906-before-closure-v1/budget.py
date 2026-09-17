from __future__ import annotations

import json
import math
from typing import Any, Dict, Protocol


class TokenEstimator(Protocol):
    version: str

    def estimate(self, serialized: str) -> int:
        ...


class ConservativeTokenEstimator:
    version = "conservative-char-v1"

    def estimate(self, serialized: str) -> int:
        if not serialized:
            return 0
        ascii_count = sum(1 for char in serialized if ord(char) < 128)
        non_ascii_count = len(serialized) - ascii_count
        return max(1, math.ceil(ascii_count / 4.0) + math.ceil(non_ascii_count / 2.0))


def estimate_payload(value: Any, estimator: TokenEstimator | None = None) -> int:
    active = estimator or ConservativeTokenEstimator()
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return active.estimate(serialized)


def compact_summary(items: list[Dict[str, Any]]) -> Dict[str, Any]:
    by_reason: Dict[str, int] = {}
    by_type: Dict[str, int] = {}
    for item in items:
        reason = str(item.get("reason") or "unspecified")
        record_type = str(item.get("record_type") or "unknown")
        by_reason[reason] = by_reason.get(reason, 0) + 1
        by_type[record_type] = by_type.get(record_type, 0) + 1
    return {
        "count": len(items),
        "by_reason": by_reason,
        "by_record_type": by_type,
        "compressed": bool(items),
    }
