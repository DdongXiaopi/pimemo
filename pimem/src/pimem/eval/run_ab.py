from __future__ import annotations

from typing import Any, Dict, Iterable, List

from .metrics import summarize


def compare(disabled_rows: Iterable[Dict[str, Any]], enabled_rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    disabled = list(disabled_rows)
    enabled = list(enabled_rows)
    return {
        "memory_disabled": summarize(disabled),
        "memory_enabled": summarize(enabled),
        "success_rate_delta": summarize(enabled)["success_rate"] - summarize(disabled)["success_rate"],
    }

