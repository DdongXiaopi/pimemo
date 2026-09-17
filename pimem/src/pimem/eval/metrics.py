from __future__ import annotations

from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List


def summarize(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(rows)
    latencies = [float(row.get("latency_ms", 0)) for row in rows]
    successes = [bool(row.get("success")) for row in rows]
    return {
        "runs": len(rows),
        "success_rate": mean(successes) if successes else 0.0,
        "latency_ms_mean": mean(latencies) if latencies else 0.0,
        "latency_ms_stddev": pstdev(latencies) if len(latencies) > 1 else 0.0,
        "pollution_count": sum(bool(row.get("pollution")) for row in rows),
    }

