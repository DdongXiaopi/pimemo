from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def replay(path: str) -> Dict[str, Any]:
    """Replay a recorded T0 JSON state without touching the real filesystem."""
    source = Path(path)
    data = json.loads(source.read_text(encoding="utf-8"))
    return {
        "case": source.stem,
        "level": "T0",
        "status": "pass",
        "observations": len(data.get("observations", [])),
        "claims": len(data.get("claims", [])),
        "evidence": len(data.get("evidence", [])),
    }

