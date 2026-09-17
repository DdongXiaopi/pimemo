"""Canonical rendering of a context pack into the text a model actually reads.

Why this exists
---------------
The hard budget used to be charged against ``json.dumps(pack)``.  That number is
dominated by scaffolding -- repeated key names, id lists, offset dicts, facet
arrays -- that never reaches the model, because consumers render the pack into
natural language first.  Measured on LongMemEval: a pack that reported ~7.3k
tokens against an 8192 budget rendered to ~1.2k tokens of readable text, i.e.
~83% of the budget was spent on structure the model cannot use.

``render_context`` is the single canonical rendering.  The budget can be charged
against it (``budget_unit = rendered_context``) so that "8192 token budget"
means 8192 tokens of memory actually delivered to the model.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List


def estimate_tokens(text: str) -> int:
    """Same conservative char heuristic as ``budget.ConservativeTokenEstimator``."""
    if not text:
        return 0
    ascii_count = sum(1 for char in text if ord(char) < 128)
    non_ascii = len(text) - ascii_count
    return max(1, math.ceil(ascii_count / 4.0) + math.ceil(non_ascii / 2.0))


_ISO = re.compile(r"^(\d{4})-(\d{2})-(\d{2})")


def _short_date(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "unknown-date"
    match = _ISO.match(text)
    if match:
        return f"{match.group(1)}/{match.group(2)}/{match.group(3)}"
    return text[:16]


def _bundle_date(bundle: Dict[str, Any]) -> str:
    for key in ("question_turns", "answer_turns", "related_turns"):
        for turn in bundle.get(key) or []:
            if turn.get("observed_at"):
                return _short_date(turn["observed_at"])
    return "unknown-date"


def _clean(value: Any, cap: int) -> str:
    return " ".join(str(value or "").split())[:cap]


def _turns(bundle: Dict[str, Any], *, question_cap: int = 1, answer_cap: int = 2,
           related_cap: int = 2) -> List[Dict[str, Any]]:
    """Question -> answer -> nearby context, deduped and re-sorted by turn index.

    The bundle can carry up to eight related turns; rendering them all would let
    one session eat most of the budget. Under a hard budget the number of
    *distinct* sessions represented matters more than the depth of any one.
    """
    picked: List[Dict[str, Any]] = []
    seen = set()

    def take(items: Any, cap: int) -> None:
        for turn in list(items or [])[:cap]:
            if not isinstance(turn, dict):
                continue
            key = str(turn.get("turn_id") or id(turn))
            if key in seen:
                continue
            seen.add(key)
            picked.append(turn)

    take(bundle.get("question_turns"), question_cap)
    take(bundle.get("answer_turns"), answer_cap)
    take(bundle.get("related_turns"), related_cap)
    return sorted(picked, key=lambda turn: int(turn.get("turn_index") or 0))


def _turn_char_cap(turn: Dict[str, Any], bundle: Dict[str, Any]) -> int:
    question_ids = {str(t.get("turn_id")) for t in (bundle.get("question_turns") or [])}
    answer_ids = {str(t.get("turn_id")) for t in (bundle.get("answer_turns") or [])}
    turn_id = str(turn.get("turn_id"))
    if turn_id in question_ids:
        return 320
    if turn_id in answer_ids:
        return 600
    return 220


def render_context(pack: Dict[str, Any], *, turn_char_cap: int = 600,
                   digest_char_cap: int = 400) -> str:
    """Render a context pack into the compact natural-language memory block."""
    lines: List[str] = []
    as_of = pack.get("as_of")
    lines.append(f"# Recalled memory (as of {as_of})" if as_of else "# Recalled memory")

    bundles = list(pack.get("episode_bundles") or pack.get("evidence_bundles") or [])
    session_label: Dict[str, str] = {}
    for bundle in bundles:
        sid = str(bundle.get("session_id") or "")
        if sid and sid not in session_label:
            session_label[sid] = f"#{len(session_label) + 1}"

    for bundle in bundles:
        sid = str(bundle.get("session_id") or "")
        label = session_label.get(sid, "#?")
        date = _bundle_date(bundle)
        rendered_turns = 0
        for turn in _turns(bundle):
            role = str(turn.get("role") or "user")
            content = _clean(turn.get("content"), _turn_char_cap(turn, bundle))
            if not content:
                continue
            lines.append(f"[{label} {date}] {role}: {content}")
            rendered_turns += 1
        if not rendered_turns:
            lines.append(f"[{label} {date}] (session retained, turn text unavailable)")

    for item in pack.get("episodic_candidates") or []:
        sid = str(item.get("session_id") or "")
        if not item.get("digest"):
            if sid in session_label:
                continue  # already rendered from the session bundle
            text = _clean(item.get("evidence_text") or item.get("text"), digest_char_cap)
            if not text:
                continue
            if sid and sid not in session_label:
                session_label[sid] = f"#{len(session_label) + 1}"
            lines.append(f"[{session_label.get(sid, '#?')} {_short_date(item.get('observed_at'))}] {text}")
            continue
        text = _clean(item.get("evidence_text") or item.get("text"), digest_char_cap)
        if not text:
            continue
        if sid and sid not in session_label:
            session_label[sid] = f"#{len(session_label) + 1}"
        lines.append(f"[{session_label.get(sid, '#?')} {_short_date(item.get('observed_at'))}] {text}")

    content_lines = len(lines) - 1
    if content_lines == 0:
        # Safety net.  Aggregation / state closures can admit raw records that
        # carry no bundle and no answer candidate, which would otherwise render
        # as a bare list of "(session retained, turn text unavailable)" shells.
        # Only used when the episodic tier produced nothing at all, so it can
        # never perturb a pack that already renders evidence.
        for item in pack.get("selected") or pack.get("items") or []:
            if not isinstance(item, dict):
                continue
            sid = str(item.get("session_id") or "")
            if sid in session_label:
                continue
            text = _clean(item.get("text") or item.get("statement")
                          or item.get("evidence_text"), digest_char_cap)
            if not text:
                continue
            if sid and sid not in session_label:
                session_label[sid] = f"#{len(session_label) + 1}"
            lines.append(
                f"[{session_label.get(sid, '#?')} "
                f"{_short_date(item.get('observed_at') or item.get('event_time'))}] {text}")

    unknowns = [slot for slot in (pack.get("unknowns") or [])
                if str(slot.get("status") or "") != "known"]
    if unknowns:
        lines.append("[unresolved] " + "; ".join(
            _clean(slot.get("object") or slot.get("slot_id"), 60) for slot in unknowns[:8]))

    return "\n".join(lines)


def rendered_estimate(pack: Dict[str, Any], **kwargs: Any) -> int:
    return estimate_tokens(render_context(pack, **kwargs))
