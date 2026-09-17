"""Offline re-renderer: build a SYNTHESIS context_text from an existing chain file.

Design (deterministic, no LLM) -- the "Memory Answer Synthesis" stage that moves
cognitive load off the weak answering model and into the packing layer:

  1. HEADER: question + as_of (query-conditioning so the model knows what to look for).
  2. PRIMARY EVIDENCE: the full content of target_session.turns -- the conversation
     session that actually contains the answer. 84% of gold evidence lives here;
     the old render buried it among 166 flat digest lines.
  3. SYSTEM ANSWER CANDIDATE: the retriever's best matched turn(s), shown so the
     model sees what the memory system flagged (and can correct it when it picked
     the wrong item from a list).
  4. SUPPORTING MEMORY: remaining episodic_candidates as compact digest lines,
     sorted by confidence, budget-capped so total stays under the 8192 budget.

Writes a copy of each chain file with only context_pack.context_text replaced,
into OUT_DIR, preserving every other field so the online runner keeps working.
"""
import os, glob, json, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "pimem", "src"))
from pimem.retrieval.render import estimate_tokens

TOKEN_BUDGET = 7800
DEFAULT_IN_DIR = "strat120r-fix1-20260915"
DEFAULT_OUT_DIR = "_packs_synth/strat120r-fix1-20260915"


def _short_date(value):
    text = str(value or "").strip()
    if not text:
        return "unknown"
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            import datetime as dt
            return dt.datetime.strptime(text[:19], fmt).strftime("%Y/%m/%d")
        except Exception:
            pass
    return text[:10]


def _clean(text, cap):
    return " ".join(str(text or "").split())[:cap]


def render_context_synth(chain: dict) -> str:
    cp = chain.get("context_pack") or {}
    qp = chain.get("query_plan") or cp.get("query_plan") or {}
    question = qp.get("query") or chain.get("question") or ""
    as_of = cp.get("as_of")
    lines = []
    lines.append(f"# Recalled memory (as of {as_of})" if as_of else "# Recalled memory")
    if question:
        lines.append(f"# Question to answer from memory: {question}")

    # --- 2. PRIMARY EVIDENCE: full gold-session transcript ---
    tgt = chain.get("target_session") or {}
    tgt_turns = tgt.get("turns") or []
    shown_obs = set()
    if tgt_turns:
        lines.append("")
        lines.append("## PRIMARY EVIDENCE (conversation session that contains the answer)")
        for t in tgt_turns:
            if not isinstance(t, dict):
                continue
            role = str(t.get("role") or "user")
            content = _clean(t.get("content"), 1400)
            if not content:
                continue
            date = _short_date(t.get("observed_at"))
            sid = str(t.get("session_id") or "")
            lines.append(f"[{date}] {role}: {content}")
            shown_obs.add(str(t.get("observation_id") or ""))

    # --- 3. SYSTEM ANSWER CANDIDATE (retriever's best match) ---
    at = chain.get("answer_turn") or {}
    at_turns = at.get("turns") or []
    if at_turns:
        lines.append("")
        lines.append("## SYSTEM ANSWER CANDIDATE (retriever's best-matched turn)")
        for t in at_turns:
            if not isinstance(t, dict):
                continue
            oid = str(t.get("observation_id") or "")
            if oid in shown_obs:
                continue
            role = str(t.get("role") or "assistant")
            content = _clean(t.get("content"), 1400)
            if not content:
                continue
            date = _short_date(t.get("observed_at"))
            lines.append(f"[{date}] {role}: {content}")
            shown_obs.add(oid)

    # --- 4. SUPPORTING MEMORY: compact remaining candidates, budget-capped ---
    ecs = cp.get("episodic_candidates") or []
    used = estimate_tokens("\n".join(lines))
    if ecs:
        lines.append("")
        lines.append("## SUPPORTING MEMORY (other recalled context, most relevant first)")
        for c in sorted(ecs, key=lambda c: -(float(c.get("confidence") or 0) or 0)):
            oid = str(c.get("observation_id") or "")
            if oid in shown_obs:
                continue
            text = str(c.get("evidence_text") or c.get("text") or "")
            if not text.strip():
                continue
            snippet = _clean(text, 200)
            line = f"- {snippet}"
            cost = estimate_tokens(line)
            if used + cost > TOKEN_BUDGET:
                break
            lines.append(line)
            used += cost

    return "\n".join(lines)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("in_dir", nargs="?", default=DEFAULT_IN_DIR)
    ap.add_argument("out_dir", nargs="?", default=DEFAULT_OUT_DIR)
    args = ap.parse_args()
    chains = sorted(glob.glob(os.path.join(args.in_dir, "**", "chain-*.json"), recursive=True))
    os.makedirs(args.out_dir, exist_ok=True)
    total = 0
    token_sum = 0
    for src in chains:
        d = json.load(open(src, encoding="utf-8"))
        new_ctx = render_context_synth(d)
        d.setdefault("context_pack", {})["context_text"] = new_ctx
        # keep a provenance marker
        d["context_pack"]["render_stage"] = "synthesis_v1"
        rel = os.path.relpath(src, args.in_dir)
        dst = os.path.join(args.out_dir, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        json.dump(d, open(dst, "w", encoding="utf-8"), ensure_ascii=False)
        total += 1
        token_sum += estimate_tokens(new_ctx)
    print(f"re-rendered {total} chains -> {args.out_dir}")
    print(f"avg synthesized context tokens: {token_sum/max(1,total):.0f} (budget {TOKEN_BUDGET})")


if __name__ == "__main__":
    main()
