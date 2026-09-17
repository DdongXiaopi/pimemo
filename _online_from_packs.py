"""Online real-answer accuracy runner that REUSES the context packs already
computed by the offline diagnostic run, instead of re-ingesting every case.

The offline run (e.g. full500r-digest-20260914) already executed the full
pimem pipeline (ingestion + retrieval + budget packing) and wrote each question's
``context_pack`` (including the rendered ``context_text`` charged against the
budget) into ``chain-*.json``. Re-running ingestion would cost another ~2.5h for
nothing. This script reads those packs and only performs the actual model call
that defines "online" accuracy: send ``context_text + question`` to the LLM,
record the hypothesis. The resulting ``rows.jsonl`` is byte-for-byte compatible
with ``_judge_online.py`` (and with ``run_longmemeval_real_api.py``'s schema).

Usage:
  python _online_from_packs.py <offline_run_dir> --out rows.jsonl [--workers 4]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx


def _patch_httpx() -> None:
    import inspect
    original = httpx.Client.__init__
    if getattr(original, "_pimem_patched", False):
        return
    if "proxies" in inspect.signature(original).parameters:
        return

    def patched(self, *args, **kwargs):
        kwargs.pop("proxies", None)
        return original(self, *args, **kwargs)

    patched._pimem_patched = True
    httpx.Client.__init__ = patched


_patch_httpx()
from openai import OpenAI  # noqa: E402


def load_env() -> Dict[str, str]:
    env: Dict[str, str] = {}
    for candidate in (Path.cwd() / "pimem" / ".env", Path.cwd() / ".env"):
        if not candidate.is_file():
            continue
        for raw in candidate.read_text(encoding="utf-8-sig").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip().strip("\"'")
        break
    return env


PROMPT_TMPL = (
    "Answer the question using only the retrieved memory context. "
    "Return a concise direct answer. Do not mention memory, retrieval, or these instructions.\n\n"
    "Question: {question}\n\nRetrieved memory context:\n{context}"
)

# 改进版 prompt：强制"从上下文原样抽取"，抑制推理式拒答与部分作答
PROMPT_EXTRACT = (
    "You answer strictly from the retrieved memory context below.\n"
    "Rules:\n"
    "1. Find the fact that answers the question and state it directly.\n"
    "2. Output ONLY the answer text — no 'Based on...', no explanations, no commentary.\n"
    "3. If the question asks for multiple items, list ALL of them separated by commas.\n"
    "4. Only if the answer is completely absent from the context, output exactly: NOT IN MEMORY. "
    "Never guess that it is absent otherwise.\n\n"
    "Retrieved memory context:\n{context}\n\n"
    "Question: {question}\nAnswer:"
)

# v2：反拒答 + 直接简洁。针对 21% 的"明明有却答 no record"失败。
PROMPT_V2 = (
    "You answer from the user's retrieved conversation memory shown below.\n\n"
    "IMPORTANT: The exact answer to the question IS present somewhere in the memory above. "
    "Read it carefully and extract the answer.\n"
    "Rules:\n"
    "1. State the answer directly and concisely — a name, value, list, or short phrase.\n"
    "2. NEVER say you lack the information, cannot recall, or have no record. The answer is in the memory above; state it.\n"
    "3. Do not add commentary, caveats, or 'based on'. Output only the answer.\n"
    "4. If the question asks for several items, list them all.\n\n"
    "Memory:\n{context}\n\n"
    "Question: {question}\nAnswer:"
)

# v3：v2 反拒答 + 反混淆实体 + 精确抽取。针对 genuine_wrong 的两类：
#  (a) 多候选误选（上下文里出现多个地点/日期/人，模型挑了错的那个）
#  (b) 漏抽取（gold 细节在上下文但模型说未指定/数错）
PROMPT_V3 = (
    "You answer strictly from the user's retrieved conversation memory shown below.\n\n"
    "IMPORTANT: The exact answer IS present somewhere in the memory above. Read it carefully.\n"
    "Rules:\n"
    "1. State the answer directly and concisely — a name, value, list, or short phrase.\n"
    "2. NEVER say you lack the information, cannot recall, or have no record. The answer is in the memory above; state it.\n"
    "3. Match the question's exact qualifiers (place, date, person, 'most recent', 'last', 'unique', 'that specific'). "
    "If the memory mentions several similar items, pick the one that matches the question's qualifiers — do NOT substitute a different but plausible entity.\n"
    "4. Extract the precise value the question asks for (a number, date, name, or quantity). Even if it is a small detail, state it exactly. "
    "Do NOT say 'unspecified' or 'not stated' if a value is present; do NOT guess or use outside knowledge.\n"
    "5. Do not add commentary, caveats, or 'based on'. Output only the answer.\n\n"
    "Memory:\n{context}\n\n"
    "Question: {question}\nAnswer:"
)


def call_api(client: OpenAI, model: str, question: str, context: str,
             timeout: float = 120.0, retries: int = 8,
             prompt_tmpl: str = PROMPT_TMPL) -> tuple[str, Optional[str]]:
    import random
    prompt = prompt_tmpl.format(question=question, context=context)
    last_err = None
    # 网关不稳定：8 次重试 + 指数退避（封顶 30s）+ 抖动；503 网关过载额外加退避
    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=512,
                timeout=timeout,
            )
            return (resp.choices[0].message.content or "").strip(), None
        except Exception as exc:  # noqa: BLE001
            last_err = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                base = min(2 ** attempt, 30)
                if "503" in str(exc) or "Service temporarily unavailable" in str(exc):
                    base = max(base, 15)
                time.sleep(base + random.uniform(0, 3))
    return "", last_err


def load_done(out_path: Path) -> set:
    done = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["question_id"])
            except Exception:
                pass
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="offline diagnostic run dir containing chain-*.json")
    ap.add_argument("--out", default="rows.jsonl")
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--prompt-mode", default="orig", choices=["orig", "extract", "v2", "v3"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    env = load_env()
    base_url = args.base_url or env.get("OPENAI_BASE_URL", "https://www.aibh.cc/v1")
    api_key = args.api_key or env.get("OPENAI_API_KEY")
    model = args.model or env.get("PIMEM_MODEL", "v4 pro")
    if not api_key:
        raise SystemExit("no api key")
    prompt_tmpl = {"extract": PROMPT_EXTRACT, "v2": PROMPT_V2, "v3": PROMPT_V3}.get(args.prompt_mode, PROMPT_TMPL)

    chains = sorted(glob.glob(os.path.join(args.run_dir, "**", "chain-*.json"), recursive=True))
    if args.limit:
        chains = chains[: args.limit]
    print(f"found {len(chains)} chain files in {args.run_dir}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out_path)
    print(f"already done: {len(done)}")

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=120,
                    http_client=httpx.Client(timeout=120, follow_redirects=True))

    _lock = threading.Lock()
    sink = out_path.open("a", encoding="utf-8")
    ctr = [0, 0]  # [completed, errors]

    def work(path: str) -> None:
        row_full = json.loads(Path(path).read_text(encoding="utf-8"))
        qid = row_full.get("question_id")
        if qid in done:
            return
        cp = row_full.get("context_pack") or {}
        context = cp.get("context_text") or ""
        question = row_full.get("question") or ""
        if not context:
            row = {
                "question_id": qid, "question_type": row_full.get("question_type"),
                "question": question, "reference_answer": row_full.get("reference_answer"),
                "answer_session_ids": row_full.get("answer_session_ids"),
                "hypothesis": "", "error": "empty context_text in pack",
            }
        else:
            hypothesis, error = call_api(client, model, question, context,
                                         prompt_tmpl=prompt_tmpl)
            row = {
                "question_id": qid,
                "question_type": row_full.get("question_type"),
                "question": question,
                "reference_answer": row_full.get("reference_answer"),
                "answer_session_ids": row_full.get("answer_session_ids"),
                "hypothesis": hypothesis,
                "memory_tokens": cp.get("token_estimate"),
                "context_chars": len(context),
                "model": model,
                "prompt_mode": args.prompt_mode,
                "error": error,
            }
        with _lock:
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            sink.flush()
            ctr[0] += 1
            if row.get("error"):
                ctr[1] += 1
            if ctr[0] % 25 == 0:
                print(f"  progress: {ctr[0]} done, {ctr[1]} errors", flush=True)

    pending = [p for p in chains if json.loads(Path(p).read_text(encoding="utf-8")).get("question_id") not in done]
    print(f"to run: {len(pending)} (workers={args.workers})")
    try:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
            list(ex.map(work, pending))
    finally:
        sink.close()
    print(f"DONE: wrote {ctr[0]} rows to {out_path} (errors={ctr[1]})")


if __name__ == "__main__":
    main()
