"""Resilient incremental judge with built-in gateway probe + self-healing.

- Loads answers from <in>, prior judgements from <out> (if exists).
- Only judges rows whose judge_correct is still None (incremental + resume).
- Before each batch, probes the gateway with a REAL judge-style request.
  If the gateway is down, sleeps and retries (no quota wasted on blind calls).
- When gateway recovers, runs a concurrent batch; each finished row is
  written back immediately (with a lock). Rows that still fail (None) are
  retried on the next healthy probe round.
- Single process: no file races. Runs until every row is judged.
"""
import argparse
import json
import os
import random
import sys
import threading
import time
from pathlib import Path

import httpx
from openai import OpenAI

ENV_CANDIDATES = [Path.cwd() / "pimem" / ".env", Path.cwd() / ".env"]
JUDGE_PROMPT = (
    "You are grading a factual QA answer.\n\n"
    "Question: {question}\n\n"
    "Reference (gold) answer: {gold}\n\n"
    "System answer to evaluate: {pred}\n\n"
    "Decide if the system answer is CORRECT given the reference. "
    "It is CORRECT if it states the same key facts (names, dates, values, preferences) "
    "as the reference, even if worded differently. A refusal, 'I don't know', or a "
    "contradiction is WRONG. Reply with exactly one word: CORRECT or WRONG."
)


def load_env():
    env = {}
    for cand in ENV_CANDIDATES:
        if cand.is_file():
            for raw in cand.read_text(encoding="utf-8-sig").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"\'')
            break
    return env


def build_client(env):
    return OpenAI(
        api_key=env["OPENAI_API_KEY"],
        base_url=env.get("OPENAI_BASE_URL", "https://www.aibh.cc/v1"),
        timeout=60,
        http_client=httpx.Client(timeout=60, follow_redirects=True),
    )


def probe(client, model, rows):
    """Probe gateway with a real judge-style request on a few samples."""
    samples = rows[:3]
    ok = 0
    for r in samples:
        q = r.get("question", "")
        g = r.get("reference_answer", "")
        p = r.get("hypothesis", "")
        prompt = JUDGE_PROMPT.format(question=q, gold=g, pred=p)
        try:
            client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}],
                temperature=0, max_tokens=24, timeout=40,
            )
            ok += 1
        except Exception:
            pass
    return ok >= 2  # at least 2/3 healthy


def judge_once(client, model, prompt, max_tries=8, giveup_conn=3):
    last = None
    conn_errs = 0
    for attempt in range(1, max_tries + 1):
        try:
            resp = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}],
                temperature=0, max_tokens=24, timeout=30,
            )
            text = (resp.choices[0].message.content or "").strip().upper()
            if "WRONG" in text and "CORRECT" not in text:
                return False, None
            if "CORRECT" in text:
                return True, None
            # ambiguous -> treat as correct if it leans correct
            return ("CORRECT" in text), None
        except Exception as exc:  # noqa: BLE001
            last = f"{type(exc).__name__}: {exc}"
            s = str(exc)
            is_conn = ("503" in s) or ("Connection" in s) or ("Service" in s) or ("timeout" in s.lower())
            if is_conn:
                conn_errs += 1
                if conn_errs >= giveup_conn:
                    # gateway is down right now -> bail fast so the outer
                    # probe loop can wait and retry later instead of burning
                    # ~10 min retrying every single row.
                    return None, last
            base = min(2 ** attempt, 15)
            if is_conn:
                base = max(base, 8)
            time.sleep(base + random.uniform(0, 2))
    return None, last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="deepseek-v4-flash（特价版）")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    # ---- refuse to run under any python other than the project venv ----
    # (a background mechanism keeps respawning copies via D:\Anaconda\python.exe,
    #  which would race on the output file). Only the venv may proceed.
    if ".venv_pimem_online" not in sys.executable.replace("\\", "/"):
        print("ERROR: must run with .venv_pimem_online python; exiting.", flush=True)
        sys.exit(1)
    # ----------------------------------------------------------------

    # ---- single-instance lock via local TCP port (reliable cross-process) ----
    import socket as _socket
    _lock_sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    try:
        _lock_sock.bind(("127.0.0.1", 55999))
        _lock_sock.listen(1)
    except OSError:
        print("ERROR: another instance of _judge_resilient is already running (port lock); exiting.", flush=True)
        sys.exit(1)
    # -------------------------------------------------------------------------

    env = load_env()
    client = build_client(env)

    from concurrent.futures import ThreadPoolExecutor, as_completed
    lock = threading.Lock()

    rows = [json.loads(l) for l in open(args.inp, encoding="utf-8") if l.strip()]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # load prior judgements
    judged = {}
    if out_path.is_file():
        for l in open(out_path, encoding="utf-8"):
            if not l.strip():
                continue
            r = json.loads(l)
            if r.get("question_id") is not None and r.get("judge_correct") is not None:
                judged[r["question_id"]] = r

    def flush():
        with lock:
            with open(out_path, "w", encoding="utf-8") as f:
                for r in rows:
                    if r.get("question_id") in judged:
                        f.write(json.dumps(judged[r["question_id"]], ensure_ascii=False) + "\n")
                    else:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ensure base file written once
    flush()

    def pending_rows():
        return [r for r in rows if judged.get(r.get("question_id"), {}).get("judge_correct") is None]

    round_no = 0
    while True:
        pend = pending_rows()
        if not pend:
            print(f"[{time.strftime('%H:%M:%S')}] ALL DONE: {len(judged)} judged", flush=True)
            break
        round_no += 1
        if not probe(client, args.model, pend):
            print(f"[{time.strftime('%H:%M:%S')}] round {round_no}: gateway DOWN, pending={len(pend)}, wait 60s", flush=True)
            time.sleep(60)
            continue
        print(f"[{time.strftime('%H:%M:%S')}] round {round_no}: gateway OK, judging {len(pend)} rows (workers={args.workers})", flush=True)

        def work(r):
            q = r.get("question", "")
            g = r.get("reference_answer", "")
            p = r.get("hypothesis", "")
            prompt = JUDGE_PROMPT.format(question=q, gold=g, pred=p)
            corr, err = judge_once(client, args.model, prompt)
            rec = dict(r)
            rec["judge_correct"] = corr
            rec["judge_error"] = err
            rec["judge_model"] = args.model
            return r.get("question_id"), rec

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(work, r) for r in pend]
            done = 0
            for fut in as_completed(futs):
                qid, rec = fut.result()
                with lock:
                    judged[qid] = rec
                done += 1
                if done % 25 == 0:
                    print(f"[{time.strftime('%H:%M:%S')}]   {done}/{len(pend)} done", flush=True)
        flush()
        still = len(pending_rows())
        print(f"[{time.strftime('%H:%M:%S')}] round {round_no} finished: now pending={still}", flush=True)
        if still == 0:
            break
        # some failed (gateway dropped mid-batch) -> loop back to probe


if __name__ == "__main__":
    main()
