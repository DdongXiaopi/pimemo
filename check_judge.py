import json
import sys
from pathlib import Path

# 用法: python check_judge.py [judged_jsonl_path]
# 默认统计 _online_synth_full500/rows_judged.jsonl

def main():
    p = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "_online_synth_full500/rows_judged.jsonl"
    )
    if not p.is_file():
        print(f"[!] judged file not found: {p}")
        print("    judge has not produced output yet (gateway down / still waiting).")
        return
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    n = len(rows)
    judged = [r for r in rows if r.get("judge_correct") is not None]
    pending = n - len(judged)
    correct = sum(1 for r in judged if r["judge_correct"])
    wrong = len(judged) - correct
    print(f"judged file : {p}")
    print(f"total rows : {n}")
    print(f"judged     : {len(judged)}  (correct={correct}, wrong={wrong})")
    print(f"pending    : {pending}")
    if judged:
        print(f"accuracy   : {correct/len(judged):.4f}")
    else:
        print("accuracy   : n/a (no rows judged yet)")


if __name__ == "__main__":
    main()
