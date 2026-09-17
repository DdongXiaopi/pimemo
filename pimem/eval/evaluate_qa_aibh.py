from __future__ import annotations

import argparse
import json
import os
import runpy
from pathlib import Path
from typing import Any

from openai import OpenAI
import httpx


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            name, value = line.split("=", 1)
            os.environ.setdefault(name.strip(), value.strip().strip('"').strip("'"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def run(args: argparse.Namespace) -> int:
    root = Path(__file__).resolve().parents[2]
    load_env(root / ".env")
    load_env(Path(args.env_file).resolve() if args.env_file else root / ".env")
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://aibh.cc/v1")
    model = args.model or os.environ.get("PIMEM_MODEL", "v4 pro")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")

    official_path = root / "LongMemEval-main" / "LongMemEval-main" / "src" / "evaluation" / "evaluate_qa.py"
    get_anscheck_prompt = runpy.run_path(str(official_path))["get_anscheck_prompt"]
    hypotheses_path = Path(args.hypotheses).resolve()
    references_path = Path(args.references).resolve()
    hypotheses = read_jsonl(hypotheses_path)
    references = json.loads(references_path.read_text(encoding="utf-8-sig"))
    reference_by_id = {entry["question_id"]: entry for entry in references}
    output = Path(args.output).resolve() if args.output else hypotheses_path.with_name(hypotheses_path.name + ".eval-results-aibh-v4-pro.jsonl")
    summary_path = output.with_suffix(output.suffix + ".summary.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=args.timeout,
                    http_client=httpx.Client(timeout=args.timeout, follow_redirects=True))
    results: list[dict[str, Any]] = []
    with output.open("w", encoding="utf-8") as output_file:
        for index, entry in enumerate(hypotheses, start=1):
            question_id = entry["question_id"]
            reference = reference_by_id[question_id]
            prompt = get_anscheck_prompt(reference["question_type"], reference["question"], reference["answer"], entry.get("hypothesis", ""), abstention="_abs" in question_id)
            result = dict(entry)
            try:
                completion = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    n=1,
                    temperature=0,
                    max_tokens=10,
                )
                judge_response = (completion.choices[0].message.content or "").strip()
                result["autoeval_label"] = {"model": model, "label": "yes" in judge_response.lower()}
                result["judge_response"] = judge_response
                result["usage"] = {
                    "prompt_tokens": getattr(completion.usage, "prompt_tokens", None),
                    "completion_tokens": getattr(completion.usage, "completion_tokens", None),
                    "total_tokens": getattr(completion.usage, "total_tokens", None),
                }
                result["error"] = None
            except Exception as exc:
                result["autoeval_label"] = {"model": model, "label": False}
                result["judge_response"] = ""
                result["usage"] = {}
                result["error"] = f"{type(exc).__name__}: {exc}"
            results.append(result)
            output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            output_file.flush()
            print(json.dumps({"completed": index, "total": len(hypotheses), "question_id": question_id, "label": result["autoeval_label"]["label"], "error": result["error"]}, ensure_ascii=False), flush=True)

    successful = [entry for entry in results if not entry["error"]]
    by_type: dict[str, list[int]] = {}
    for entry in successful:
        question_type = reference_by_id[entry["question_id"]]["question_type"]
        by_type.setdefault(question_type, []).append(int(entry["autoeval_label"]["label"]))
    summary = {
        "suite": "longmemeval-qa-aibh-judge",
        "status": "completed" if len(successful) == len(results) else "partial_api_failure",
        "model": model,
        "base_url": base_url,
        "hypotheses": str(hypotheses_path),
        "references": str(references_path),
        "case_count": len(results),
        "api_success_count": len(successful),
        "api_error_count": len(results) - len(successful),
        "correct_count": sum(int(entry["autoeval_label"]["label"]) for entry in successful),
        "accuracy": sum(int(entry["autoeval_label"]["label"]) for entry in successful) / len(successful) if successful else 0.0,
        "accuracy_by_question_type": {key: sum(values) / len(values) for key, values in by_type.items()},
        "result_file": str(output),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


parser = argparse.ArgumentParser()
parser.add_argument("hypotheses")
parser.add_argument("references")
parser.add_argument("--output")
parser.add_argument("--env-file")
parser.add_argument("--model")
parser.add_argument("--timeout", type=float, default=120.0)
raise SystemExit(run(parser.parse_args()))
