from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            name, value = line.split("=", 1)
            os.environ.setdefault(name.strip(), value.strip().strip('"').strip("'"))


def load_generation_runner(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("pimem_real_api_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load generation runner: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def run_generation(stage: str, runner: Any, *, input_path: Path, output_dir: Path,
                   api_key: str, base_url: str, model: str, args: argparse.Namespace) -> Dict[str, Any]:
    print(json.dumps({"stage": stage, "action": "generation_started", "input": str(input_path),
                      "output": str(output_dir)}, ensure_ascii=False), flush=True)
    generation_args = SimpleNamespace(
        input=str(input_path), base_url=base_url, api_key=api_key, model=model,
        limit=args.limit, offset=args.offset, retrieval_limit=args.retrieval_limit,
        context_token_budget=args.context_token_budget, max_tokens=args.max_tokens,
        timeout=args.timeout, output=str(output_dir), no_context_budget=args.no_context_budget,
    )
    report = runner.run(generation_args)
    report_path = output_dir / "report.json"
    if not report_path.exists():
        write_json(report_path, report)
    hypotheses_path = output_dir / "hypotheses.jsonl"
    if not hypotheses_path.exists():
        raise RuntimeError(f"generation did not produce hypotheses: {hypotheses_path}")
    print(json.dumps({"stage": stage, "action": "generation_finished",
                      "case_count": report.get("case_count"),
                      "api_success_count": report.get("api_success_count"),
                      "api_error_count": report.get("api_error_count"),
                      "hypotheses": str(hypotheses_path)}, ensure_ascii=False), flush=True)
    return report


def run_judge(stage: str, *, hypotheses: Path, references: Path, output: Path,
              project_root: Path, env_file: Path, api_key: str, base_url: str,
              model: str, timeout: float) -> Dict[str, Any]:
    print(json.dumps({"stage": stage, "action": "judge_started", "hypotheses": str(hypotheses),
                      "output": str(output)}, ensure_ascii=False), flush=True)
    judge_script = project_root / "eval" / "evaluate_qa_aibh.py"
    child_env = os.environ.copy()
    child_env["OPENAI_API_KEY"] = api_key
    child_env["OPENAI_BASE_URL"] = base_url
    child_env["PIMEM_MODEL"] = model
    command = [sys.executable, str(judge_script), str(hypotheses), str(references),
               "--output", str(output), "--model", model, "--timeout", str(timeout),
               "--env-file", str(env_file)]
    completed = subprocess.run(command, cwd=str(project_root), env=child_env, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"{stage} judge failed with exit code {completed.returncode}")
    summary_path = Path(str(output) + ".summary.json")
    if not summary_path.exists():
        raise RuntimeError(f"judge did not produce summary: {summary_path}")
    summary = read_json(summary_path)
    print(json.dumps({"stage": stage, "action": "judge_finished",
                      "case_count": summary.get("case_count"),
                      "api_success_count": summary.get("api_success_count"),
                      "api_error_count": summary.get("api_error_count"),
                      "accuracy": summary.get("accuracy"),
                      "summary": str(summary_path)}, ensure_ascii=False), flush=True)
    return summary


def compare_summary(current: Dict[str, Any], baseline: Optional[Dict[str, Any]], baseline_path: Path) -> Dict[str, Any]:
    if baseline is None:
        return {"available": False, "path": str(baseline_path), "reason": "baseline_summary_not_found"}
    current_types = current.get("accuracy_by_question_type", {})
    baseline_types = baseline.get("accuracy_by_question_type", {})
    type_delta = {
        key: {"current": current_types.get(key), "baseline": baseline_types.get(key),
              "delta": current_types.get(key) - baseline_types.get(key)
              if key in current_types and key in baseline_types else None}
        for key in sorted(set(current_types) | set(baseline_types))
    }
    current_accuracy = current.get("accuracy")
    baseline_accuracy = baseline.get("accuracy")
    return {
        "available": True,
        "path": str(baseline_path),
        "baseline_suite": baseline.get("suite"),
        "baseline_model": baseline.get("model"),
        "baseline_base_url": baseline.get("base_url"),
        "current_accuracy": current_accuracy,
        "baseline_accuracy": baseline_accuracy,
        "accuracy_delta": current_accuracy - baseline_accuracy
        if current_accuracy is not None and baseline_accuracy is not None else None,
        "current_case_count": current.get("case_count"),
        "baseline_case_count": baseline.get("case_count"),
        "current_api_error_count": current.get("api_error_count"),
        "baseline_api_error_count": baseline.get("api_error_count"),
        "accuracy_by_question_type": type_delta,
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    workspace_root = project_root.parent
    source_root = project_root / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    env_file = Path(args.env_file).resolve() if args.env_file else project_root / ".env"
    load_env(env_file)
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured; use --api-key or --env-file")
    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL", "https://aibh.cc/v1")
    model = args.model or os.environ.get("PIMEM_MODEL", "v4 pro")
    output_root = Path(args.output_root).resolve() if args.output_root else project_root / "official_qa" / (
        "serial-comparison-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    output_root.mkdir(parents=True, exist_ok=True)
    s_output = output_root / "s-generation"
    m_output = output_root / "m-generation"
    s_judge_output = output_root / "s-judge.jsonl"
    m_judge_output = output_root / "m-judge.jsonl"
    s_input = Path(args.s_input).resolve() if args.s_input else workspace_root / "longmemeval_s_cleaned.json"
    m_input = Path(args.m_input).resolve() if args.m_input else workspace_root / "longmemeval_m_cleaned.json"
    s_references = Path(args.s_references).resolve() if args.s_references else s_input
    m_references = Path(args.m_references).resolve() if args.m_references else m_input
    generation_runner = load_generation_runner(project_root / "eval" / "run_longmemeval_real_api.py")

    s_report = run_generation("S", generation_runner, input_path=s_input, output_dir=s_output,
                              api_key=api_key, base_url=base_url, model=model, args=args)
    if s_report.get("api_error_count", 0) and not args.allow_api_errors:
        raise RuntimeError("S generation has API errors; M generation and both judges were not started. "
                           "Use --allow-api-errors only if blank/error answers should be judged.")
    m_report = run_generation("M", generation_runner, input_path=m_input, output_dir=m_output,
                              api_key=api_key, base_url=base_url, model=model, args=args)
    if m_report.get("api_error_count", 0) and not args.allow_api_errors:
        raise RuntimeError("M generation has API errors; judges were not started. "
                           "Use --allow-api-errors only if blank/error answers should be judged.")

    s_summary = run_judge("S", hypotheses=s_output / "hypotheses.jsonl", references=s_references,
                          output=s_judge_output, project_root=project_root, env_file=env_file,
                          api_key=api_key, base_url=base_url, model=model, timeout=args.timeout)
    m_summary = run_judge("M", hypotheses=m_output / "hypotheses.jsonl", references=m_references,
                          output=m_judge_output, project_root=project_root, env_file=env_file,
                          api_key=api_key, base_url=base_url, model=model, timeout=args.timeout)

    default_s_baseline = project_root / "official_qa" / "s-500-all.eval-results-aibh-v4-pro.jsonl.summary.json"
    default_m_baseline = project_root / "official_qa" / "m-500-all.eval-results-aibh-v4-pro-20260831.jsonl.summary.json"
    s_baseline_path = Path(args.s_baseline).resolve() if args.s_baseline else default_s_baseline
    m_baseline_path = Path(args.m_baseline).resolve() if args.m_baseline else default_m_baseline
    comparison = {
        "suite": "longmemeval-sm-serial-comparison", "status": "completed",
        "generated_at": datetime.now(timezone.utc).isoformat(), "model": model, "base_url": base_url,
        "execution_order": ["S generation", "M generation", "S judge", "M judge"], "serial": True,
        "generation": {"S": s_report, "M": m_report}, "judge": {"S": s_summary, "M": m_summary},
        "comparison": {
            "S": compare_summary(s_summary, read_json(s_baseline_path) if s_baseline_path.exists() else None, s_baseline_path),
            "M": compare_summary(m_summary, read_json(m_baseline_path) if m_baseline_path.exists() else None, m_baseline_path),
        },
        "artifacts": {"output_root": str(output_root), "s_generation": str(s_output),
                      "m_generation": str(m_output), "s_judge": str(s_judge_output), "m_judge": str(m_judge_output)},
    }
    comparison_path = output_root / "comparison.json"
    write_json(comparison_path, comparison)
    print(json.dumps({"status": "completed", "comparison": str(comparison_path),
                      "s_accuracy": s_summary.get("accuracy"), "m_accuracy": m_summary.get("accuracy")},
                     ensure_ascii=False, indent=2), flush=True)
    return comparison


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Serially run LongMemEval S/M generation, then S/M Judge, and compare baselines"
    )
    parser.add_argument("--s-input")
    parser.add_argument("--m-input")
    parser.add_argument("--s-references")
    parser.add_argument("--m-references")
    parser.add_argument("--s-baseline")
    parser.add_argument("--m-baseline")
    parser.add_argument("--output-root")
    parser.add_argument("--env-file")
    parser.add_argument("--api-key")
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--retrieval-limit", type=int, default=20)
    parser.add_argument("--context-token-budget", type=int, default=4096)
    parser.add_argument("--no-context-budget", action="store_true",
                        help="diagnostic mode: retain all retrieved evidence without applying the pack budget")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--allow-api-errors", action="store_true")
    try:
        run(parser.parse_args())
        return 0
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
