from __future__ import annotations

import argparse
import json
import os
import random
import sqlite3
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List


TASKS = [
    {
        "id": "T1",
        "query": "project test command",
        "prompt": "Before answering, call memory_search with query `project test command`. Reply with exactly the command from the retrieved memory and nothing else.",
        "expected": "node --test tests",
        "auto_prompt": "What test command should I use? Reply with exactly the command from project memory and nothing else. If there is no relevant project memory, reply UNKNOWN.",
    },
    {
        "id": "T2",
        "query": "test-first preference",
        "prompt": "Before answering, call memory_search with query `test-first preference`. Reply with exactly `TEST_FIRST` if the retrieved memory says to verify tests before alternatives; otherwise reply `UNKNOWN`.",
        "expected": "TEST_FIRST",
        "auto_prompt": "Does the project memory say to verify tests before suggesting alternatives? Reply with exactly TEST_FIRST or UNKNOWN.",
    },
    {
        "id": "T3",
        "query": "npm test failure",
        "prompt": "Before answering, call memory_search with query `npm test failure`. Reply with exactly `AVOID_NPM_TEST` if the retrieved memory says npm test fails and gives node --test tests as the alternative; otherwise reply `UNKNOWN`.",
        "expected": "AVOID_NPM_TEST",
        "auto_prompt": "What should I avoid based on the project's past command experience? Reply with exactly AVOID_NPM_TEST or UNKNOWN.",
    },
]


def _json_events(stdout: str) -> List[Dict[str, Any]]:
    events = []
    for line in stdout.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            events.append(value)
    return events


def _assistant_text(events: Iterable[Dict[str, Any]]) -> str:
    text = ""
    for event in events:
        if event.get("type") != "message_end":
            continue
        message = event.get("message")
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        text = "".join(part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text")
    return text.strip()


def _usage(events: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    usage = {"input": 0, "output": 0, "total": 0}
    for event in events:
        message = event.get("message")
        if event.get("type") != "message_end" or not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        value = message.get("usage")
        if not isinstance(value, dict):
            continue
        for key in usage:
            source_key = "totalTokens" if key == "total" and "total" not in value else key
            if isinstance(value.get(source_key), (int, float)):
                usage[key] = max(usage[key], int(value[source_key]))
    return usage


def _tool_names(events: Iterable[Dict[str, Any]]) -> List[str]:
    names = []
    for event in events:
        if event.get("type") not in {"tool_execution_start", "tool_call_start"}:
            continue
        name = event.get("toolName") or event.get("tool_name") or event.get("name")
        if isinstance(name, str):
            names.append(name)
    return names


def _memory_tokens(events: Iterable[Dict[str, Any]]) -> int:
    total = 0
    for event in events:
        if event.get("type") != "tool_execution_end" or event.get("toolName") != "memory_search":
            continue
        result = event.get("result")
        details = result.get("details") if isinstance(result, dict) else None
        estimate = details.get("tokenEstimate") if isinstance(details, dict) else None
        if isinstance(estimate, (int, float)):
            total += int(estimate)
    return total


def _injected_tokens(memory_dir: Path) -> int:
    database = memory_dir / "memories.sqlite3"
    if not database.exists():
        return 0
    try:
        with sqlite3.connect(database) as connection:
            rows = connection.execute("SELECT metadata_json FROM operations WHERE type = 'inject'").fetchall()
    except sqlite3.Error:
        return 0
    total = 0
    for (metadata_json,) in rows:
        try:
            metadata = json.loads(metadata_json or "{}")
        except json.JSONDecodeError:
            continue
        value = metadata.get("tokenEstimate") if isinstance(metadata, dict) else None
        if isinstance(value, (int, float)):
            total += int(value)
    return total


def run_once(task: Dict[str, str], condition: str, *, pi_root: Path, provider_extension: Path,
             memory_extension: Path, workdir: Path, memory_dir: Path, agent_dir: Path,
             memory_mode: str) -> Dict[str, Any]:
    environment = os.environ.copy()
    environment.update({
        "PI_CODING_AGENT_DIR": str(agent_dir),
        "PI_SEMANTIC_MEMORY_DIR": str(memory_dir),
        "PI_SEMANTIC_MEMORY_AUTO_CAPTURE": "0",
        "PI_SEMANTIC_MEMORY_AUTO_INJECT": "1" if memory_mode == "auto" and condition == "memory_enabled" else "0",
    })
    command = [
        "node", str(pi_root / "packages/coding-agent/dist/bundle/cli.js"),
        "--provider", "aibh", "--model", "v4 pro",
        "--extension", str(provider_extension), "--extension", str(memory_extension),
        "--print", "--mode", "json", "--thinking", "off",
    ]
    if memory_mode == "auto" or condition == "memory_disabled":
        command.append("--no-tools")
    else:
        command.extend(["--tools", "memory_search"])
    command.extend(["--", task["auto_prompt"] if memory_mode == "auto" else task["prompt"]])
    injected_tokens_before = _injected_tokens(memory_dir)
    started = time.perf_counter()
    try:
        completed = subprocess.run(command, cwd=workdir, env=environment, capture_output=True, text=True,
                                   encoding="utf-8", errors="replace", timeout=90)
    except subprocess.TimeoutExpired as error:
        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        partial = error.stdout if isinstance(error.stdout, str) else ""
        return {
            "suite": "v0.5-pi-ab", "run_id": "run_" + uuid.uuid4().hex[:8], "condition": condition,
            "task_id": task["id"], "query": task["query"], "success": False, "expected": task["expected"],
            "response": "", "tool_calls": [], "memory_search_called": False, "tool_error_count": 0,
            "provider_input_tokens": 0, "provider_output_tokens": 0, "provider_total_tokens": 0,
            "memory_tokens": max(0, _injected_tokens(memory_dir) - injected_tokens_before), "latency_ms": elapsed_ms,
            "memory_mode": memory_mode if condition == "memory_enabled" else "disabled",
            "pollution": False, "returncode": None, "stderr": f"timeout after 90s; partial={partial[-1000:]}",
        }
    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    events = _json_events(completed.stdout)
    text = _assistant_text(events)
    usage = _usage(events)
    tool_names = _tool_names(events)
    return {
        "suite": "v0.5-pi-ab",
        "run_id": "run_" + uuid.uuid4().hex[:8],
        "condition": condition,
        "task_id": task["id"],
        "query": task["query"],
        "success": completed.returncode == 0 and task["expected"] in text,
        "expected": task["expected"],
        "response": text,
        "tool_calls": tool_names,
        "memory_search_called": "memory_search" in tool_names,
        "tool_error_count": sum(1 for event in events if event.get("type") == "tool_execution_end" and event.get("isError")),
        "provider_input_tokens": usage["input"],
        "provider_output_tokens": usage["output"],
        "provider_total_tokens": usage["total"],
        "memory_tokens": _memory_tokens(events) if memory_mode == "explicit" else max(0, _injected_tokens(memory_dir) - injected_tokens_before),
        "latency_ms": elapsed_ms,
        "memory_mode": memory_mode if condition == "memory_enabled" else "disabled",
        "pollution": False,
        "returncode": completed.returncode,
        "stderr": completed.stderr[-2000:],
    }


def _summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "runs": len(rows),
        "success_rate": sum(bool(row["success"]) for row in rows) / len(rows) if rows else 0.0,
        "memory_search_rate": sum(bool(row["memory_search_called"]) for row in rows) / len(rows) if rows else 0.0,
        "input_tokens": sum(row["provider_input_tokens"] for row in rows),
        "output_tokens": sum(row["provider_output_tokens"] for row in rows),
        "latency_ms_mean": sum(row["latency_ms"] for row in rows) / len(rows) if rows else 0.0,
        "tool_error_count": sum(row["tool_error_count"] for row in rows),
        "pollution_count": sum(bool(row["pollution"]) for row in rows),
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required and is never written to the report")
    pi_root = Path(args.pi_root).resolve()
    provider_extension = Path(args.provider_extension).resolve()
    memory_extension = Path(args.memory_extension).resolve()
    root = Path(args.output).resolve() if args.output else Path.cwd() / "eval" / "runs" / ("v05-pi-ab-" + uuid.uuid4().hex[:12])
    root.mkdir(parents=True, exist_ok=True)
    workdir = Path(args.workdir).resolve() if args.workdir else root / "repository"
    workdir.mkdir(parents=True, exist_ok=True)
    memory_dir = Path(args.memory_dir).resolve() if args.memory_dir else root / "semantic-memory"
    agent_dir = root / "pi-agent"
    seed_environment = os.environ.copy()
    seed_environment.update({"PI_SEMANTIC_MEMORY_DIR": str(memory_dir), "PI_AB_WORKDIR": str(workdir)})
    seed = subprocess.run(["node", "--experimental-strip-types", str(Path(args.seed_script).resolve())],
                          cwd=pi_root, env=seed_environment, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=30)
    if seed.returncode != 0:
        raise RuntimeError(f"memory seed failed: {seed.stderr[-2000:]}")
    order = [(task, condition) for task in TASKS for condition in ("memory_disabled", "memory_enabled")]
    random.Random(args.seed).shuffle(order)
    rows = [run_once(task, condition, pi_root=pi_root, provider_extension=provider_extension,
                     memory_extension=memory_extension, workdir=workdir, memory_dir=memory_dir, agent_dir=agent_dir,
                     memory_mode=args.memory_mode)
            for task, condition in order]
    by_condition = {condition: _summary([row for row in rows if row["condition"] == condition])
                    for condition in ("memory_disabled", "memory_enabled")}
    report = {
        "suite": "v0.5-pi-ab",
        "status": "pass" if all(row["success"] for row in rows if row["condition"] == "memory_enabled") else "review",
        "seed": args.seed,
        "model": {"provider": "aibh", "id": "v4 pro"},
        "memory_mode": "explicit_tool_only" if args.memory_mode == "explicit" else "auto_inject",
        "auto_inject": args.memory_mode == "auto",
        "task_count": len(TASKS),
        "order": [{"task_id": task["id"], "condition": condition} for task, condition in order],
        "rows": rows,
        "summary": by_condition,
        "success_rate_delta": by_condition["memory_enabled"]["success_rate"] - by_condition["memory_disabled"]["success_rate"],
        "artifact_dir": str(root),
        "report_path": str(root / "report.json"),
    }
    (root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run fixed Pi V0.5 explicit-memory A/B tasks")
    parser.add_argument("--pi-root", required=True)
    parser.add_argument("--provider-extension", required=True)
    parser.add_argument("--memory-extension", required=True)
    parser.add_argument("--seed-script", required=True)
    parser.add_argument("--workdir")
    parser.add_argument("--memory-dir")
    parser.add_argument("--output")
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--memory-mode", choices=["explicit", "auto"], default="explicit")
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
