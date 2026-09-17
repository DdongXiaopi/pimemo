from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict

from pimem.core.models import Scope
from pimem.core.procedures import (
    experience_from_dict,
    experience_to_dict,
    procedure_from_dict,
    procedure_to_dict,
    propose_candidate_procedure,
    replay_candidate,
    shadow_candidate,
)
from pimem.eval.longmemeval import run_longmemeval
from pimem.eval.runner import run_v0
from pimem.eval.replay_state import replay
from pimem.runtime import MemoryRuntime
from pimem.adapters.generic import GenericMemoryAdapter


def _runtime(args: argparse.Namespace) -> MemoryRuntime:
    return MemoryRuntime(args.db or os.environ.get("PIMEM_DB", str(Path.cwd() / ".pimem" / "memory.sqlite3")))


def _scope(value: str, runtime: MemoryRuntime, branch: str | None = None) -> Scope:
    return Scope(repository=runtime.store.resolve_repository(value), branch=branch)


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def _json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pimem", description="Pi-first local memory runtime V0")
    parser.add_argument("--db", help="SQLite path")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init")
    init.add_argument("--repository", required=True)
    ingest = sub.add_parser("ingest")
    ingest.add_argument("--event", required=True, help="JSON object or path to JSON file")
    propose = sub.add_parser("propose")
    propose.add_argument("--observation", required=True)
    commit = sub.add_parser("commit")
    commit.add_argument("--candidate", required=True)
    commit.add_argument("--truth-status", default="supported", choices=["candidate", "supported", "disputed"])
    commit.add_argument("--use-policy", default="context_allowed", choices=["reference_only", "context_allowed", "verification_required", "blocked"])
    patch = sub.add_parser("patch")
    patch.add_argument("--claim", required=True)
    patch.add_argument("--base-revision", required=True, type=int)
    patch.add_argument("--covered-scope", required=True, help="JSON object")
    patch.add_argument("--changes", required=True, help="JSON object of explicit field paths")
    patch.add_argument("--expected-old", required=True, help="JSON object matching changes")
    patch.add_argument("--idempotency-key")
    recall = sub.add_parser("recall")
    recall.add_argument("--query", required=True)
    recall.add_argument("--repository", required=True)
    recall.add_argument("--branch")
    recall.add_argument("--limit", type=int, default=10)
    shadow = sub.add_parser("shadow")
    shadow.add_argument("--query", required=True)
    shadow.add_argument("--repository", required=True)
    shadow.add_argument("--branch")
    shadow.add_argument("--limit", type=int, default=50)
    shadow.add_argument("--token-budget", type=int, default=256)
    shadow.add_argument("--level", choices=["L0", "L1"], default="L0")
    context = sub.add_parser("context", help="Prepare a model-neutral Context Pack v0.6")
    context.add_argument("--query", required=True)
    context.add_argument("--repository", required=True)
    context.add_argument("--branch")
    context.add_argument("--as-of")
    context.add_argument("--limit", type=int, default=50)
    context.add_argument("--token-budget", type=int, default=256)
    generic_ingest = sub.add_parser("generic-ingest", help="Ingest a model-neutral observation JSON object")
    generic_ingest.add_argument("--input", required=True, help="JSON object or path to JSON file")
    generic_query = sub.add_parser("generic-query", help="Prepare Context Pack from a model-neutral query JSON object")
    generic_query.add_argument("--input", required=True, help="JSON object or path to JSON file")
    explain = sub.add_parser("explain")
    explain.add_argument("--claim", required=True)
    sub.add_parser("rebuild")
    export = sub.add_parser("export")
    export.add_argument("--out", required=True)
    forget = sub.add_parser("forget")
    forget.add_argument("--target", required=True)
    forget.add_argument("--policy", default="local_runtime")
    replay = sub.add_parser("replay-state")
    replay.add_argument("--case", required=True)
    probe = sub.add_parser("probe")
    probe.add_argument("--pi-command", default="pi")
    evaluation = sub.add_parser("eval")
    evaluation.add_argument("--suite", default="v0", choices=["v0", "longmemeval"])
    evaluation.add_argument("--input", help="LongMemEval JSON input")
    evaluation.add_argument("--limit", type=int)
    evaluation.add_argument("--offset", type=int, default=0)
    evaluation.add_argument("--output-dir")
    procedure_propose = sub.add_parser("procedure-propose")
    procedure_propose.add_argument("--input", required=True, help="JSON file containing experiences and procedure fields")
    procedure_replay = sub.add_parser("procedure-replay")
    procedure_replay.add_argument("--procedure", required=True, help="Candidate procedure JSON file")
    procedure_replay.add_argument("--experience", required=True, help="Experience JSON file")
    procedure_shadow = sub.add_parser("procedure-shadow")
    procedure_shadow.add_argument("--procedure", required=True, help="Candidate procedure JSON file")
    procedure_shadow.add_argument("--experiences", required=True, help="JSON file containing an experiences array")

    args = parser.parse_args(argv)
    try:
        if args.command == "eval":
            if args.suite == "longmemeval":
                if not args.input:
                    raise ValueError("--input is required for the longmemeval suite")
                _json(run_longmemeval(args.input, artifact_root=args.output_dir, limit=args.limit, offset=args.offset))
            else:
                _json(run_v0())
            return 0
        if args.command == "procedure-propose":
            payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
            experiences = [experience_from_dict(value) for value in payload["experiences"]]
            procedure = propose_candidate_procedure(
                experiences,
                procedure_id=payload["procedure_id"],
                trigger=payload["trigger"],
                preconditions=payload["preconditions"],
                actions=payload["actions"],
                success_criteria=payload["success_criteria"],
                failure_modes=payload.get("failure_modes", []),
                stop_conditions=payload["stop_conditions"],
                rollback=payload["rollback"],
                scope=Scope(**payload["scope"]),
                risk_level=payload.get("risk_level", "S1"),
            )
            _json(procedure_to_dict(procedure))
            return 0
        if args.command == "procedure-replay":
            procedure = procedure_from_dict(json.loads(Path(args.procedure).read_text(encoding="utf-8")))
            experience = experience_from_dict(json.loads(Path(args.experience).read_text(encoding="utf-8")))
            _json(replay_candidate(procedure, experience))
            return 0
        if args.command == "procedure-shadow":
            procedure = procedure_from_dict(json.loads(Path(args.procedure).read_text(encoding="utf-8")))
            payload = json.loads(Path(args.experiences).read_text(encoding="utf-8"))
            experiences = [experience_from_dict(value) for value in payload["experiences"]]
            _json(shadow_candidate(procedure, experiences))
            return 0
        runtime = _runtime(args)
        if args.command == "init":
            _json({"repository_id": runtime.init_repository(args.repository), "db": runtime.store.path})
        elif args.command == "ingest":
            raw = Path(args.event).read_text(encoding="utf-8") if Path(args.event).exists() else args.event
            _json(runtime.ingest_event(json.loads(raw)))
        elif args.command == "propose":
            _json(runtime.propose_claims(args.observation))
        elif args.command == "commit":
            _json(runtime.store.claim_to_dict(runtime.commit_claim(args.candidate, truth_status=args.truth_status, use_policy=args.use_policy)))
        elif args.command == "patch":
            _json(runtime.store.claim_to_dict(runtime.patch_claim(
                args.claim,
                base_revision=args.base_revision,
                covered_scope=json.loads(args.covered_scope),
                changes=json.loads(args.changes),
                expected_old=json.loads(args.expected_old),
                idempotency_key=args.idempotency_key,
            )))
        elif args.command == "recall":
            _json(runtime.recall(args.query, _scope(args.repository, runtime, args.branch), args.limit))
        elif args.command == "shadow":
            _json(runtime.shadow(args.query, _scope(args.repository, runtime, args.branch),
                                 token_budget=args.token_budget, level=args.level, limit=args.limit))
        elif args.command == "context":
            _json(runtime.prepare_context(args.query, _scope(args.repository, runtime, args.branch),
                                          as_of=args.as_of, token_budget=args.token_budget, limit=args.limit))
        elif args.command in {"generic-ingest", "generic-query"}:
            raw = Path(args.input).read_text(encoding="utf-8") if Path(args.input).exists() else args.input
            adapter = GenericMemoryAdapter(runtime)
            _json(adapter.ingest(json.loads(raw)) if args.command == "generic-ingest" else adapter.query(json.loads(raw)))
        elif args.command == "explain":
            _json(runtime.explain(args.claim))
        elif args.command == "rebuild":
            _json({"reindexed_claims": runtime.rebuild_indexes()})
        elif args.command == "export":
            _json(runtime.store.export(args.out))
        elif args.command == "forget":
            runtime.forget(args.target, args.policy)
            _json({"forgotten": args.target, "policy": args.policy})
        elif args.command == "replay-state":
            _json(replay(args.case) if Path(args.case).exists() else {"case": args.case, "level": "T0", "status": "not_recorded", "message": "V0 only replays recorded state cases."})
        elif args.command == "probe":
            runtime.pi.pi_command = args.pi_command
            _json(runtime.pi.capabilities())
        return 0
    except (KeyError, ValueError, OSError, json.JSONDecodeError) as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
