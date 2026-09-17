from __future__ import annotations

import copy
import json
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from pimem.core.models import Claim, ClaimRelation, Evidence, Observation, Scope, stable_hash, utc_now
from pimem.core.policy import validate_claim_policy
from .migrations import SCHEMA


class SQLiteStore:
    def __init__(self, path: str) -> None:
        self.path = str(Path(path).expanduser())
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=2.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=2000")
        return connection

    @contextmanager
    def _open(self):
        """Transactional connection that is always closed.

        ``with sqlite3.connect(...)`` only commits or rolls back, it never
        closes the handle. Long-lived callers (the CLI, and especially the MCP
        server) therefore accumulated open connections until the garbage
        collector happened to reclaim them, which kept SQLite files locked on
        Windows and made every temp-directory test fail during teardown.
        """
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._open() as connection:
            connection.executescript(SCHEMA)
            columns = {row["name"] for row in connection.execute("PRAGMA table_info(claims)").fetchall()}
            for name in ("valid_from", "valid_to", "time_precision", "object_path", "event_type", "event_time", "assertion_family_key"):
                if name not in columns:
                    connection.execute(f"ALTER TABLE claims ADD COLUMN {name} TEXT")

    @contextmanager
    def transaction(self):
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def register_repository(self, canonical_path: str) -> str:
        canonical = str(Path(canonical_path).expanduser().resolve())
        repository_id = "repo_" + __import__("hashlib").sha256(canonical.encode("utf-8")).hexdigest()[:16]
        with self._open() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO repositories(repository_id, canonical_path, created_at) VALUES (?, ?, ?)",
                (repository_id, canonical, utc_now()),
            )
        return repository_id

    def resolve_repository(self, value: str) -> str:
        with self._open() as connection:
            row = connection.execute(
                "SELECT repository_id FROM repositories WHERE repository_id = ? OR canonical_path = ?",
                (value, str(Path(value).expanduser().resolve())),
            ).fetchone()
        if not row:
            raise KeyError(f"unknown repository: {value}")
        return str(row["repository_id"])

    def insert_observation(self, observation: Observation) -> Tuple[Observation, bool]:
        with self.transaction() as connection:
            existing = connection.execute(
                "SELECT * FROM observations WHERE idempotency_key = ?", (observation.idempotency_key,)
            ).fetchone()
            if existing:
                return self._observation_from_row(existing), False
            connection.execute(
                """INSERT INTO observations
                (observation_id, content, source_type, source_ref, scope_json, observed_at,
                 recorded_at, content_hash, sensitivity, idempotency_key, trace_id, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (observation.id, observation.content, observation.source_type, observation.source_ref,
                 json.dumps(observation.scope.as_dict(), ensure_ascii=False), observation.observed_at,
                 observation.recorded_at, observation.content_hash, observation.sensitivity,
                 observation.idempotency_key, observation.trace_id,
                 json.dumps(observation.metadata, ensure_ascii=False, sort_keys=True)),
            )
            self._operation(connection, "capture", observation.id, observation.idempotency_key,
                            observation.trace_id, {"source_type": observation.source_type})
            connection.execute(
                "INSERT INTO observations_fts(observation_id, content, source_type, scope_repository) VALUES (?, ?, ?, ?)",
                (observation.id, observation.content, observation.source_type, observation.scope.repository),
            )
        return observation, True

    def bulk_insert_observations_claims(self, observations: Iterable[Observation], claims: Iterable[Claim]) -> Tuple[int, int]:
        observation_list = list(observations)
        claim_list = list(claims)
        for claim in claim_list:
            validate_claim_policy(claim.predicate, claim.truth_status, claim.use_policy)
            self._validate_time(claim.valid_from, claim.valid_to, claim.time_precision)
        inserted_observations = 0
        inserted_claims = 0
        with self.transaction() as connection:
            observation_ids: Dict[str, str] = {}
            for observation in observation_list:
                existing = connection.execute(
                    "SELECT observation_id FROM observations WHERE idempotency_key = ?",
                    (observation.idempotency_key,),
                ).fetchone()
                if existing:
                    observation_ids[observation.idempotency_key] = str(existing["observation_id"])
                    continue
                connection.execute(
                    """INSERT INTO observations
                    (observation_id, content, source_type, source_ref, scope_json, observed_at,
                     recorded_at, content_hash, sensitivity, idempotency_key, trace_id, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (observation.id, observation.content, observation.source_type, observation.source_ref,
                     json.dumps(observation.scope.as_dict(), ensure_ascii=False), observation.observed_at,
                     observation.recorded_at, observation.content_hash, observation.sensitivity,
                     observation.idempotency_key, observation.trace_id,
                     json.dumps(observation.metadata, ensure_ascii=False, sort_keys=True)),
                )
                self._operation(connection, "capture", observation.id, observation.idempotency_key,
                                observation.trace_id, {"source_type": observation.source_type})
                connection.execute(
                    "INSERT INTO observations_fts(observation_id, content, source_type, scope_repository) VALUES (?, ?, ?, ?)",
                    (observation.id, observation.content, observation.source_type, observation.scope.repository),
                )
                observation_ids[observation.idempotency_key] = observation.id
                inserted_observations += 1
            for claim in claim_list:
                existing = connection.execute(
                    "SELECT claim_id FROM claims WHERE idempotency_key = ?", (claim.idempotency_key,)
                ).fetchone()
                if existing:
                    continue
                source_ids = [observation_ids.get(claim_source, claim_source) for claim_source in claim.source_observation_ids]
                for observation_id in source_ids:
                    if not connection.execute(
                        "SELECT 1 FROM observations WHERE observation_id = ?", (observation_id,)
                    ).fetchone():
                        raise KeyError(f"unknown source observation: {observation_id}")
                connection.execute(
                    """INSERT INTO claims
                    (claim_id, subject, predicate, object_json, scope_json, valid_from, valid_to, time_precision,
                     truth_status, lifecycle_status, use_policy, utility_status, revision, created_at, updated_at, idempotency_key,
                     object_path, event_type, event_time, assertion_family_key)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (claim.claim_id, claim.subject, claim.predicate, json.dumps(claim.object, ensure_ascii=False),
                     json.dumps(claim.scope.as_dict(), ensure_ascii=False), claim.valid_from, claim.valid_to,
                     claim.time_precision, claim.truth_status, claim.lifecycle_status, claim.use_policy,
                     claim.utility_status, claim.revision, claim.created_at, claim.updated_at, claim.idempotency_key, claim.object_path, claim.event_type,
                      claim.event_time, claim.assertion_family_key),
                )
                for index, observation_id in enumerate(source_ids):
                    evidence = Evidence(f"ev_{claim.claim_id}_{index}", claim.claim_id, observation_id)
                    connection.execute(
                        "INSERT INTO evidence_links(evidence_id, claim_id, observation_id, relation, created_at) VALUES (?, ?, ?, ?, ?)",
                        (evidence.id, evidence.claim_id, evidence.observation_id, evidence.relation, evidence.created_at),
                    )
                if claim.lifecycle_status != "deleted" and claim.use_policy != "blocked":
                    connection.execute(
                        "INSERT INTO claims_fts(claim_id, statement, predicate, scope_repository) VALUES (?, ?, ?, ?)",
                        (claim.claim_id, claim.statement(), claim.predicate, claim.scope.repository),
                    )
                self._operation(connection, "claim_commit", claim.claim_id, f"claim:{claim.idempotency_key}", None,
                                {"predicate": claim.predicate, "revision": claim.revision})
                inserted_claims += 1
        return inserted_observations, inserted_claims

    def get_observation(self, observation_id: str) -> Observation:
        with self._open() as connection:
            row = connection.execute("SELECT * FROM observations WHERE observation_id = ?", (observation_id,)).fetchone()
        if not row:
            raise KeyError(f"unknown observation: {observation_id}")
        return self._observation_from_row(row)

    def list_observations(self, scope: Scope, *, session_id: Optional[str] = None,
                          limit: int = 200) -> List[Observation]:
        scope_values = (scope.repository, scope.branch, scope.service, scope.module, scope.file, scope.session, scope.task)
        session_clause = ""
        params: List[Any] = list(scope_values)
        if session_id is not None:
            session_clause = " AND json_extract(o.metadata_json, '$.session_id') = ?"
            params.append(session_id)
        params.append(max(limit, 1))
        scope_sql = """json_extract(o.scope_json, '$.repository') = ?
            AND (json_extract(o.scope_json, '$.branch') IS NULL OR json_extract(o.scope_json, '$.branch') = ?)
            AND (json_extract(o.scope_json, '$.service') IS NULL OR json_extract(o.scope_json, '$.service') = ?)
            AND (json_extract(o.scope_json, '$.module') IS NULL OR json_extract(o.scope_json, '$.module') = ?)
            AND (json_extract(o.scope_json, '$.file') IS NULL OR json_extract(o.scope_json, '$.file') = ?)
            AND (json_extract(o.scope_json, '$.session') IS NULL OR json_extract(o.scope_json, '$.session') = ?)
            AND (json_extract(o.scope_json, '$.task') IS NULL OR json_extract(o.scope_json, '$.task') = ?)"""
        with self._open() as connection:
            rows = connection.execute(
                f"SELECT o.* FROM observations o WHERE {scope_sql}{session_clause} "
                "ORDER BY CAST(json_extract(o.metadata_json, '$.turn_index') AS INTEGER), o.observed_at, o.observation_id LIMIT ?",
                params,
            ).fetchall()
        return [self._observation_from_row(row) for row in rows]

    def save_candidate(self, candidate_id: str, claim: Claim, observation_id: str) -> None:
        with self._open() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO claim_candidates(candidate_id, claim_json, observation_id, created_at) VALUES (?, ?, ?, ?)",
                (candidate_id, json.dumps(self.claim_to_dict(claim), ensure_ascii=False, sort_keys=True), observation_id, utc_now()),
            )

    def get_candidate(self, candidate_id: str) -> Claim:
        with self._open() as connection:
            row = connection.execute("SELECT claim_json FROM claim_candidates WHERE candidate_id = ?", (candidate_id,)).fetchone()
        if not row:
            raise KeyError(f"unknown candidate: {candidate_id}")
        return self.claim_from_dict(json.loads(row["claim_json"]))

    def commit_claim(self, claim: Claim) -> Tuple[Claim, bool]:
        self._validate_time(claim.valid_from, claim.valid_to, claim.time_precision)
        with self.transaction() as connection:
            existing = connection.execute(
                "SELECT * FROM claims WHERE idempotency_key = ?", (claim.idempotency_key,)
            ).fetchone()
            if existing:
                existing_claim = self._claim_from_row(existing)
                evidence = connection.execute(
                    "SELECT observation_id FROM evidence_links WHERE claim_id = ? ORDER BY created_at",
                    (existing_claim.claim_id,),
                ).fetchall()
                existing_claim.source_observation_ids = [item["observation_id"] for item in evidence]
                return existing_claim, False
            for observation_id in claim.source_observation_ids:
                if not connection.execute("SELECT 1 FROM observations WHERE observation_id = ?", (observation_id,)).fetchone():
                    raise KeyError(f"unknown source observation: {observation_id}")
            connection.execute(
                """INSERT INTO claims
                (claim_id, subject, predicate, object_json, scope_json, valid_from, valid_to, time_precision,
                 truth_status, lifecycle_status, use_policy, utility_status, revision, created_at, updated_at, idempotency_key,
                object_path, event_type, event_time, assertion_family_key)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (claim.claim_id, claim.subject, claim.predicate, json.dumps(claim.object, ensure_ascii=False),
                 json.dumps(claim.scope.as_dict(), ensure_ascii=False), claim.valid_from, claim.valid_to, claim.time_precision,
                 claim.truth_status,
                 claim.lifecycle_status, claim.use_policy, claim.utility_status, claim.revision,
                 claim.created_at, claim.updated_at, claim.idempotency_key, claim.object_path, claim.event_type,
                      claim.event_time, claim.assertion_family_key),
            )
            for index, observation_id in enumerate(claim.source_observation_ids):
                evidence = Evidence(f"ev_{claim.claim_id}_{index}", claim.claim_id, observation_id)
                connection.execute(
                    "INSERT INTO evidence_links(evidence_id, claim_id, observation_id, relation, created_at) VALUES (?, ?, ?, ?, ?)",
                    (evidence.id, evidence.claim_id, evidence.observation_id, evidence.relation, evidence.created_at),
                )
            self._operation(connection, "claim_commit", claim.claim_id, f"claim:{claim.idempotency_key}", None,
                            {"predicate": claim.predicate, "revision": claim.revision})
        try:
            self.index_claim(claim)
        except sqlite3.Error:
            self.record_operation("index_pending", claim.claim_id, f"index:{claim.claim_id}", None, {})
        return claim, True

    def get_claim(self, claim_id: str) -> Claim:
        with self._open() as connection:
            row = connection.execute("SELECT * FROM claims WHERE claim_id = ?", (claim_id,)).fetchone()
            evidence = connection.execute(
                "SELECT observation_id FROM evidence_links WHERE claim_id = ? ORDER BY created_at", (claim_id,)
            ).fetchall()
        if not row:
            raise KeyError(f"unknown claim: {claim_id}")
        claim = self._claim_from_row(row)
        claim.source_observation_ids = [item["observation_id"] for item in evidence]
        return claim

    def patch_claim(self, claim_id: str, *, base_revision: int, covered_scope: Dict[str, Any],
                    changes: Dict[str, Any], expected_old: Dict[str, Any],
                    idempotency_key: Optional[str] = None) -> Claim:
        claim = self.get_claim(claim_id)
        operation_key = idempotency_key or (
            f"patch:{claim_id}:{base_revision}:{stable_hash({'changes': changes, 'expected_old': expected_old})}"
        )
        with self._open() as connection:
            if connection.execute("SELECT 1 FROM memory_operations WHERE idempotency_key = ?", (operation_key,)).fetchone():
                return claim
        if claim.revision != base_revision:
            raise ValueError(f"base_revision_mismatch: expected {claim.revision}, got {base_revision}")
        if not covered_scope or "repository" not in covered_scope:
            raise ValueError("covered_scope must include the claim repository")
        if covered_scope["repository"] != claim.scope.repository:
            raise ValueError("covered_scope_mismatch: repository")
        for key, value in covered_scope.items():
            if key not in {"repository", "branch", "service", "module", "file"}:
                raise ValueError(f"unsupported covered_scope field: {key}")
            if value is not None and value != getattr(claim.scope, key):
                raise ValueError(f"covered_scope_mismatch: {key}")
        if not changes:
            raise ValueError("changes cannot be empty")
        if set(changes) != set(expected_old):
            raise ValueError("expected_old must cover exactly the changed field paths")

        updated = copy.deepcopy(claim)
        for path, value in changes.items():
            current = self._claim_field(claim, path)
            if current != expected_old[path]:
                raise ValueError(f"expected_old_mismatch: {path}")
            self._set_claim_field(updated, path, value)
        updated.revision = claim.revision + 1
        updated.updated_at = utc_now()
        updated.idempotency_key = operation_key
        validate_claim_policy(updated.predicate, updated.truth_status, updated.use_policy)
        self._validate_time(updated.valid_from, updated.valid_to, updated.time_precision)
        if updated.lifecycle_status not in {"active", "deleted", "superseded", "expired", "archived"}:
            raise ValueError(f"invalid lifecycle_status: {updated.lifecycle_status}")

        with self.transaction() as connection:
            connection.execute(
                """UPDATE claims SET subject = ?, predicate = ?, object_json = ?, valid_from = ?, valid_to = ?,
                   time_precision = ?, truth_status = ?,
                   lifecycle_status = ?, use_policy = ?, utility_status = ?, revision = ?, updated_at = ?,
                   idempotency_key = ? WHERE claim_id = ? AND revision = ?""",
                (updated.subject, updated.predicate, json.dumps(updated.object, ensure_ascii=False),
                 updated.valid_from, updated.valid_to, updated.time_precision,
                 updated.truth_status, updated.lifecycle_status, updated.use_policy, updated.utility_status,
                 updated.revision, updated.updated_at, updated.idempotency_key, claim_id, base_revision),
            )
            if connection.total_changes != 1:
                raise ValueError("claim_patch_not_applied")
            self._operation(connection, "claim_patch", claim_id, updated.idempotency_key, None, {
                "before_revision": base_revision,
                "after_revision": updated.revision,
                "covered_scope": covered_scope,
                "changes": changes,
                "expected_old": expected_old,
            })
        try:
            self.index_claim(updated)
        except sqlite3.Error:
            self.record_operation("index_pending", claim_id, f"index:{claim_id}:{updated.revision}", None, {})
        return self.get_claim(claim_id)

    @staticmethod
    def _claim_field(claim: Claim, path: str) -> Any:
        if path in {"subject", "predicate", "object", "valid_from", "valid_to", "time_precision", "truth_status", "lifecycle_status", "use_policy", "utility_status"}:
            return getattr(claim, path)
        if path.startswith("object.") and isinstance(claim.object, dict):
            value: Any = claim.object
            for part in path.split(".")[1:]:
                if not isinstance(value, dict) or part not in value:
                    raise ValueError(f"unknown claim field path: {path}")
                value = value[part]
            return value
        raise ValueError(f"unsupported claim field path: {path}")

    @staticmethod
    def _set_claim_field(claim: Claim, path: str, value: Any) -> None:
        if path in {"subject", "predicate", "object", "valid_from", "valid_to", "time_precision", "truth_status", "lifecycle_status", "use_policy", "utility_status"}:
            setattr(claim, path, value)
            return
        if path.startswith("object.") and isinstance(claim.object, dict):
            target: Any = claim.object
            parts = path.split(".")[1:]
            for part in parts[:-1]:
                if not isinstance(target, dict) or part not in target:
                    raise ValueError(f"unknown claim field path: {path}")
                target = target[part]
            if not isinstance(target, dict):
                raise ValueError(f"unknown claim field path: {path}")
            target[parts[-1]] = value
            return
        raise ValueError(f"unsupported claim field path: {path}")

    @staticmethod
    def _validate_time(valid_from: Optional[str], valid_to: Optional[str], precision: Optional[str]) -> None:
        if precision is not None and precision not in {"year", "month", "day", "minute", "second", "unknown"}:
            raise ValueError(f"invalid time_precision: {precision}")
        parsed_from = SQLiteStore._parse_time(valid_from)
        parsed_to = SQLiteStore._parse_time(valid_to)
        if parsed_from is not None and parsed_to is not None and parsed_from > parsed_to:
            raise ValueError("valid_from must not be after valid_to")

    @staticmethod
    def _parse_time(value: Optional[str]) -> Optional[datetime]:
        if value is None:
            return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            try:
                parsed = None
                for pattern in ("%Y-%m-%d", "%Y/%m/%d", "%Y/%m/%d (%a) %H:%M"):
                    try:
                        parsed = datetime.strptime(value, pattern).replace(tzinfo=timezone.utc)
                        break
                    except ValueError:
                        continue
                if parsed is None:
                    raise ValueError(value)
            except ValueError:
                raise ValueError(f"invalid ISO time: {value}") from error
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def list_claims(self) -> List[Claim]:
        with self._open() as connection:
            rows = connection.execute("SELECT * FROM claims ORDER BY created_at").fetchall()
            evidence_rows = connection.execute(
                "SELECT claim_id, observation_id FROM evidence_links ORDER BY created_at"
            ).fetchall()
        evidence_by_claim: Dict[str, List[str]] = {}
        for evidence in evidence_rows:
            evidence_by_claim.setdefault(evidence["claim_id"], []).append(evidence["observation_id"])
        claims = []
        for row in rows:
            claim = self._claim_from_row(row)
            claim.source_observation_ids = evidence_by_claim.get(claim.claim_id, [])
            claims.append(claim)
        return claims

    def index_claim(self, claim: Claim) -> None:
        with self._open() as connection:
            connection.execute("DELETE FROM claims_fts WHERE claim_id = ?", (claim.claim_id,))
            if claim.lifecycle_status != "deleted" and claim.use_policy != "blocked":
                connection.execute(
                    "INSERT INTO claims_fts(claim_id, statement, predicate, scope_repository) VALUES (?, ?, ?, ?)",
                    (claim.claim_id, claim.statement(), claim.predicate, claim.scope.repository),
                )

    def rebuild_indexes(self) -> int:
        claims = self.list_claims()
        with self._open() as connection:
            connection.execute("DELETE FROM claims_fts")
            connection.execute("DELETE FROM observations_fts")
            for claim in claims:
                if claim.lifecycle_status != "deleted" and claim.use_policy != "blocked":
                    connection.execute(
                        "INSERT INTO claims_fts(claim_id, statement, predicate, scope_repository) VALUES (?, ?, ?, ?)",
                        (claim.claim_id, claim.statement(), claim.predicate, claim.scope.repository),
                    )
            observation_rows = connection.execute(
                "SELECT observation_id, content, source_type, scope_json FROM observations ORDER BY observation_id"
            ).fetchall()
            for observation in observation_rows:
                connection.execute(
                    "INSERT INTO observations_fts(observation_id, content, source_type, scope_repository) VALUES (?, ?, ?, ?)",
                    (observation["observation_id"], observation["content"], observation["source_type"],
                     json.loads(observation["scope_json"])["repository"]),
                )
        return len(claims)

    def search_observation_records(self, query: str, scope: Scope, *, limit: int = 10,
                                   tokens: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        search_tokens = tokens or [token for token in __import__("re").findall(r"[\w-]+", query) if len(token) > 1]
        if not search_tokens:
            return []
        match = " OR ".join('"' + token.replace('"', '""') + '"' for token in search_tokens)
        scope_values = (scope.repository, scope.branch, scope.service, scope.module, scope.file, scope.session, scope.task)
        scope_sql = """json_extract(o.scope_json, '$.repository') = ?
            AND (json_extract(o.scope_json, '$.branch') IS NULL OR json_extract(o.scope_json, '$.branch') = ?)
            AND (json_extract(o.scope_json, '$.service') IS NULL OR json_extract(o.scope_json, '$.service') = ?)
            AND (json_extract(o.scope_json, '$.module') IS NULL OR json_extract(o.scope_json, '$.module') = ?)
            AND (json_extract(o.scope_json, '$.file') IS NULL OR json_extract(o.scope_json, '$.file') = ?)
            AND (json_extract(o.scope_json, '$.session') IS NULL OR json_extract(o.scope_json, '$.session') = ?)
            AND (json_extract(o.scope_json, '$.task') IS NULL OR json_extract(o.scope_json, '$.task') = ?)"""
        with self._open() as connection:
            try:
                rows = connection.execute(
                    f"""SELECT o.*, bm25(observations_fts) AS fts_score
                        FROM observations_fts JOIN observations o ON o.observation_id = observations_fts.observation_id
                        WHERE observations_fts MATCH ? AND {scope_sql}
                        ORDER BY fts_score, o.observed_at DESC LIMIT ?""",
                    (match, *scope_values, max(limit, 1)),
                ).fetchall()
            except sqlite3.OperationalError:
                rows = []
        result = []
        for row in rows:
            metadata = json.loads(row["metadata_json"] or "{}")
            result.append({
                "record_type": "observation", "record_id": str(row["observation_id"]),
                "observation_id": str(row["observation_id"]), "claim_id": None,
                "content": str(row["content"]), "observed_at": str(row["observed_at"]),
                "metadata": metadata, "fts_score": float(row["fts_score"] or 0.0),
            })
        return result

    def search_claims(self, query: str, scope: Scope, limit: int = 10) -> List[Claim]:
        return [record["claim"] for record in self.search_claim_records(query, scope, limit=limit)]

    def search_claim_records(self, query: str, scope: Scope, *, limit: int = 10,
                             as_of: Optional[str] = None, tokens: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        search_tokens = tokens or [token for token in __import__("re").findall(r"[\w-]+", query) if len(token) > 1]
        if not search_tokens:
            return []
        match = " OR ".join('"' + token.replace('"', '""') + '"' for token in search_tokens)
        current_time = self._parse_time(as_of) if as_of else datetime.now(timezone.utc)
        scope_values = (scope.repository, scope.branch, scope.service, scope.module, scope.file, scope.session, scope.task)
        scope_sql = """json_extract(c.scope_json, '$.repository') = ?
            AND (json_extract(c.scope_json, '$.branch') IS NULL OR json_extract(c.scope_json, '$.branch') = ?)
            AND (json_extract(c.scope_json, '$.service') IS NULL OR json_extract(c.scope_json, '$.service') = ?)
            AND (json_extract(c.scope_json, '$.module') IS NULL OR json_extract(c.scope_json, '$.module') = ?)
            AND (json_extract(c.scope_json, '$.file') IS NULL OR json_extract(c.scope_json, '$.file') = ?)
            AND (json_extract(c.scope_json, '$.session') IS NULL OR json_extract(c.scope_json, '$.session') = ?)
            AND (json_extract(c.scope_json, '$.task') IS NULL OR json_extract(c.scope_json, '$.task') = ?)"""
        with self._open() as connection:
            try:
                rows = connection.execute(
                    "SELECT c.*, bm25(claims_fts) AS fts_score FROM claims_fts JOIN claims c ON c.claim_id = claims_fts.claim_id "
                    + "WHERE claims_fts MATCH ? AND " + scope_sql
                     + (" AND c.lifecycle_status NOT IN ('deleted', 'expired', 'archived')" if as_of else " AND c.lifecycle_status NOT IN ('deleted', 'superseded', 'expired', 'archived')")
                    + " AND c.use_policy != 'blocked' ORDER BY fts_score, c.updated_at DESC LIMIT ?",
                    (match, *scope_values, max(limit * 4, 20)),
                ).fetchall()
            except sqlite3.OperationalError:
                rows = connection.execute(
                    "SELECT c.*, 0.0 AS fts_score FROM claims c WHERE " + scope_sql
                     + (" AND c.lifecycle_status NOT IN ('deleted', 'expired', 'archived')" if as_of else " AND c.lifecycle_status NOT IN ('deleted', 'superseded', 'expired', 'archived')")
                    + " AND c.use_policy != 'blocked' ORDER BY c.updated_at DESC LIMIT ?",
                    (*scope_values, max(limit * 4, 20)),
                ).fetchall()
            claim_ids = [str(row["claim_id"]) for row in rows]
            evidence_by_claim: Dict[str, List[sqlite3.Row]] = {}
            if claim_ids:
                placeholders = ",".join("?" for _ in claim_ids)
                evidence_rows = connection.execute(
                    "SELECT e.claim_id, e.observation_id, e.relation, o.content, o.observed_at, o.metadata_json "
                    + "FROM evidence_links e JOIN observations o ON o.observation_id = e.observation_id "
                    + "WHERE e.claim_id IN (" + placeholders + ") ORDER BY e.created_at",
                    claim_ids,
                ).fetchall()
                for evidence in evidence_rows:
                    evidence_by_claim.setdefault(str(evidence["claim_id"]), []).append(evidence)
        records: List[Dict[str, Any]] = []
        for row in rows:
            claim = self._claim_from_row(row)
            evidence = evidence_by_claim.get(claim.claim_id, [])
            claim.source_observation_ids = [str(item["observation_id"]) for item in evidence]
            valid_from = self._parse_time(claim.valid_from)
            valid_to = self._parse_time(claim.valid_to)
            if valid_from is not None and valid_from > current_time:
                continue
            if valid_to is not None and current_time >= valid_to:
                continue
            if not evidence:
                records.append({"record_type": "claim", "record_id": claim.claim_id, "claim": claim,
                                "claim_id": claim.claim_id, "observation_id": None, "relation": None,
                                "content": claim.statement(), "observed_at": claim.updated_at,
                                "metadata": {}, "fts_score": float(row["fts_score"] or 0.0)})
                continue
            for item in evidence:
                records.append({
                    "record_type": "claim", "record_id": claim.claim_id, "claim": claim,
                    "claim_id": claim.claim_id, "observation_id": str(item["observation_id"]),
                    "relation": str(item["relation"]), "content": str(item["content"]),
                    "observed_at": str(item["observed_at"]), "metadata": json.loads(item["metadata_json"] or "{}"),
                    "fts_score": float(row["fts_score"] or 0.0),
                })
        return records

    def add_claim_relation(self, relation: ClaimRelation) -> Tuple[ClaimRelation, bool]:
        self._validate_time(relation.effective_at, None, None)
        with self.transaction() as connection:
            source = connection.execute("SELECT scope_json FROM claims WHERE claim_id = ?", (relation.from_claim_id,)).fetchone()
            target = connection.execute("SELECT scope_json FROM claims WHERE claim_id = ?", (relation.to_claim_id,)).fetchone()
            if not source or not target:
                raise KeyError("claim relation references an unknown claim")
            source_scope = Scope(**json.loads(source["scope_json"]))
            target_scope = Scope(**json.loads(target["scope_json"]))
            if source_scope.repository != target_scope.repository:
                raise ValueError("claim relation claims must share a repository")
            existing = connection.execute(
                "SELECT * FROM claim_relations WHERE idempotency_key = ?", (relation.idempotency_key,)
            ).fetchone()
            if existing:
                return self._relation_from_row(existing), False
            if relation.relation_type in {"supersedes", "refines", "retracts"}:
                rows = connection.execute(
                    "SELECT from_claim_id, to_claim_id, relation_type FROM claim_relations "
                    "WHERE relation_type IN ('supersedes','refines','retracts')"
                ).fetchall()
                edges: Dict[str, List[str]] = {}
                for row in rows:
                    edges.setdefault(str(row["from_claim_id"]), []).append(str(row["to_claim_id"]))
                pending = [relation.to_claim_id]
                visited = set()
                while pending:
                    current = pending.pop()
                    if current == relation.from_claim_id:
                        raise ValueError("claim relation would create a directed cycle")
                    if current in visited:
                        continue
                    visited.add(current)
                    pending.extend(edges.get(current, []))
            connection.execute(
                """INSERT INTO claim_relations
                (relation_id, from_claim_id, to_claim_id, relation_type, assertion_family_key,
                 covered_paths_json, effective_at, source_observation_id, metadata_json, created_at, trace_id, idempotency_key)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (relation.relation_id, relation.from_claim_id, relation.to_claim_id, relation.relation_type,
                 relation.assertion_family_key, json.dumps(relation.covered_paths, ensure_ascii=False), relation.effective_at,
                 relation.source_observation_id, json.dumps(relation.metadata, ensure_ascii=False, sort_keys=True),
                 relation.created_at, relation.trace_id, relation.idempotency_key),
            )
            relation_is_effective = not relation.effective_at or self._parse_time(relation.effective_at) <= datetime.now(timezone.utc)
            if relation_is_effective and relation.relation_type == "supersedes":
                connection.execute(
                    "UPDATE claims SET lifecycle_status = 'superseded' WHERE claim_id = ? AND lifecycle_status NOT IN ('deleted','archived')",
                    (relation.to_claim_id,),
                )
            elif relation_is_effective and relation.relation_type == "retracts":
                connection.execute(
                    "UPDATE claims SET lifecycle_status = 'deleted', truth_status = 'retracted', use_policy = 'blocked' WHERE claim_id = ?",
                    (relation.to_claim_id,),
                )
            self._operation(connection, "claim_relation_add", relation.relation_id,
                            relation.idempotency_key, relation.trace_id,
                            {"from_claim_id": relation.from_claim_id, "to_claim_id": relation.to_claim_id,
                             "relation_type": relation.relation_type})
        self.rebuild_indexes()
        return relation, True

    def list_claim_relations(self, assertion_family_key: Optional[str] = None,
                             *, as_of: Optional[str] = None) -> List[ClaimRelation]:
        sql = "SELECT * FROM claim_relations"
        params: List[Any] = []
        clauses = []
        if assertion_family_key is not None:
            clauses.append("assertion_family_key = ?")
            params.append(assertion_family_key)
        if as_of is not None:
            clauses.append("(effective_at IS NULL OR effective_at <= ?)")
            params.append(as_of)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY COALESCE(effective_at, created_at), created_at, relation_id"
        with self._open() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [self._relation_from_row(row) for row in rows]

    def claim_state_closure(self, seed_claim_ids: Iterable[str], *, as_of: Optional[str] = None,
                            max_claims: int = 256) -> Tuple[List[Claim], Dict[str, Any]]:
        seeds = list(dict.fromkeys(str(value) for value in seed_claim_ids if value))
        if not seeds:
            return [], {"seed_count": 0, "closure_count": 0, "closure_truncated": False}
        anchor = self._parse_time(as_of) if as_of else datetime.now(timezone.utc)
        with self._open() as connection:
            rows = connection.execute(
                "SELECT * FROM claims WHERE claim_id IN (" + ",".join("?" for _ in seeds) + ")", seeds
            ).fetchall()
            family_keys = {str(row["assertion_family_key"] or "") for row in rows}
            family_keys.discard("")
            if family_keys:
                placeholders = ",".join("?" for _ in family_keys)
                rows = connection.execute(
                    "SELECT * FROM claims WHERE assertion_family_key IN (" + placeholders + ")",
                    list(family_keys),
                ).fetchall()
            claims_by_id = {str(row["claim_id"]): row for row in rows}
            relation_rows = connection.execute("SELECT * FROM claim_relations ORDER BY created_at, relation_id").fetchall()
            relations = [self._relation_from_row(row) for row in relation_rows
                         if not row["effective_at"] or self._parse_time(row["effective_at"]) <= anchor]
            related_ids = set(seeds)
            related_ids.update(claims_by_id)
            changed = True
            while changed:
                changed = False
                for relation in relations:
                    if relation.from_claim_id in related_ids or relation.to_claim_id in related_ids:
                        for claim_id in (relation.from_claim_id, relation.to_claim_id):
                            if claim_id in claims_by_id and claim_id not in related_ids:
                                related_ids.add(claim_id)
                                changed = True
            ordered_ids = [claim_id for claim_id in claims_by_id if claim_id in related_ids]
            truncated = len(ordered_ids) > max_claims
            ordered_ids = ordered_ids[:max_claims]
            selected_rows = [claims_by_id[claim_id] for claim_id in ordered_ids]
            evidence_rows = []
            if ordered_ids:
                placeholders = ",".join("?" for _ in ordered_ids)
                evidence_rows = connection.execute(
                    "SELECT claim_id, observation_id FROM evidence_links WHERE claim_id IN (" + placeholders + ") ORDER BY created_at",
                    ordered_ids,
                ).fetchall()
        evidence_by_claim: Dict[str, List[str]] = {}
        for row in evidence_rows:
            evidence_by_claim.setdefault(str(row["claim_id"]), []).append(str(row["observation_id"]))
        claims = []
        for row in selected_rows:
            claim = self._claim_from_row(row)
            claim.source_observation_ids = evidence_by_claim.get(claim.claim_id, [])
            claims.append(claim)
        return claims, {"seed_count": len(seeds), "closure_count": len(claims),
                        "closure_truncated": truncated, "family_count": len(family_keys)}

    def rebuild_state(self) -> Dict[str, Any]:
        from pimem.core.state_machine import resolve_claim_state
        claims = self.list_claims()
        relations = self.list_claim_relations()
        by_family: Dict[str, List[Claim]] = {}
        for claim in claims:
            by_family.setdefault(claim.assertion_family_key or claim.fingerprint(), []).append(claim)
        resolved = 0
        with self.transaction() as connection:
            for family_claims in by_family.values():
                selected, _ = resolve_claim_state(family_claims, relations=relations)
                selected_ids = {claim.claim_id for claim in selected}
                for claim in family_claims:
                    if claim.lifecycle_status in {"deleted", "archived"}:
                        continue
                    status = "active" if claim.claim_id in selected_ids else claim.lifecycle_status
                    connection.execute("UPDATE claims SET lifecycle_status = ? WHERE claim_id = ?", (status, claim.claim_id))
                    resolved += 1
        return {"claim_count": len(claims), "relation_count": len(relations), "resolved_count": resolved}

    def rebuild_all_derived(self) -> Dict[str, Any]:
        state = self.rebuild_state()
        state["indexed_claims"] = self.rebuild_indexes()
        return state

    def explain(self, claim_id: str) -> Dict[str, Any]:
        claim = self.get_claim(claim_id)
        with self._open() as connection:
            rows = connection.execute(
                "SELECT observation_id, relation, created_at FROM evidence_links WHERE claim_id = ? ORDER BY created_at",
                (claim_id,),
            ).fetchall()
        return {"claim": self.claim_to_dict(claim), "evidence": [dict(row) for row in rows]}

    def forget_claim(self, claim_id: str, policy: str = "local_runtime") -> None:
        if policy != "local_runtime":
            raise ValueError("V0 only supports forget policy local_runtime")
        with self.transaction() as connection:
            row = connection.execute("SELECT revision FROM claims WHERE claim_id = ?", (claim_id,)).fetchone()
            if not row:
                raise KeyError(f"unknown claim: {claim_id}")
            connection.execute(
                "UPDATE claims SET truth_status = 'retracted', lifecycle_status = 'deleted', use_policy = 'blocked', revision = revision + 1, updated_at = ? WHERE claim_id = ?",
                (utc_now(), claim_id),
            )
            self._operation(connection, "forget", claim_id, f"forget:{claim_id}:{row['revision'] + 1}", None, {"policy": policy})
        with self._open() as connection:
            connection.execute("DELETE FROM claims_fts WHERE claim_id = ?", (claim_id,))

    def record_operation(self, operation_type: str, target_id: Optional[str], idempotency_key: str,
                         trace_id: Optional[str], details: Dict[str, Any]) -> None:
        with self._open() as connection:
            self._operation(connection, operation_type, target_id, idempotency_key, trace_id, details)

    def export(self, output_dir: str) -> Dict[str, Any]:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        with self._open() as connection:
            tables = ("repositories", "observations", "claims", "evidence_links", "memory_operations")
            counts = {}
            for table in tables:
                rows = connection.execute(f"SELECT * FROM {table}").fetchall()
                counts[table] = len(rows)
                with (destination / f"{table}.jsonl").open("w", encoding="utf-8") as handle:
                    for row in rows:
                        value = dict(row)
                        serialized = json.dumps(value, ensure_ascii=False, sort_keys=True)
                        import re
                        if re.search(r"-----BEGIN [A-Z ]*PRIVATE KEY-----|\b(?:password|passwd|pwd)\s*[:=]\s*\S+|\b(?:api[_-]?key|secret[_-]?key|access[_-]?token)\s*[:=]\s*['\"]?[A-Za-z0-9._~+/=-]{12,}", serialized, re.IGNORECASE):
                            raise ValueError("export_rejected_sensitive")
                        handle.write(serialized + "\n")
        manifest = {"schema_version": "memory.export.experimental", "exported_at": utc_now(), "counts": counts}
        (destination / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return manifest

    def _operation(self, connection: sqlite3.Connection, operation_type: str, target_id: Optional[str],
                   idempotency_key: str, trace_id: Optional[str], details: Dict[str, Any]) -> None:
        connection.execute(
            "INSERT OR IGNORE INTO memory_operations(operation_id, operation_type, target_id, idempotency_key, trace_id, details_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("op_" + uuid.uuid4().hex, operation_type, target_id, idempotency_key, trace_id,
             json.dumps(details, ensure_ascii=False, sort_keys=True), utc_now()),
        )

    @staticmethod
    def _relation_from_row(row: sqlite3.Row) -> ClaimRelation:
        return ClaimRelation(
            relation_id=row["relation_id"], from_claim_id=row["from_claim_id"], to_claim_id=row["to_claim_id"],
            relation_type=row["relation_type"], assertion_family_key=row["assertion_family_key"],
            covered_paths=json.loads(row["covered_paths_json"] or "[]"), effective_at=row["effective_at"],
            source_observation_id=row["source_observation_id"], metadata=json.loads(row["metadata_json"] or "{}"),
            created_at=row["created_at"], trace_id=row["trace_id"], idempotency_key=row["idempotency_key"],
        )

    @staticmethod
    def _observation_from_row(row: sqlite3.Row) -> Observation:
        return Observation(id=row["observation_id"], content=row["content"], source_type=row["source_type"],
                           source_ref=row["source_ref"], scope=Scope(**json.loads(row["scope_json"])),
                           observed_at=row["observed_at"], recorded_at=row["recorded_at"],
                           content_hash=row["content_hash"], sensitivity=row["sensitivity"],
                           idempotency_key=row["idempotency_key"], trace_id=row["trace_id"],
                           metadata=json.loads(row["metadata_json"]))

    @staticmethod
    def _claim_from_row(row: sqlite3.Row) -> Claim:
        return Claim(claim_id=row["claim_id"], subject=row["subject"], predicate=row["predicate"],
                     object=json.loads(row["object_json"]), scope=Scope(**json.loads(row["scope_json"])),
                     source_observation_ids=[], valid_from=row["valid_from"], valid_to=row["valid_to"],
                     time_precision=row["time_precision"], truth_status=row["truth_status"],
                     lifecycle_status=row["lifecycle_status"], use_policy=row["use_policy"],
                     utility_status=row["utility_status"], revision=row["revision"],
                     created_at=row["created_at"], updated_at=row["updated_at"],
                     idempotency_key=row["idempotency_key"], object_path=row["object_path"] if "object_path" in row.keys() else None,
                     event_type=row["event_type"] if "event_type" in row.keys() and row["event_type"] else "state",
                     event_time=row["event_time"] if "event_time" in row.keys() else None,
                     assertion_family_key=row["assertion_family_key"] if "assertion_family_key" in row.keys() else None)

    @staticmethod
    def claim_to_dict(claim: Claim) -> Dict[str, Any]:
        value = claim.__dict__.copy()
        value["scope"] = claim.scope.as_dict()
        return value

    @staticmethod
    def claim_from_dict(value: Dict[str, Any]) -> Claim:
        value = dict(value)
        value["scope"] = Scope(**value["scope"])
        return Claim(**value)
