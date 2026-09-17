SCHEMA = """
CREATE TABLE IF NOT EXISTS repositories (
    repository_id TEXT PRIMARY KEY,
    canonical_path TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS observations (
    observation_id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    source_type TEXT NOT NULL,
    source_ref TEXT NOT NULL,
    scope_json TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    sensitivity TEXT NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE,
    trace_id TEXT,
    metadata_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS claim_candidates (
    candidate_id TEXT PRIMARY KEY,
    claim_json TEXT NOT NULL,
    observation_id TEXT NOT NULL REFERENCES observations(observation_id),
    created_at TEXT NOT NULL,
    UNIQUE(candidate_id)
);
CREATE TABLE IF NOT EXISTS claims (
    claim_id TEXT PRIMARY KEY,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object_json TEXT NOT NULL,
    scope_json TEXT NOT NULL,
    valid_from TEXT,
    valid_to TEXT,
    time_precision TEXT,
    truth_status TEXT NOT NULL,
    lifecycle_status TEXT NOT NULL,
    use_policy TEXT NOT NULL,
    utility_status TEXT NOT NULL,
    revision INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE,
    object_path TEXT,
    event_type TEXT NOT NULL DEFAULT 'state',
    event_time TEXT,
    assertion_family_key TEXT
);
CREATE TABLE IF NOT EXISTS evidence_links (
    evidence_id TEXT PRIMARY KEY,
    claim_id TEXT NOT NULL REFERENCES claims(claim_id),
    observation_id TEXT NOT NULL REFERENCES observations(observation_id),
    relation TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(claim_id, observation_id, relation)
);
CREATE TABLE IF NOT EXISTS claim_relations (
    relation_id TEXT PRIMARY KEY,
    from_claim_id TEXT NOT NULL REFERENCES claims(claim_id),
    to_claim_id TEXT NOT NULL REFERENCES claims(claim_id),
    relation_type TEXT NOT NULL,
    assertion_family_key TEXT NOT NULL,
    covered_paths_json TEXT NOT NULL,
    effective_at TEXT,
    source_observation_id TEXT REFERENCES observations(observation_id),
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    trace_id TEXT,
    idempotency_key TEXT NOT NULL UNIQUE,
    CHECK(from_claim_id <> to_claim_id),
    CHECK(relation_type IN ('duplicate','supersedes','refines','contradicts','retracts'))
);
CREATE TABLE IF NOT EXISTS memory_operations (
    operation_id TEXT PRIMARY KEY,
    operation_type TEXT NOT NULL,
    target_id TEXT,
    idempotency_key TEXT NOT NULL UNIQUE,
    trace_id TEXT,
    details_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_claims_scope ON claims(scope_json, predicate, lifecycle_status);
CREATE INDEX IF NOT EXISTS idx_claims_revision ON claims(revision);
CREATE INDEX IF NOT EXISTS idx_claims_family ON claims(assertion_family_key, lifecycle_status);
CREATE INDEX IF NOT EXISTS idx_evidence_claim ON evidence_links(claim_id);
CREATE INDEX IF NOT EXISTS idx_evidence_observation ON evidence_links(observation_id);
CREATE INDEX IF NOT EXISTS idx_relations_from ON claim_relations(from_claim_id, effective_at);
CREATE INDEX IF NOT EXISTS idx_relations_to ON claim_relations(to_claim_id, effective_at);
CREATE INDEX IF NOT EXISTS idx_relations_family ON claim_relations(assertion_family_key, effective_at);
CREATE INDEX IF NOT EXISTS idx_operations_trace_time ON memory_operations(trace_id, created_at);
CREATE VIRTUAL TABLE IF NOT EXISTS claims_fts USING fts5(
    claim_id UNINDEXED,
    statement,
    predicate,
    scope_repository UNINDEXED
);
CREATE VIRTUAL TABLE IF NOT EXISTS observations_fts USING fts5(
    observation_id UNINDEXED,
    content,
    source_type UNINDEXED,
    scope_repository UNINDEXED
);
"""
