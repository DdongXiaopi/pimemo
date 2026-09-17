"""Regression tests for hard-budget pack slimming (2026-09-13).

The hard-budget pack used to serialize the same evidence several times:
the anchor list was aliased under both ``selected`` and ``episodic_candidates``,
and the bundle list under both ``episode_bundles`` and ``evidence_bundles``.
These tests pin the de-duplicated behaviour.
"""
from __future__ import annotations

import unittest
from typing import Any, Dict, List

from pimem.retrieval.compiler import (
    _compact_episode_bundle,
    _compact_hard_episodic_candidate,
    _minimal_budget_pack,
    ConservativeTokenEstimator,
)


def _candidate(record_id: str, text: str = "answer text") -> Dict[str, Any]:
    return {
        "record_type": "answer_candidate", "record_id": record_id,
        "candidate_id": record_id, "observation_id": "obs-" + record_id,
        "source_ids": ["obs-" + record_id], "episode_id": "episode-1",
        "session_id": "session-1", "turn_id": "turn-1", "turn_index": 3,
        "role": "assistant", "speech_act": "answer", "answer_shape": "literal",
        "score": 10.0, "text": text, "answer_span": {"kind": "answer_span", "text": "long span " * 200},
        "source_span": [{"kind": "sentence", "text": "span %d" % i} for i in range(12)],
        "answer_units": [{"kind": "sentence", "value": "unit %d" % i} for i in range(20)],
        "facets": [{"kind": "facet", "value": "facet %d" % i} for i in range(20)],
        "structured_facts": [{"kind": "fact", "value": "fact %d" % i} for i in range(20)],
        "question_text": "the original question " * 20,
        "provenance": {"episode_id": "episode-1", "turn_id": "turn-1", "observation_id": "obs-x",
                       "extra": {"blob": "x" * 400}},
    }


def _bundle(session_id: str) -> Dict[str, Any]:
    return {
        "bundle_id": "bundle-1", "episode_id": "episode-1", "session_id": session_id,
        "anchor_candidates": ["candidate:a"], "question_turn_ids": ["turn:q"],
        "answer_turn_ids": ["turn:target"], "primary_answer_candidate_ids": ["candidate:a"],
        "primary_answer_turn_ids": ["turn:target"], "question_pair_ids": ["turn:q"],
        "required_answer_units": [{"kind": "sentence", "value": "u%d" % i} for i in range(96)],
        "optional_related_turn_ids": ["turn:r1", "turn:r2"], "related_turn_ids": ["turn:r1"],
        "chronological_order": [1, 2], "role_sequence": ["user", "assistant"],
        "source_observations": ["obs-a"], "coverage": {"primary_answer": True},
        "confidence": 0.9, "reasons": ["episode_seed"],
        "structured_facts": [{"kind": "fact", "value": "f"}] * 96,
        "facets": [{"kind": "facet", "value": "f"}] * 96,
        "answer_units": [{"kind": "sentence", "value": "u"}] * 96,
        "source_spans": [{"kind": "sentence", "text": "s"}] * 96,
    }


class HardCandidateSlimTests(unittest.TestCase):
    def test_hard_candidate_drops_pure_duplicates(self):
        compact = _compact_hard_episodic_candidate(_candidate("candidate:a"))
        for absent in ("object", "question_text", "provenance"):
            self.assertNotIn(absent, compact)
        self.assertTrue(compact["text"])
        self.assertTrue(compact["evidence_text"])

    def test_hard_candidate_caps_redundant_lists(self):
        compact = _compact_hard_episodic_candidate(_candidate("candidate:a"))
        self.assertLessEqual(len(compact["source_span"]), 6)
        self.assertLessEqual(len(compact["answer_units"]), 12)
        self.assertLessEqual(len(compact["facets"]), 8)
        self.assertLessEqual(len(compact["structured_facts"]), 8)


class SlimBundleTests(unittest.TestCase):
    def test_slim_bundle_drops_evidence_duplicates(self):
        slim = _compact_episode_bundle(_bundle("session-1"), include_answer_candidates=False,
                                        include_turn_text=False, slim_evidence=True)
        for absent in ("required_answer_units", "structured_facts", "facets",
                       "answer_units", "source_spans"):
            self.assertNotIn(absent, slim)
        # Identity and turn anchors stay.
        self.assertEqual(slim["episode_id"], "episode-1")
        self.assertEqual(slim["answer_turn_ids"], ["turn:target"])

    def test_default_bundle_keeps_evidence_fields(self):
        full = _compact_episode_bundle(_bundle("session-1"))
        self.assertIn("required_answer_units", full)
        self.assertIn("answer_units", full)


class MinimalPackDedupTests(unittest.TestCase):
    def test_minimal_pack_does_not_alias_evidence_views(self):
        anchors_data: List[Dict[str, Any]] = [_candidate("candidate:a")]
        base = {"schema_version": "semantic-context-pack.v0.6", "semantic_mode": "episode_evidence",
                "query": "q", "scope": {}, "query_plan": {}, "token_budget": 4096,
                "budget_unit": "pimem_internal_estimate",
                "episode_bundles": [_bundle("session-1")]}
        pack = _minimal_budget_pack(base, [], anchors_data, [], [], ConservativeTokenEstimator())
        self.assertIsNot(pack["selected"], pack["episodic_candidates"])
        self.assertIsNot(pack["episode_bundles"], pack["evidence_bundles"])
        # selected is metadata-only; the evidence text lives on episodic_candidates.
        self.assertFalse(any("text" in item for item in pack["selected"]))
        self.assertTrue(all(item.get("text") for item in pack["episodic_candidates"]))
        # evidence_bundles is a lightweight session index, not a bundle copy.
        for entry in pack["evidence_bundles"]:
            self.assertIn("session_id", entry)
            self.assertNotIn("required_answer_units", entry)
        # answer_span text is capped.
        for item in pack["episodic_candidates"]:
            span = item.get("answer_span") or {}
            self.assertLessEqual(len(str(span.get("text") or "")), 400)


if __name__ == "__main__":
    unittest.main()
