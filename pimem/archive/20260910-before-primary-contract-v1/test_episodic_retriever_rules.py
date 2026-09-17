import unittest

from pimem.retrieval.episodic_retriever import (
    _focused_session_closure,
    _ordered_session_candidates,
    _primary_answer_closure,
)
from pimem.retrieval.query_plan import build_query_plan


class EpisodicRetrieverRuleTests(unittest.TestCase):
    def _item(self, **kwargs):
        base = {
            "record_id": kwargs.get("record_id", f"record:{kwargs['turn_index']}") ,
            "session_id": "session:test",
            "turn_index": kwargs.get("turn_index", 0),
            "role": kwargs.get("role", "assistant"),
            "speech_act": kwargs.get("speech_act", "assistant_answer"),
            "pair_type": kwargs.get("pair_type", "nearest_preceding_user"),
            "score": kwargs.get("score", 0),
            "content": kwargs.get("content", ""),
            "question_turn_id": kwargs.get("question_turn_id", "q1"),
            "requested_entity_hits": kwargs.get("requested_entity_hits", 0),
            "literal_hits": kwargs.get("literal_hits", 0),
            "ordinal_match": kwargs.get("ordinal_match", 0),
            "typed_field_match": kwargs.get("typed_field_match", 0),
            "currency_relevance": kwargs.get("currency_relevance", 0),
            "numeric_hits": kwargs.get("numeric_hits", 0),
            "relation_hits": kwargs.get("relation_hits", 0),
            "unit_score": kwargs.get("unit_score", 0),
            "typed_relation_score": kwargs.get("typed_relation_score", 0),
            "answer_overlap": kwargs.get("answer_overlap", 0),
            "question_overlap": kwargs.get("question_overlap", 0),
            "question_entity_hits": kwargs.get("question_entity_hits", 0),
            "answer_anchor_hits": kwargs.get("answer_anchor_hits", 0),
            "answer_shape": kwargs.get("answer_shape", "entity"),
            "answer_units": kwargs.get("answer_units", []),
        }
        return base

    def test_prefers_answer_like_turn_over_prompt(self):
        plan = build_query_plan(
            "I was looking back at our previous conversation about buying unique engagement rings directly from designers. "
            "Can you remind me of the Instagram handle of the UK-based designer who works with unusual gemstones?"
        )
        items = [
            self._item(
                turn_index=0,
                role="user",
                speech_act="user_statement",
                pair_type="user_without_preceding_question",
                score=1390,
                content="Write a blog post about guidance on how to buy a unique engagement rings direct from the designer",
                question_turn_id=None,
            ),
            self._item(
                turn_index=1,
                role="assistant",
                speech_act="assistant_recommendation",
                pair_type="nearest_preceding_user",
                score=1143,
                content="1. Catbird: ... @jessica_poole_jewellery",
                question_turn_id="q1",
                requested_entity_hits=1,
                literal_hits=1,
                unit_score=212,
                answer_overlap=7,
                answer_units=[{"kind": "quoted_span", "text": "@jessica_poole_jewellery"}],
            ),
        ]
        closure = _focused_session_closure(items, 1, plan)
        self.assertEqual([item["turn_index"] for item in closure], [1])

    def test_prefers_earliest_valid_answer_revision(self):
        plan = build_query_plan("I wanted to follow up on our previous conversation about posture videos. Can you remind me what video you recommended?")
        items = [
            self._item(
                turn_index=1,
                role="assistant",
                speech_act="assistant_recommendation",
                pair_type="nearest_preceding_user",
                score=200,
                content="Mayo Clinic video recommendation",
                question_turn_id="q1",
                requested_entity_hits=1,
                answer_overlap=4,
                answer_units=[{"kind": "quoted_span", "text": "Mayo Clinic"}],
            ),
            self._item(
                turn_index=5,
                role="assistant",
                speech_act="assistant_answer",
                pair_type="nearest_preceding_user",
                score=9999,
                content="Another later answer revision",
                question_turn_id="q1",
                requested_entity_hits=1,
                answer_overlap=10,
                answer_units=[{"kind": "quoted_span", "text": "Mayo Clinic"}],
            ),
        ]
        closure = _focused_session_closure(items, 1, plan)
        self.assertEqual([item["turn_index"] for item in closure], [1])

    def test_relevant_question_pair_beats_generic_later_answer_anchor(self):
        plan = build_query_plan(
            "I wanted to follow up on our previous conversation about two-factor authentication. "
            "Can you remind me what methods you mentioned?"
        )
        items = [
            self._item(
                turn_index=1,
                speech_act="assistant_answer",
                score=100,
                question_overlap=4,
                answer_anchor_hits=2,
                answer_units=[{"kind": "sentence", "text": "Biometric authentication or one-time passwords."}],
            ),
            self._item(
                turn_index=5,
                speech_act="assistant_answer",
                score=9999,
                question_overlap=0,
                answer_anchor_hits=9,
                answer_units=[{"kind": "sentence", "text": "Fingerprint or facial recognition."}],
            ),
        ]
        closure = _focused_session_closure(items, 1, plan)
        self.assertEqual([item["turn_index"] for item in closure], [1])

    def test_unbounded_order_keeps_all_candidates_but_places_primary_first(self):
        plan = build_query_plan(
            "I'm trying to recall what the designation on my jumpsuit was that helped me find the file number?"
        )
        items = [
            self._item(
                turn_index=0,
                role="user",
                speech_act="user_statement",
                pair_type="user_without_preceding_user",
                score=9000,
                content="Write a story about finding a file number.",
                question_turn_id=None,
            ),
            self._item(
                turn_index=1,
                role="assistant",
                speech_act="assistant_recommendation",
                score=4000,
                content="I think back to the file number.",
                answer_units=[{"kind": "sentence", "text": "file number"}],
            ),
            self._item(
                turn_index=5,
                role="assistant",
                speech_act="assistant_instruction",
                score=500,
                content='The designation on the jumpsuit was "LIV".',
                literal_hits=1,
                requested_entity_hits=1,
                answer_units=[{"kind": "quoted_span", "text": "LIV"}],
            ),
        ]
        ordered = _ordered_session_candidates(items, plan)
        self.assertEqual(len(ordered), len(items))
        self.assertEqual(ordered[0]["turn_index"], 5)
        self.assertEqual({item["turn_index"] for item in ordered}, {0, 1, 5})

    def test_primary_answer_closure_ignores_high_score_distractor(self):
        plan = build_query_plan(
            "Can you remind me which restaurant in Rome you recommended?"
        )
        distractor = self._item(
            record_id="record:distractor",
            turn_index=1,
            score=9999,
            content="La Pergola is a fine dining restaurant in Rome.",
            answer_units=[{"kind": "sentence", "text": "La Pergola"}],
        )
        answer = self._item(
            record_id="record:answer",
            turn_index=5,
            score=100,
            content="I recommended Roscioli, an authentic Italian restaurant in Rome.",
            requested_entity_hits=1,
            relation_hits=2,
            answer_overlap=6,
            answer_units=[{"kind": "attribute_sentence", "text": "Roscioli"}],
        )
        closure = _primary_answer_closure([distractor, answer], plan)
        self.assertEqual([item["record_id"] for item in closure], ["record:answer"])

    def test_relation_evidence_beats_context_entity_hit(self):
        plan = build_query_plan(
            "I'm planning to revisit Orlando. Can you remind me of the dessert shop with giant milkshakes we discussed?"
        )
        context_match = self._item(
            record_id="record:context",
            turn_index=13,
            speech_act="assistant_recommendation",
            score=280,
            question_entity_hits=1,
            question_overlap=3,
            requested_entity_hits=1,
            answer_units=[{"kind": "list_item", "text": "Other fun things to do in Orlando."}],
        )
        relation_match = self._item(
            record_id="record:relation",
            turn_index=7,
            speech_act="assistant_instruction",
            score=6881,
            question_overlap=2,
            relation_hits=2,
            answer_overlap=6,
            answer_units=[{"kind": "list_item", "text": "The Sugar Factory offers giant milkshakes."}],
        )
        closure = _focused_session_closure([context_match, relation_match], 1, plan)
        self.assertEqual([item["record_id"] for item in closure], ["record:relation"])


if __name__ == "__main__":
    unittest.main()
