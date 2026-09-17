import unittest

from pimem.retrieval.episodic_retriever import _focused_session_closure
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


if __name__ == "__main__":
    unittest.main()
