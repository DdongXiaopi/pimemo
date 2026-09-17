import unittest

from pimem.retrieval.evidence_units import (
    _speech_act,
    _answer_excerpt,
    _answer_units,
    answer_candidates,
    _list_item_spans,
    _locate_excerpt,
)
from pimem.retrieval.compiler import _answer_evidence_text
from pimem.retrieval.episodes import Episode, Turn
from pimem.retrieval.query_plan import build_query_plan


class EvidenceUnitRuleTests(unittest.TestCase):
    def test_count_plan_extracts_the_counted_noun(self):
        plan = build_query_plan("How many mummies will the party face in the temple?")
        self.assertEqual(plan.primary_intent, "count")
        self.assertEqual(plan.requested_objects, ["mummies"])
        self.assertEqual([slot["entity"] for slot in plan.requested_slots], ["mummies"])

    def test_recommendation_plan_preserves_named_list_item_shape(self):
        plan = build_query_plan(
            "Can you remind me of the Mayo Clinic video you recommended?"
        )
        self.assertEqual(plan.answer_shape, "list_item")
        self.assertIn("mayo clinic video", plan.relation_terms)

    def test_role_phrase_becomes_typed_relation(self):
        plan = build_query_plan(
            "Who is the President's Chief Advisor for Science and Technology mentioned in the article?"
        )
        self.assertIn("president's chief advisor for science and technology", plan.relation_terms)

    def test_answer_evidence_text_keeps_following_answer_sentence(self):
        item = {
            "answer_units": [{"kind": "sentence", "text": "This trail has panoramic views."}],
            "answer_span": {"text": "This trail has panoramic views. The GR-90 is the recommended route."},
        }
        evidence_text = _answer_evidence_text(item)
        self.assertIn("GR-90", evidence_text)

    def test_numbered_list_does_not_treat_citation_as_item(self):
        self.assertEqual(_list_item_spans("According to Borges, 1941) text."), [])
        items = _list_item_spans("1. Foo - detail 2. Bar")
        self.assertEqual([item["explicit_index"] for item in items], [1, 2])
        self.assertEqual([item["value"] for item in items], ["Foo - detail", "Bar"])

    def test_table_coordinate_keeps_header_and_source_spans(self):
        plan = build_query_plan("What was Admon's rotation on Sunday?")
        text = "| Day | 8 am - 4 pm (Day Shift) |\n| --- | --- |\n| Sunday | Admon |"
        units = _answer_units(text, plan)
        coordinates = [unit for unit in units if unit["kind"] == "table_coordinate"]
        self.assertTrue(coordinates)
        coordinate = coordinates[0]
        self.assertIn("8 am - 4 pm", coordinate["text"])
        self.assertIn("Admon", coordinate["text"])
        self.assertEqual(len(coordinate["coordinate_source_spans"]), 2)

    def test_historical_user_answer_selects_example_sentence(self):
        plan = build_query_plan("I mentioned a show as an example. What was it?")
        self.assertEqual(plan.speech_act_target, "historical_answer")
        text = 'I wanted access to old shows. I will give an example, "Doc Martin" show went down.'
        self.assertIn("Doc Martin", _answer_excerpt(text, plan))

    def test_normalized_excerpt_location_maps_back_to_source(self):
        text = "Intro\n  hello   world."
        start, end = _locate_excerpt(text, "hello world.")
        self.assertEqual(text[start:end], "hello   world.")

    def test_location_question_keeps_location_sentence_over_list_item(self):
        plan = build_query_plan("Where did I redeem a $5 coupon on coffee creamer?")
        text = ("The Cartwheel app is a fantastic tool for saving money on household items and more at Target. "
                "Using a binder with labeled sections is a great way to organize your coupons.\n\n"
                "1. **Use clear plastic sleeves:** Add clear plastic sleeves.")
        units = _answer_units(text, plan)
        self.assertTrue(any(unit.get("kind") == "location_sentence" for unit in units))
        self.assertIn("Target", _answer_excerpt(text, plan, max_chars=None))

    def test_typed_historical_summary_beats_acknowledgement(self):
        plan = build_query_plan(
            "I'm looking back at our previous conversation about the Bajimaya v Reward Homes Pty Ltd case."
            " Can you remind me what year the construction of the house began?"
        )
        episode = Episode("episode:case", "case", {}, [], 0, 1)
        turns = [
            Turn("turn:summary", "summary", episode.episode_id, 0, "user",
                 "Bajimaya v Reward Homes Pty Ltd. The construction of the house began in 2014, "
                 "and the contract was signed in 2015.", "", ""),
            Turn("turn:ack", "ack", episode.episode_id, 1, "assistant",
                 "I understand the case summary of Bajimaya v Reward Homes Pty Ltd.", "", ""),
        ]
        candidates = answer_candidates(episode, turns, plan, max_chars=None)
        self.assertEqual(candidates[0]["turn_index"], 0)
        self.assertGreater(candidates[0]["typed_relation_score"], 0)
        self.assertIn("2014", candidates[0]["text"])

    def test_question_marks_inside_answer_content_do_not_drop_the_answer(self):
        episode = Episode("episode:question-mark", "question-mark", {}, [], 0, 1)
        turns = [
            Turn("turn:user", "user", episode.episode_id, 0, "user",
                 "Can you summarize the paper?", "", ""),
            Turn("turn:assistant", "assistant", episode.episode_id, 1, "assistant",
                 "The paper titled 'To Adapt or Not to Adapt?' reports a 20% improvement. "
                 "See https://example.test/report?v=2.", "", ""),
        ]
        plan = build_query_plan("What was the average improvement in framerate?")
        self.assertEqual(_speech_act(turns[1]), "assistant_answer")
        candidates = answer_candidates(episode, turns, plan, max_chars=None)
        self.assertTrue(any(item["turn_index"] == 1 for item in candidates))

    def test_prose_count_is_not_rejected_by_table_cell_shape(self):
        episode = Episode("episode:count", "count", {}, [], 0, 1)
        turns = [
            Turn("turn:user", "user", episode.episode_id, 0, "user",
                 "How many games were played at Arrowhead Stadium?", "", ""),
            Turn("turn:assistant", "assistant", episode.episode_id, 1, "assistant",
                 "The teams played 12 games at Arrowhead Stadium.", "", ""),
        ]
        plan = build_query_plan("How many times did the Chiefs play the Jaguars at Arrowhead Stadium?")
        candidates = answer_candidates(episode, turns, plan, max_chars=None)
        self.assertTrue(any(item["turn_index"] == 1 for item in candidates))

    def test_quoted_designation_is_an_attribute_sentence(self):
        plan = build_query_plan("What was the designation on my jumpsuit?")
        text = 'I look down at my jumpsuit and see the designation "LIV" and a square around it.'
        units = _answer_units(text, plan)
        attribute_units = [unit for unit in units if unit.get("kind") == "attribute_sentence"]
        self.assertTrue(attribute_units)
        self.assertEqual(attribute_units[0]["value"], '"LIV"')
        self.assertIn('designation "LIV"', _answer_excerpt(text, plan, max_chars=None))

    def test_after_chess_literal_returns_next_move(self):
        plan = build_query_plan("What was the move after 27. Kg2 Bd5+?")
        text = "The position was 27. Kg2 Bd5+ 28. Kg3 Be6."
        self.assertEqual(_answer_excerpt(text, plan, max_chars=None), "28. Kg3")


if __name__ == "__main__":
    unittest.main()
