import unittest

from pimem.retrieval.evidence_units import (
    _answer_excerpt,
    _answer_units,
    answer_candidates,
    _list_item_spans,
    _locate_excerpt,
)
from pimem.retrieval.episodes import Episode, Turn
from pimem.retrieval.query_plan import build_query_plan


class EvidenceUnitRuleTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
