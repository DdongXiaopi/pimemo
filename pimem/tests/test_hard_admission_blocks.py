"""Admission breadth for the hard budget.

``_select_hard_answer_closure`` used to stop after the first session bundle
whenever the requested slots were covered. Most LongMemEval questions expose a
single requested slot, so the pack admitted exactly one session and missed the
gold session whenever it was not ranked first (the ``session_seed`` failures).
``max_blocks`` makes the admission breadth explicit; the default of 1 keeps the
historical behaviour for every existing caller.
"""

import unittest

from pimem.retrieval.compiler import _select_hard_answer_closure
from pimem.retrieval.query_plan import build_query_plan


PLAN = build_query_plan("What are the working hours on Sunday for Admon?")


def _bundles(*session_ids):
    return [
        {
            "episode_id": "episode:%s" % session_id,
            "session_id": session_id,
            "primary_answer_candidate_ids": ["candidate:%s" % session_id],
            "coverage": {"primary_answer": True},
        }
        for session_id in session_ids
    ]


def _candidate(session_id, *, structured=True, score=100):
    return {
        "record_type": "answer_candidate",
        "record_id": "candidate:%s" % session_id,
        "session_id": session_id,
        "answer_shape": "table_cell",
        "score": score,
        "text": "| Sunday | %s | 8 am - 4 pm (Day Shift) |" % session_id,
        "answer_units": ([{"kind": "table_coordinate", "value": "8 am - 4 pm (Day Shift)"}]
                         if structured else [{"kind": "quoted_span", "value": session_id}]),
    }


class HardAdmissionBreadthTests(unittest.TestCase):
    def test_default_admits_a_single_session(self):
        target = _candidate("target")
        distractor = _candidate("distractor", structured=False, score=900)
        selected = _select_hard_answer_closure([distractor, target], PLAN,
                                               _bundles("target", "distractor"), token_budget=4096)
        self.assertEqual(len(selected), 1)
        self.assertEqual(len({item["session_id"] for item in selected}), 1)

    def test_max_blocks_admits_additional_sessions(self):
        target = _candidate("target")
        distractor = _candidate("distractor", structured=False, score=900)
        selected = _select_hard_answer_closure([distractor, target], PLAN,
                                               _bundles("target", "distractor"), token_budget=4096,
                                               max_blocks=2)
        self.assertEqual({item["session_id"] for item in selected}, {"target", "distractor"})

    def test_admission_is_ordered_by_evidence_quality(self):
        """Raising the floor must not reorder the sessions that were already admitted."""
        items = [_candidate("a"), _candidate("b", structured=False), _candidate("c", structured=False)]
        bundles = _bundles("a", "b", "c")
        narrow = _select_hard_answer_closure(items, PLAN, bundles, token_budget=4096, max_blocks=1)
        wide = _select_hard_answer_closure(items, PLAN, bundles, token_budget=4096, max_blocks=2)
        self.assertEqual(wide[0]["record_id"], narrow[0]["record_id"])

    def test_max_blocks_is_capped_by_available_bundles(self):
        items = [_candidate("a"), _candidate("b", structured=False), _candidate("c", structured=False)]
        selected = _select_hard_answer_closure(items, PLAN, _bundles("a", "b", "c"),
                                               token_budget=4096, max_blocks=10)
        self.assertEqual(len(selected), 3)
        self.assertEqual(len({item["session_id"] for item in selected}), 3)

    def test_max_blocks_grows_the_payload(self):
        """Breadth is only free before the packer; assert the trade-off is visible."""
        items = [_candidate("a"), _candidate("b", structured=False), _candidate("c", structured=False)]
        bundles = _bundles("a", "b", "c")
        narrow = _select_hard_answer_closure(items, PLAN, bundles, token_budget=4096, max_blocks=1)
        wide = _select_hard_answer_closure(items, PLAN, bundles, token_budget=4096, max_blocks=3)
        self.assertGreater(len(wide), len(narrow))


if __name__ == "__main__":
    unittest.main()
