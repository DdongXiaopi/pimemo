import unittest
import importlib.util
from pathlib import Path

from pimem.eval.longmemeval import LongMemEvalCase


_DIAGNOSTIC_PATH = Path(__file__).parents[1] / "eval" / "diagnose_longmemeval_evidence.py"
_DIAGNOSTIC_SPEC = importlib.util.spec_from_file_location("diagnose_longmemeval_evidence", _DIAGNOSTIC_PATH)
_DIAGNOSTIC_MODULE = importlib.util.module_from_spec(_DIAGNOSTIC_SPEC)
assert _DIAGNOSTIC_SPEC.loader is not None
_DIAGNOSTIC_SPEC.loader.exec_module(_DIAGNOSTIC_MODULE)
_gold_answer_turns = _DIAGNOSTIC_MODULE._gold_answer_turns


class DiagnosticRuleTests(unittest.TestCase):
    def test_direct_answer_and_later_transcript_are_both_answer_turns(self):
        case = LongMemEvalCase(
            question_id="chess",
            question_type="single-session-assistant",
            question="What was the move after 27. Kg2 Bd5+?",
            answer="28. Kg3",
            question_date="2023/05/29",
            haystack_dates=["2023/05/28"],
            haystack_session_ids=["session:chess"],
            haystack_sessions=[[]],
            answer_session_ids=["session:chess"],
        )
        observations = [
            {
                "observation_id": "obs:direct",
                "session_id": "session:chess",
                "turn_index": 19,
                "role": "assistant",
                "content": "28. Kg3 would be my move.",
            },
            {
                "observation_id": "obs:transcript",
                "session_id": "session:chess",
                "turn_index": 21,
                "role": "assistant",
                "content": "The position is 27. Kg2 Bd5+ 28. Kg3 Be6.",
            },
        ]
        turns = _gold_answer_turns(case, observations, {})
        self.assertEqual({item["observation_id"] for item in turns}, {"obs:direct", "obs:transcript"})


if __name__ == "__main__":
    unittest.main()
