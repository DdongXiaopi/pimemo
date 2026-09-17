import json
import tempfile
import unittest
from pathlib import Path

from pimem.eval.longmemeval import load_cases, run_longmemeval


class LongMemEvalAdapterTests(unittest.TestCase):
    def test_adapter_ingests_timestamps_and_reports_evidence_recall(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            fixture = [{
                "question_id": "q1", "question_type": "single-session-user",
                "question": "What project test command uses pytest?", "answer": "pytest -q",
                "question_date": "2024/01/03",
                "haystack_dates": ["2024/01/01 (Mon) 10:00", "2024/01/02 (Tue) 10:00"],
                "haystack_session_ids": ["s1", "s2"],
                "haystack_sessions": [[{"role": "user", "content": "The project uses pytest -q."}],
                                      [{"role": "assistant", "content": "A different unrelated conversation."}]],
                "answer_session_ids": ["s1"],
            }, {
                "question_id": "q2_abs", "question_type": "abstention",
                "question": "What is the secret launch code?", "answer": "I don't know",
                "question_date": "2024/01/03", "haystack_dates": ["2024/01/01"],
                "haystack_session_ids": ["s3"],
                "haystack_sessions": [[{"role": "user", "content": "We discussed test commands."}]],
                "answer_session_ids": [],
            }]
            input_path = root / "longmemeval.json"
            input_path.write_text(json.dumps(fixture), encoding="utf-8")
            cases = load_cases(str(input_path))
            report = run_longmemeval(str(input_path), artifact_root=str(root / "runs"))
            self.assertEqual(len(cases), 2)
            self.assertEqual(report["case_count"], 2)
            self.assertEqual(report["observed_turns"], 3)
            self.assertEqual(report["indexed_turns"], 3)
            self.assertEqual(report["active_claim_traceability"], 1.0)
            self.assertEqual(report["evidence_hit_rate"], 1.0)
            self.assertTrue(Path(report["database_path"]).exists())

    def test_loader_rejects_misaligned_haystack_lists(self):
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "invalid.json"
            path.write_text(json.dumps([{
                "question_id": "q1", "question_type": "single-session-user", "question": "q", "answer": "a",
                "question_date": "2024/01/01", "haystack_dates": [], "haystack_session_ids": ["s1"],
                "haystack_sessions": [], "answer_session_ids": [],
            }]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "equal lengths"):
                load_cases(str(path))

    def test_bulk_ingest_keeps_sensitive_turns_out_of_canonical_store(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            fixture = [{
                "question_id": "q_sensitive", "question_type": "single-session-user",
                "question": "What pytest command is used?", "answer": "pytest -q",
                "question_date": "2024/01/03", "haystack_dates": ["2024/01/01"],
                "haystack_session_ids": ["s1"],
                "haystack_sessions": [[
                    {"role": "user", "content": "The project uses pytest -q."},
                    {"role": "assistant", "content": "api_key=sk-12345678901234567890"},
                ]],
                "answer_session_ids": ["s1"],
            }]
            input_path = root / "longmemeval-sensitive.json"
            input_path.write_text(json.dumps(fixture), encoding="utf-8")
            report = run_longmemeval(str(input_path), artifact_root=str(root / "runs"))
            self.assertEqual(report["observed_turns"], 2)
            self.assertEqual(report["indexed_turns"], 1)
            self.assertEqual(report["active_claim_traceability"], 0.5)
            self.assertEqual(report["evidence_hit_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
