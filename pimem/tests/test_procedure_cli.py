import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class ProcedureCliTests(unittest.TestCase):
    def test_propose_replay_and_shadow_commands(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            experience = {
                "experience_id": "exp-cli",
                "task_id": "T1",
                "goal": "find test command",
                "scope": {"repository": "repo-a"},
                "actions": [{"name": "memory_search"}],
                "outcome": {"status": "success"},
                "evidence_ids": ["obs-cli"],
            }
            payload = {
                "experiences": [experience],
                "procedure_id": "proc-cli",
                "trigger": "user asks for test command",
                "preconditions": ["repository is known"],
                "actions": [{"name": "memory_search"}],
                "success_criteria": ["command returned"],
                "failure_modes": ["memory unavailable"],
                "stop_conditions": ["no result"],
                "rollback": ["do not change files"],
                "scope": {"repository": "repo-a"},
            }
            input_path = root / "input.json"
            procedure_path = root / "procedure.json"
            experience_path = root / "experience.json"
            experiences_path = root / "experiences.json"
            input_path.write_text(json.dumps(payload), encoding="utf-8")
            experience_path.write_text(json.dumps(experience), encoding="utf-8")
            experiences_path.write_text(json.dumps({"experiences": [experience]}), encoding="utf-8")

            def command(*args: str):
                return subprocess.run(
                    [sys.executable, "-m", "pimem.cli", *args],
                    cwd=Path(__file__).resolve().parents[1],
                    env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
                    capture_output=True,
                    text=True,
                    check=True,
                )

            proposed = json.loads(command("procedure-propose", "--input", str(input_path)).stdout)
            procedure_path.write_text(json.dumps(proposed), encoding="utf-8")
            replay = json.loads(command("procedure-replay", "--procedure", str(procedure_path), "--experience", str(experience_path)).stdout)
            shadow = json.loads(command("procedure-shadow", "--procedure", str(procedure_path), "--experiences", str(experiences_path)).stdout)

            self.assertEqual(proposed["status"], "candidate")
            self.assertEqual(replay["status"], "pass")
            self.assertFalse(shadow["active"])


if __name__ == "__main__":
    unittest.main()
