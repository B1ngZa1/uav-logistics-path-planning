"""Short end-to-end smoke test for the authoritative Wuhan PPO entrypoint."""

import json
import tempfile
import unittest
from pathlib import Path

from run_wuhan_training import main


class TestWuhanTrainingSmoke(unittest.TestCase):
    def test_checkpoint_log_and_resume(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "wuhan_ppo.pt"
            log_file = root / "wuhan_ppo.jsonl"
            checkpoint_dir = root / "checkpoints"

            first = main(
                [
                    "--zone",
                    "B",
                    "--episodes",
                    "2",
                    "--eval-episodes",
                    "1",
                    "--max-steps",
                    "1",
                    "--seed",
                    "7",
                    "--checkpoint-interval",
                    "1",
                    "--output",
                    str(output),
                    "--checkpoint-dir",
                    str(checkpoint_dir),
                    "--log-file",
                    str(log_file),
                ]
            )

            self.assertEqual(first["completed_episodes"], 2)
            self.assertTrue(output.is_file())
            latest = checkpoint_dir / "latest.pt"
            self.assertTrue(latest.is_file())
            self.assertTrue(
                (checkpoint_dir / "checkpoint_ep000002.pt").is_file()
            )

            resumed = main(
                [
                    "--resume",
                    str(latest),
                    "--episodes",
                    "1",
                    "--eval-episodes",
                    "0",
                ]
            )
            self.assertEqual(resumed["run_id"], first["run_id"])
            self.assertEqual(resumed["completed_episodes"], 3)

            records = [
                json.loads(line)
                for line in log_file.read_text(encoding="utf-8").splitlines()
            ]
            self.assertGreaterEqual(len(records), 10)
            run_ids = {record["run_id"] for record in records}
            self.assertEqual(run_ids, {first["run_id"]})
            self.assertEqual(
                sum(record["event"] == "run_start" for record in records),
                2,
            )
            self.assertEqual(
                sum(record["event"] == "train_episode" for record in records),
                3,
            )
            self.assertEqual(
                sum(
                    record["event"] == "evaluation_summary"
                    for record in records
                ),
                2,
            )
            first_start = next(
                record for record in records if record["event"] == "run_start"
            )
            self.assertEqual(first_start["seed"], 7)
            self.assertEqual(first_start["config"]["zone"], "B")


if __name__ == "__main__":
    unittest.main()
