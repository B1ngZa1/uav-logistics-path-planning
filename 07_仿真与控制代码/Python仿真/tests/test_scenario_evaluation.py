"""Regression tests for the authoritative multi-drone evaluator."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from experiments.run_scenario import (
    build_agent_for_env,
    build_scenario_env,
    run_scenario,
)


class TestScenarioEvaluation(unittest.TestCase):
    def test_s1_three_episodes_loads_checkpoint_with_multidrone_shapes(self):
        """S1 must run three episodes with its flattened 33/9 contract."""

        env = build_scenario_env("S1", max_steps=2)
        self.assertEqual(env.observation_space.shape, (33,))
        self.assertEqual(env.action_space.shape, (3, 3))

        agent = build_agent_for_env(env, hidden_dim=32)
        self.assertEqual(agent.state_dim, 33)
        self.assertEqual(agent.action_dim, 9)

        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "s1_ppo.pt"
            agent.save(checkpoint)
            results = run_scenario(
                "S1",
                num_episodes=3,
                checkpoint=checkpoint,
                seed=123,
                deterministic=True,
                max_steps=2,
                verbose=False,
            )
            repeated_results = run_scenario(
                "S1",
                num_episodes=3,
                checkpoint=checkpoint,
                seed=123,
                deterministic=True,
                max_steps=2,
                verbose=False,
            )

        self.assertEqual(len(results), 3)
        self.assertEqual([item["episode"] for item in results], [0, 1, 2])
        self.assertEqual([item["seed"] for item in results], [123, 124, 125])
        self.assertEqual(results, repeated_results)


if __name__ == "__main__":
    unittest.main()
