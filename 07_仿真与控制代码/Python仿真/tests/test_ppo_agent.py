"""PPO 动作概率、更新轮数和检查点的回归测试。"""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from algorithm.ppo_agent import PPOAgent


def make_agent(**overrides):
    config = {
        "state_dim": 6,
        "action_dim": 4,
        "hidden_dim": 32,
        "lr": 1e-3,
        "update_epochs": 2,
        "batch_size": 2,
        "action_limit": 1.0,
    }
    config.update(overrides)
    return PPOAgent(config)


class TestPPOAgent(unittest.TestCase):
    def setUp(self):
        np.random.seed(3)
        torch.manual_seed(3)

    def test_tanh_action_is_bounded_and_finite(self):
        agent = make_agent()
        action, log_prob, value = agent.select_action(
            np.zeros(6, dtype=np.float32)
        )
        self.assertEqual(action.shape, (4,))
        self.assertTrue(np.isfinite(action).all())
        self.assertTrue(np.all(np.abs(action) <= 1.0))
        self.assertTrue(np.isfinite(log_prob))
        self.assertTrue(np.isfinite(value))

    def test_update_uses_update_epochs_and_clears_buffer(self):
        agent = make_agent()
        for index in range(4):
            state = np.full(6, index / 10, dtype=np.float32)
            action, log_prob, value = agent.select_action(state)
            agent.store_transition(
                state,
                action,
                reward=0.1,
                done=index == 3,
                log_prob=log_prob,
                value=value,
            )
        policy_loss, value_loss = agent.update(0.0)
        self.assertTrue(np.isfinite(policy_loss))
        self.assertTrue(np.isfinite(value_loss))
        self.assertEqual(agent.last_update_metrics["updates"], 4)
        self.assertEqual(agent.states, [])

    def test_invalid_action_limit_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            make_agent(action_limit=np.ones(3, dtype=np.float32))

    def test_checkpoint_round_trip(self):
        agent = make_agent()
        state = np.linspace(-1, 1, 6, dtype=np.float32)
        expected = agent.select_action(
            state, deterministic=True
        )[0]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "ppo.pt"
            agent.save(path)
            restored = make_agent()
            restored.load(path, load_optimizers=True)
            actual = restored.select_action(
                state, deterministic=True
            )[0]
        np.testing.assert_allclose(expected, actual, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
