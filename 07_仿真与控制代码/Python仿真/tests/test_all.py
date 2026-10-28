"""
单元测试
项目：基于深度强化学习的多无人机协同物流配送路径规划
"""

import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.airspace_env import AirspaceEnv
from environment.uav_dynamics import UAVDynamics
from environment.energy_model import EnergyModel
from environment.collision_detector import CollisionDetector
from algorithm.ppo_agent import PPOAgent
from utils.state_encoder import StateEncoder
from utils.action_processor import ActionProcessor


class TestAirspaceEnv(unittest.TestCase):
    """测试三维空域环境"""

    def setUp(self):
        self.config = {
            'space_size': (100, 100, 50),
            'num_uavs': 3,
            'num_targets': 5,
            'num_obstacles': 5,
            'dt': 0.5,
            'max_steps': 100
        }
        self.env = AirspaceEnv(self.config)

    def test_reset(self):
        """测试环境重置"""
        obs = self.env.reset()
        self.assertEqual(len(obs), self.config['num_uavs'] * 11)
        self.assertEqual(self.env.uav_positions.shape, (3, 3))
        self.assertEqual(self.env.target_positions.shape, (5, 3))

    def test_step(self):
        """测试环境步进"""
        self.env.reset()
        actions = np.zeros((3, 3))
        obs, reward, done, info = self.env.step(actions)
        self.assertEqual(len(obs), 33)
        self.assertTrue(isinstance(reward, (float, np.floating)))
        self.assertIsInstance(done, bool)

    def test_boundary(self):
        """测试边界约束"""
        self.env.reset()
        # 大动作应被裁剪到边界内
        actions = np.ones((3, 3)) * 100
        for _ in range(10):
            obs, _, _, _ = self.env.step(actions)
        for i in range(3):
            self.assertGreaterEqual(self.env.uav_positions[i, 2], self.config.get('z_min', 10))


class TestUAVDynamics(unittest.TestCase):
    """测试无人机动力学"""

    def setUp(self):
        self.dynamics = UAVDynamics({'v_max': 15.0, 'a_max': 3.0, 'dt': 0.5})

    def test_update(self):
        """测试状态更新"""
        state = np.array([0, 0, 10, 0, 0, 0], dtype=float)
        action = np.array([1, 0, 0], dtype=float)
        new_state = self.dynamics.update(state, action)
        self.assertAlmostEqual(new_state[3], 0.5, places=5)  # vx = 0 + 1*0.5

    def test_velocity_constraint(self):
        """测试速度约束"""
        state = np.array([0, 0, 10, 14, 0, 0], dtype=float)
        action = np.array([10, 0, 0], dtype=float)
        new_state = self.dynamics.update(state, action)
        speed = np.linalg.norm(new_state[3:6])
        self.assertLessEqual(speed, 15.0 + 1e-3)


class TestEnergyModel(unittest.TestCase):
    """测试能耗模型"""

    def setUp(self):
        self.energy = EnergyModel({
            'k1': 0.8, 'k2': 1.2, 'k3': 0.08,
            'E_capacity': 500.0, 'eta_motor': 0.8
        })

    def test_flight_power(self):
        """测试飞行功率计算"""
        velocity = np.array([10, 0, 0], dtype=float)
        power = self.energy.compute_flight_power(velocity, 15.0)
        self.assertGreater(power, 0)

    def test_hover_power(self):
        """测试悬停功率"""
        power = self.energy.compute_hover_power(15.0)
        self.assertGreater(power, 0)

    def test_soc_update(self):
        """测试SOC更新"""
        soc = self.energy.update_soc(1.0, 1000.0, 0.5)
        self.assertLess(soc, 1.0)
        self.assertGreater(soc, 0)


class TestCollisionDetector(unittest.TestCase):
    """测试碰撞检测"""

    def setUp(self):
        self.detector = CollisionDetector({'d_safe': 5.0, 'd_obs': 3.0})

    def test_uav_collision(self):
        """测试无人机间碰撞检测"""
        pos_i = np.array([0, 0, 10])
        pos_j = np.array([3, 0, 10])
        collision, dist = self.detector.check_uav_collision(pos_i, pos_j)
        self.assertTrue(collision)
        self.assertAlmostEqual(dist, 3.0)

    def test_no_collision(self):
        """测试无碰撞情况"""
        pos_i = np.array([0, 0, 10])
        pos_j = np.array([10, 0, 10])
        collision, dist = self.detector.check_uav_collision(pos_i, pos_j)
        self.assertFalse(collision)


class TestPPOAgent(unittest.TestCase):
    """测试PPO智能体"""

    def setUp(self):
        self.agent = PPOAgent({
            'state_dim': 11,
            'action_dim': 3,
            'hidden_dim': 64
        })

    def test_select_action(self):
        """测试动作选择"""
        state = np.random.randn(11).astype(np.float32)
        action, log_prob, value = self.agent.select_action(state)
        self.assertEqual(action.shape, (3,))
        self.assertIsInstance(log_prob, float)

    def test_store_transition(self):
        """测试经验存储"""
        state = np.random.randn(11).astype(np.float32)
        action = np.random.randn(3)
        self.agent.store_transition(state, action, 1.0, False, 0.0, 0.0)
        self.assertEqual(len(self.agent.states), 1)


if __name__ == '__main__':
    unittest.main()
