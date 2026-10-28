"""武汉参考场景环境的接口、确定性和安全语义测试。"""

import unittest

import numpy as np

from algorithm.delivery_strategy import ScoreFirstDeliveryPlanner
from environment.wuhan_drone_env import WuhanDroneDeliveryEnv
from environment.wuhan_scene import ScenePackage


FAST_CONFIG = {
    "zone_id": "B",
    "num_drones": 1,
    "map_size": 8,
    "lidar_rays": 8,
    "lidar_range_m": 40.0,
    "max_episode_time_s": 5.0,
}


class TestScenePackage(unittest.TestCase):
    def test_load_all_zones(self):
        expected_task_ids = {
            "A": [1, 2, 3],
            "B": [4, 5, 6],
            "C": [7, 8, 9],
            "D": [10, 11, 12],
        }
        for zone_id, task_ids in expected_task_ids.items():
            scene = ScenePackage.load(zone_id)
            self.assertEqual(
                scene.coordinate_system, "local_ENU_meters_z_up"
            )
            self.assertEqual(
                [task.task_id for task in scene.tasks], task_ids
            )
            self.assertGreater(len(scene.static_prisms), 0)
            self.assertEqual(scene.extent_m.shape, (2,))
            self.assertEqual(len(scene.manifest_hash), 64)

    def test_known_building_collision(self):
        scene = ScenePackage.load("B")
        building = next(
            item
            for item in scene.static_prisms
            if item.kind == "building"
        )
        polygon = building.footprints[0]
        point = np.array(
            [
                float(polygon[:, 0].mean()),
                float(polygon[:, 1].mean()),
                (building.z_min_m + building.z_max_m) / 2,
            ]
        )
        self.assertIsNotNone(scene.static_collision(point, 1.0))


class TestWuhanDroneDeliveryEnv(unittest.TestCase):
    def test_reset_shapes_and_space(self):
        env = WuhanDroneDeliveryEnv(FAST_CONFIG)
        observation, info = env.reset(seed=7)
        self.assertEqual(observation["flight"].shape, (18,))
        self.assertEqual(
            observation["semantic_map"].shape, (9, 8, 8)
        )
        self.assertEqual(observation["lidar"].shape, (3, 8))
        self.assertEqual(
            observation["delivery_mask"].shape, (12,)
        )
        self.assertTrue(
            env.observation_space.contains(observation)
        )
        self.assertEqual(info["zone_id"], "B")
        self.assertEqual(info["local_task_ids"], [4, 5, 6])

    def test_step_five_tuple_and_reward_terms(self):
        env = WuhanDroneDeliveryEnv(FAST_CONFIG)
        env.reset(seed=11)
        result = env.step(np.zeros(16, dtype=np.float32))
        self.assertEqual(len(result), 5)
        observation, reward, terminated, truncated, info = result
        self.assertTrue(np.isfinite(reward))
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIn("progress", info["reward_terms"])
        self.assertTrue(
            env.observation_space.contains(observation)
        )

    def test_seed_is_deterministic(self):
        env1 = WuhanDroneDeliveryEnv(FAST_CONFIG)
        env2 = WuhanDroneDeliveryEnv(FAST_CONFIG)
        obs1, _ = env1.reset(seed=123)
        obs2, _ = env2.reset(seed=123)
        action = np.linspace(
            -0.5, 0.5, 16, dtype=np.float32
        )
        next1 = env1.step(action)[0]
        next2 = env2.step(action)[0]
        np.testing.assert_allclose(
            obs1["flight"], obs2["flight"]
        )
        np.testing.assert_allclose(
            next1["gps"], next2["gps"]
        )

    def test_invalid_action_rejected(self):
        env = WuhanDroneDeliveryEnv(FAST_CONFIG)
        env.reset(seed=1)
        with self.assertRaises(ValueError):
            env.step(np.zeros(15, dtype=np.float32))
        bad = np.zeros(16, dtype=np.float32)
        bad[0] = np.nan
        with self.assertRaises(ValueError):
            env.step(bad)

    def test_flatten_observation_is_stable(self):
        env = WuhanDroneDeliveryEnv(FAST_CONFIG)
        observation, _ = env.reset(seed=2)
        flat = env.flatten_observation(observation)
        expected = 18 + 6 + 12 + 4 + 3 * 8 + 8 + 6 + 4
        self.assertEqual(flat.shape, (expected,))
        self.assertTrue(np.isfinite(flat).all())

    def test_multidrone_shapes(self):
        config = dict(FAST_CONFIG)
        config["num_drones"] = 3
        env = WuhanDroneDeliveryEnv(config)
        observation, _ = env.reset(seed=9)
        self.assertEqual(
            observation["flight"].shape, (3, 18)
        )
        self.assertEqual(env.action_space.shape, (3, 16))
        result = env.step(
            np.zeros((3, 16), dtype=np.float32)
        )
        self.assertEqual(len(result), 5)

    def test_delivery_targets_continue_before_return(self):
        config = dict(FAST_CONFIG)
        config["max_episode_time_s"] = 900.0
        env = WuhanDroneDeliveryEnv(config)
        _, info = env.reset(seed=5)
        depot_id = env.scene.depot().task_id
        delivery_ids = [
            task.task_id
            for task in env.scene.tasks
            if task.task_id != depot_id
        ]
        valid = env._valid_target_mask()
        self.assertFalse(valid[depot_id - 1])
        env.served_mask[delivery_ids[0] - 1] = 1
        env._update_mission_phase()
        valid = env._valid_target_mask()
        self.assertFalse(valid[depot_id - 1])
        self.assertTrue(valid[delivery_ids[1] - 1])
        self.assertEqual(info["mission_phase"], "delivery")

    def test_all_delivered_switches_to_return(self):
        env = WuhanDroneDeliveryEnv(FAST_CONFIG)
        env.reset(seed=6)
        depot_id = env.scene.depot().task_id
        for task in env.scene.tasks:
            if task.task_id != depot_id:
                env.served_mask[task.task_id - 1] = 1
        env._update_mission_phase()
        valid = env._valid_target_mask()
        self.assertEqual(env._mission_phase, "return")
        self.assertTrue(valid[depot_id - 1])
        self.assertEqual(valid.sum(), 1)

    def test_energy_reserve_forces_partial_return(self):
        config = dict(FAST_CONFIG)
        config["max_episode_time_s"] = 900.0
        env = WuhanDroneDeliveryEnv(config)
        env.reset(seed=8)
        env.drone_states[0].battery_wh = 1.0
        env._update_mission_phase()
        depot_id = env.scene.depot().task_id
        valid = env._valid_target_mask()
        self.assertEqual(env._mission_phase, "return")
        self.assertEqual(env._return_trigger, "energy_reserve")
        self.assertTrue(valid[depot_id - 1])
        self.assertEqual(valid.sum(), 1)

    def test_safe_return_is_required_for_mission_score(self):
        config = dict(FAST_CONFIG)
        config["max_episode_time_s"] = 900.0
        env = WuhanDroneDeliveryEnv(config)
        env.reset(seed=12)
        depot_id = env.scene.depot().task_id
        delivery_id = next(
            task.task_id
            for task in env.scene.tasks
            if task.task_id != depot_id
        )
        env.served_mask[delivery_id - 1] = 1
        env._enter_return_phase("energy_reserve")
        env.drone_states[0].position_m = env._service_position(
            env.scene.depot()
        )
        terms = {
            "return": 0.0,
            "all_delivered": 0.0,
        }
        self.assertTrue(env._task_terminal(terms))
        info = env._get_info()
        self.assertEqual(
            info["termination_reason"], "safe_return_partial"
        )
        self.assertGreater(info["mission_score"], 250.0)

    def test_planner_maximizes_feasible_target_count(self):
        config = dict(FAST_CONFIG)
        config["max_episode_time_s"] = 900.0
        env = WuhanDroneDeliveryEnv(config)
        env.reset(seed=13)
        planner = ScoreFirstDeliveryPlanner()
        plan = planner.plan(env)
        self.assertTrue(plan.feasible)
        self.assertGreaterEqual(len(plan.target_ids), 1)
        action = planner.compose_action(
            env, np.zeros(4, dtype=np.float32)
        )
        self.assertEqual(action.shape, (16,))
        self.assertEqual(
            int(np.argmax(action[4:])) + 1,
            plan.target_ids[0],
        )


if __name__ == "__main__":
    unittest.main()
