"""
三维空域环境
项目：基于深度强化学习的多无人机协同物流配送路径规划
"""

import numpy as np
import gym
from gym import spaces


class AirspaceEnv(gym.Env):
    """三维空域多无人机配送环境"""

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, config):
        super().__init__()

        # 空域参数
        self.space_size = config.get('space_size', (100, 100, 50))  # 空域尺寸
        self.num_uavs = config.get('num_uavs', 3)                    # 无人机数量
        self.num_targets = config.get('num_targets', 5)              # 配送目标数
        self.num_obstacles = config.get('num_obstacles', 10)         # 障碍物数量
        self.dt = config.get('dt', 0.5)                              # 时间步长
        self.max_steps = config.get('max_steps', 500)                # 最大步数

        # 无人机参数
        self.v_max = config.get('v_max', 15.0)
        self.a_max = config.get('a_max', 3.0)
        self.z_min = config.get('z_min', 10.0)
        self.z_max = config.get('z_max', 40.0)
        self.d_safe = config.get('d_safe', 5.0)

        # 动作空间：每架无人机3维加速度
        self.action_space = spaces.Box(
            low=-self.a_max, high=self.a_max,
            shape=(self.num_uavs, 3), dtype=np.float32
        )

        # 观测空间：每架无人机11维状态
        obs_dim = self.num_uavs * 11
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,), dtype=np.float32
        )

        # 初始化状态
        self.uav_positions = None
        self.uav_velocities = None
        self.target_positions = None
        self.obstacle_positions = None
        self.step_count = 0
        self.delivered = None

    def reset(self):
        """重置环境"""
        self.step_count = 0

        # 随机生成无人机起始位置
        self.uav_positions = np.random.uniform(
            low=[0, 0, self.z_min],
            high=[self.space_size[0], self.space_size[1], self.z_max],
            size=(self.num_uavs, 3)
        ).astype(np.float32)

        # 初始化速度为0
        self.uav_velocities = np.zeros((self.num_uavs, 3), dtype=np.float32)

        # 随机生成配送目标位置
        self.target_positions = np.random.uniform(
            low=[0, 0, self.z_min],
            high=[self.space_size[0], self.space_size[1], self.z_max],
            size=(self.num_targets, 3)
        ).astype(np.float32)

        # 随机生成障碍物
        self.obstacle_positions = np.random.uniform(
            low=[10, 10, self.z_min],
            high=[self.space_size[0]-10, self.space_size[1]-10, self.z_max],
            size=(self.num_obstacles, 3)
        ).astype(np.float32)
        self.obstacle_radii = np.random.uniform(3, 8, size=self.num_obstacles)

        # 配送状态
        self.delivered = np.zeros(self.num_targets, dtype=bool)

        # 重置进度记录
        self._prev_dist = {}

        return self._get_obs()

    def step(self, actions):
        """
        执行一步动作

        Args:
            actions: (num_uavs, 3) 加速度动作

        Returns:
            obs, reward, done, info
        """
        self.step_count += 1

        # 更新无人机状态
        for i in range(self.num_uavs):
            # 加速度裁剪
            action = np.clip(actions[i], -self.a_max, self.a_max)

            # 速度更新
            self.uav_velocities[i] += action * self.dt

            # 速度约束
            speed = np.linalg.norm(self.uav_velocities[i])
            if speed > self.v_max:
                self.uav_velocities[i] = self.uav_velocities[i] / speed * self.v_max

            # 位置更新
            self.uav_positions[i] += self.uav_velocities[i] * self.dt

            # 边界约束
            self.uav_positions[i, 0] = np.clip(self.uav_positions[i, 0], 0, self.space_size[0])
            self.uav_positions[i, 1] = np.clip(self.uav_positions[i, 1], 0, self.space_size[1])
            self.uav_positions[i, 2] = np.clip(self.uav_positions[i, 2], self.z_min, self.z_max)

        # 计算奖励
        reward = self._compute_reward()

        # 检查终止条件
        done = self._check_done()

        info = {
            'delivered': np.sum(self.delivered),
            'collisions': self._count_collisions(),
            'step': self.step_count
        }

        return self._get_obs(), reward, done, info

    def _get_obs(self):
        """获取观测

        每架无人机11维状态：
          位置(3) + 速度(3) + d_goal(1) + d_obs(1) + d_uav(1) + E_remain(1) + p_delivery(1)
        """
        obs = []
        for i in range(self.num_uavs):
            # --- d_goal: 到最近未完成目标的距离 ---
            d_goal = np.linalg.norm(self.space_size)  # 默认为最大距离
            for j in range(self.num_targets):
                if not self.delivered[j]:
                    dist = np.linalg.norm(self.uav_positions[i] - self.target_positions[j])
                    if dist < d_goal:
                        d_goal = dist
            # 如果所有目标已完成，d_goal保持为0
            if np.all(self.delivered):
                d_goal = 0.0

            # --- d_obs: 到最近障碍物表面的距离 ---
            d_obs = np.linalg.norm(self.space_size)  # 默认为最大距离
            if self.num_obstacles > 0:
                for j in range(self.num_obstacles):
                    dist = np.linalg.norm(self.uav_positions[i] - self.obstacle_positions[j])
                    surface_dist = dist - self.obstacle_radii[j]
                    if surface_dist < d_obs:
                        d_obs = surface_dist
            d_obs = max(d_obs, 0.0)  # 若在障碍物内部则截断为0

            # --- d_uav: 到最近其他无人机的距离 ---
            d_uav = np.inf
            for k in range(self.num_uavs):
                if k != i:
                    dist = np.linalg.norm(self.uav_positions[i] - self.uav_positions[k])
                    if dist < d_uav:
                        d_uav = dist
            if d_uav == np.inf:
                d_uav = np.linalg.norm(self.space_size)  # 仅一架无人机时设为最大距离

            # --- p_delivery: 配送进度 ---
            p_delivery = np.sum(self.delivered) / self.num_targets

            # 拼接11维状态
            state = np.concatenate([
                self.uav_positions[i],      # 位置 (3)
                self.uav_velocities[i],     # 速度 (3)
                [d_goal],                    # 目标距离 (1)
                [d_obs],                     # 障碍物距离 (1)
                [d_uav],                     # 最近无人机距离 (1)
                [0.8],                       # 剩余电量 E_remain placeholder (1)
                [p_delivery]                 # 配送进度 (1)
            ])
            obs.append(state)

        return np.concatenate(obs).astype(np.float32)

    def _compute_reward(self):
        """计算奖励（改进版）"""
        reward = 0.0

        for i in range(self.num_uavs):
            # 距离奖励（改进：使用进度奖励而非绝对距离惩罚）
            for j in range(self.num_targets):
                if not self.delivered[j]:
                    dist = np.linalg.norm(self.uav_positions[i] - self.target_positions[j])

                    # 进度奖励（正向引导）
                    if not hasattr(self, '_prev_dist'):
                        self._prev_dist = {}
                    prev = self._prev_dist.get((i, j), dist)
                    progress = (prev - dist) / max(prev, 1e-6)
                    reward += 10.0 * progress  # 进度奖励
                    self._prev_dist[(i, j)] = dist

                    # 接近奖励（距离越近奖励越大）
                    if dist < 50.0:
                        reward += 50.0 * (1.0 - dist / 50.0)

                    # 到达目标奖励（增加阈值到10米）
                    if dist < 10.0:
                        self.delivered[j] = True
                        reward += 500.0  # 增加到达奖励

            # 碰撞惩罚（降低幅度）
            for j in range(self.num_obstacles):
                dist = np.linalg.norm(self.uav_positions[i] - self.obstacle_positions[j])
                if dist < self.obstacle_radii[j]:
                    reward -= 50.0  # 从200降低到50
                elif dist < self.d_safe:
                    reward -= 5.0  # 从10降低到5

            # 无人机间碰撞惩罚
            for k in range(i + 1, self.num_uavs):
                dist = np.linalg.norm(self.uav_positions[i] - self.uav_positions[k])
                if dist < self.d_safe:
                    reward -= 20.0  # 从100降低到20

            # 时间惩罚（降低幅度）
            reward -= 0.1  # 从0.5降低到0.1

        # 全部完成奖励
        if np.all(self.delivered):
            reward += 1000.0  # 从500增加到1000

        return reward

    def _check_done(self):
        """检查是否结束"""
        # 全部配送完成
        if np.all(self.delivered):
            return True

        # 超过最大步数
        if self.step_count >= self.max_steps:
            return True

        return False

    def _count_collisions(self):
        """统计碰撞次数"""
        collisions = 0
        for i in range(self.num_uavs):
            for j in range(self.num_obstacles):
                dist = np.linalg.norm(self.uav_positions[i] - self.obstacle_positions[j])
                if dist < self.obstacle_radii[j]:
                    collisions += 1
            for k in range(i + 1, self.num_uavs):
                dist = np.linalg.norm(self.uav_positions[i] - self.uav_positions[k])
                if dist < self.d_safe:
                    collisions += 1
        return collisions

    def render(self, mode='human'):
        """渲染环境"""
        pass

    def seed(self, seed=None):
        """设置随机种子"""
        np.random.seed(seed)
        return [seed]
