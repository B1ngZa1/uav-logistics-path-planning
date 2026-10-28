"""
CTDE框架（Centralized Training with Decentralized Execution）
项目：基于深度强化学习的多无人机协同物流配送路径规划
"""

import numpy as np
import torch
import torch.nn as nn

from utils.state_encoder import StateEncoder


class CentralizedCritic(nn.Module):
    """集中式Critic网络（训练时使用全局状态）"""

    def __init__(self, global_state_dim, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(global_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, global_state):
        return self.network(global_state)


class CTDEFramework:
    """CTDE多智能体框架"""

    def __init__(self, config):
        self.num_uavs = config.get('num_uavs', 3)
        self.local_state_dim = config.get('local_state_dim', 11)
        self.action_dim = config.get('action_dim', 3)

        # 全局状态 = 所有智能体局部状态拼接
        self.global_state_dim = self.num_uavs * self.local_state_dim

        # 状态编码器（用于归一化）
        self.state_encoder = StateEncoder(config)

        # 集中式Critic
        self.centralized_critic = CentralizedCritic(self.global_state_dim)

        # 每个智能体独立的Actor（共享权重）
        self.shared_actor = None  # 由PPOAgent提供

    def get_global_state(self, local_states):
        """
        构建全局状态

        Args:
            local_states: list of local observations for each UAV

        Returns:
            global_state: concatenated global state
        """
        return np.concatenate(local_states)

    def get_local_obs(self, env_state, uav_id):
        """
        获取单个智能体的局部观测

        Args:
            env_state: 环境全局状态
            uav_id: 智能体ID

        Returns:
            local_obs: 局部观测
        """
        # 提取本机状态
        uav_pos = env_state['uav_positions'][uav_id]
        uav_vel = env_state['uav_velocities'][uav_id]
        soc = env_state['soc'][uav_id]

        # 提取目标信息
        nearest_target = env_state['nearest_targets'][uav_id]
        nearest_dist = np.linalg.norm(uav_pos - nearest_target)

        # 拼接
        local_obs = np.concatenate([
            uav_pos, uav_vel, [soc],
            nearest_target, [nearest_dist]
        ])

        return local_obs.astype(np.float32)

    def _get_local_observations(self, env_state, uav_id):
        """
        获取单个无人机的11维局部观测向量（带归一化）

        状态空间定义：
            pos(3) + vel(3) + d_goal(1) + d_obs(1) + d_uav(1) + E_remain(1) + p_delivery(1)

        Args:
            env_state: 环境状态字典，包含以下字段：
                - uav_positions: (num_uavs, 3) 无人机位置数组
                - uav_velocities: (num_uavs, 3) 无人机速度数组
                - soc: (num_uavs,) 剩余电量数组 (0~1)
                - target_positions: (num_targets, 3) 配送目标位置
                - delivered: (num_targets,) 配送完成状态 (bool)
                - obstacle_positions: (num_obstacles, 3) 障碍物位置
                - obstacle_radii: (num_obstacles,) 障碍物半径
            uav_id: 当前无人机的索引

        Returns:
            local_obs: 11维归一化局部观测向量 (np.float32)
        """
        uav_pos = env_state['uav_positions'][uav_id]
        uav_vel = env_state['uav_velocities'][uav_id]
        soc = env_state['soc'][uav_id]
        target_positions = env_state['target_positions']
        delivered = env_state['delivered']
        obstacle_positions = env_state['obstacle_positions']
        obstacle_radii = env_state['obstacle_radii']
        num_uavs = env_state['uav_positions'].shape[0]

        # --- d_goal: 到最近未完成目标的距离 ---
        d_goal = np.inf
        for j in range(len(target_positions)):
            if not delivered[j]:
                dist = np.linalg.norm(uav_pos - target_positions[j])
                if dist < d_goal:
                    d_goal = dist
        if d_goal == np.inf:
            d_goal = 0.0  # 所有目标已完成

        # --- d_obs: 到最近障碍物表面的距离 ---
        d_obs = np.inf
        for j in range(len(obstacle_positions)):
            dist = np.linalg.norm(uav_pos - obstacle_positions[j])
            surface_dist = dist - obstacle_radii[j]
            if surface_dist < d_obs:
                d_obs = surface_dist
        d_obs = max(d_obs, 0.0)  # 若在障碍物内部则截断为0

        # --- d_uav: 到最近其他无人机的距离 ---
        d_uav = np.inf
        for k in range(num_uavs):
            if k != uav_id:
                dist = np.linalg.norm(uav_pos - env_state['uav_positions'][k])
                if dist < d_uav:
                    d_uav = dist
        if d_uav == np.inf:
            d_uav = np.linalg.norm(np.array([100, 100, 50]))  # 仅一架无人机时设为最大距离

        # --- p_delivery: 配送进度 ---
        p_delivery = np.sum(delivered) / len(delivered)

        # 使用StateEncoder归一化并编码为11维状态向量
        local_obs = self.state_encoder.encode_state(
            uav_pos=uav_pos,
            uav_vel=uav_vel,
            soc=soc,
            d_goal=d_goal,
            d_obs=d_obs,
            d_uav=d_uav,
            p_delivery=p_delivery
        )

        return local_obs

    def _encode_global_state(self, local_states):
        """
        编码全局状态：将所有无人机的局部观测拼接为一个全局状态向量

        Args:
            local_states: list of np.ndarray, 每个元素为一架无人机的11维局部观测

        Returns:
            global_state: (num_uavs * 11,) 全局状态向量 (np.float32)
        """
        global_state = np.concatenate(local_states).astype(np.float32)
        return global_state

    def compute_shared_reward(self, individual_rewards):
        """
        计算共享奖励

        Args:
            individual_rewards: 各智能体的个体奖励

        Returns:
            shared_reward: 共享奖励
        """
        # 全局奖励 = 所有个体奖励之和
        return np.sum(individual_rewards)

    def communicate(self, uav_states, comm_range=500.0):
        """
        智能体间通信（可选）

        Args:
            uav_states: 各智能体状态
            comm_range: 通信范围

        Returns:
            messages: 接收到的消息
        """
        messages = []
        for i in range(self.num_uavs):
            msg = []
            for j in range(self.num_uavs):
                if i != j:
                    dist = np.linalg.norm(
                        uav_states[i]['position'] - uav_states[j]['position']
                    )
                    if dist <= comm_range:
                        msg.append({
                            'sender': j,
                            'position': uav_states[j]['position'],
                            'velocity': uav_states[j]['velocity'],
                            'soc': uav_states[j]['soc']
                        })
            messages.append(msg)
        return messages
