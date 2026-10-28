"""
奖励函数
项目：基于深度强化学习的多无人机协同物流配送路径规划
"""

import numpy as np


class RewardFunction:
    """多目标奖励函数（8项分量）"""

    def __init__(self, config):
        # 奖励权重
        self.w_goal = config.get('w_goal', 200.0)          # 目标到达奖励
        self.w_collision = config.get('w_collision', -200.0)  # 碰撞惩罚
        self.w_proximity = config.get('w_proximity', -100.0)  # 间距惩罚
        self.w_formation = config.get('w_formation', -1.0)    # 队形保持
        self.w_energy = config.get('w_energy', -0.5)          # 能耗惩罚
        self.w_time = config.get('w_time', -0.5)              # 时间惩罚
        self.w_smooth = config.get('w_smooth', -0.1)          # 平滑惩罚
        self.w_delivery = config.get('w_delivery', 50.0)      # 配送完成奖励

        # 安全距离
        self.d_safe = config.get('d_safe', 5.0)
        self.d_obs = config.get('d_obs', 3.0)

    def compute(self, uav_positions, uav_velocities, uav_actions,
                target_positions, delivered, obstacle_positions, obstacle_radii,
                prev_actions, energy_consumed):
        """
        计算总奖励

        Args:
            uav_positions: (N, 3) 无人机位置
            uav_velocities: (N, 3) 无人机速度
            uav_actions: (N, 3) 当前动作
            target_positions: (M, 3) 目标位置
            delivered: (M,) 配送状态
            obstacle_positions: (K, 3) 障碍物位置
            obstacle_radii: (K,) 障碍物半径
            prev_actions: (N, 3) 上一步动作
            energy_consumed: (N,) 能耗

        Returns:
            total_reward, reward_dict
        """
        num_uavs = len(uav_positions)
        reward_dict = {
            'goal': 0, 'collision': 0, 'proximity': 0,
            'formation': 0, 'energy': 0, 'time': 0,
            'smooth': 0, 'delivery': 0
        }

        # 1. 目标到达奖励 + 距离奖励
        for i in range(num_uavs):
            for j in range(len(target_positions)):
                if not delivered[j]:
                    dist = np.linalg.norm(uav_positions[i] - target_positions[j])
                    reward_dict['goal'] += -0.1 * dist  # 距离引导
                    if dist < 3.0:
                        reward_dict['goal'] += self.w_goal  # 到达奖励

        # 2. 碰撞惩罚
        for i in range(num_uavs):
            # 障碍物碰撞
            for k in range(len(obstacle_positions)):
                dist = np.linalg.norm(uav_positions[i] - obstacle_positions[k])
                if dist < obstacle_radii[k]:
                    reward_dict['collision'] += self.w_collision
                elif dist < self.d_obs:
                    reward_dict['collision'] += self.w_collision * 0.1

        # 3. 无人机间距惩罚
        for i in range(num_uavs):
            for j in range(i + 1, num_uavs):
                dist = np.linalg.norm(uav_positions[i] - uav_positions[j])
                if dist < self.d_safe:
                    reward_dict['proximity'] += self.w_proximity * (1 - dist / self.d_safe)

        # 4. 能耗惩罚
        reward_dict['energy'] += self.w_energy * np.sum(energy_consumed)

        # 5. 时间惩罚
        reward_dict['time'] += self.w_time * num_uavs

        # 6. 动作平滑惩罚
        if prev_actions is not None:
            smoothness = np.linalg.norm(uav_actions - prev_actions, axis=1)
            reward_dict['smooth'] += self.w_smooth * np.mean(smoothness)

        # 7. 配送完成奖励
        reward_dict['delivery'] += self.w_delivery * np.sum(delivered)

        total_reward = sum(reward_dict.values())
        return total_reward, reward_dict
