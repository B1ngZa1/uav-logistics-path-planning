"""
无人机动力学模块
项目：基于深度强化学习的多无人机协同物流配送路径规划
"""

import numpy as np


class UAVDynamics:
    """无人机动力学模型"""

    def __init__(self, config):
        self.v_max = config.get('v_max', 15.0)
        self.v_z_max = config.get('v_z_max', 5.0)
        self.a_max = config.get('a_max', 3.0)
        self.z_min = config.get('z_min', 10.0)
        self.z_max = config.get('z_max', 120.0)
        self.dt = config.get('dt', 0.5)

    def update(self, state, action):
        """
        更新无人机状态

        Args:
            state: [x, y, z, vx, vy, vz] 状态向量
            action: [ax, ay, az] 加速度动作

        Returns:
            new_state: 更新后的状态
        """
        pos = state[:3]
        vel = state[3:6]

        # 加速度裁剪
        action = np.clip(action, -self.a_max, self.a_max)

        # 速度更新
        new_vel = vel + action * self.dt

        # 速度约束
        speed_xy = np.linalg.norm(new_vel[:2])
        if speed_xy > self.v_max:
            new_vel[:2] = new_vel[:2] / speed_xy * self.v_max
        new_vel[2] = np.clip(new_vel[2], -self.v_z_max, self.v_z_max)

        # 位置更新
        new_pos = pos + vel * self.dt + 0.5 * action * self.dt**2

        # 高度约束
        new_pos[2] = np.clip(new_pos[2], self.z_min, self.z_max)

        return np.concatenate([new_pos, new_vel])

    def check_constraints(self, state):
        """检查状态约束"""
        vel = state[3:6]
        pos = state[:3]

        if np.linalg.norm(vel) > self.v_max + 1e-3:
            return False
        if pos[2] < self.z_min - 1e-3 or pos[2] > self.z_max + 1e-3:
            return False
        return True
