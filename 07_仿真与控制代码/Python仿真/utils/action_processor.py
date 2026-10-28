"""
动作处理工具
项目：基于深度强化学习的多无人机协同物流配送路径规划
"""

import numpy as np


class ActionProcessor:
    """动作处理器"""

    def __init__(self, config):
        self.a_max = config.get('a_max', 3.0)
        self.v_max = config.get('v_max', 15.0)
        self.dt = config.get('dt', 0.5)

    def process_action(self, raw_action):
        """
        处理原始动作

        Args:
            raw_action: (3,) 原始加速度

        Returns:
            action: (3,) 处理后的加速度
        """
        # 裁剪到有效范围
        action = np.clip(raw_action, -self.a_max, self.a_max)
        return action

    def check_velocity_constraint(self, velocity, action):
        """
        检查速度约束

        Args:
            velocity: 当前速度
            action: 加速度动作

        Returns:
            valid_action: 满足约束的动作
        """
        new_velocity = velocity + action * self.dt
        speed = np.linalg.norm(new_velocity)

        if speed > self.v_max:
            # 缩放动作以满足速度约束
            scale = (self.v_max - np.linalg.norm(velocity)) / (np.linalg.norm(action) * self.dt + 1e-8)
            action = action * max(scale, 0)

        return action

    def smooth_action(self, current_action, prev_action, alpha=0.3):
        """
        动作平滑

        Args:
            current_action: 当前动作
            prev_action: 上一步动作
            alpha: 平滑系数

        Returns:
            smoothed_action: 平滑后的动作
        """
        if prev_action is None:
            return current_action
        return alpha * current_action + (1 - alpha) * prev_action
