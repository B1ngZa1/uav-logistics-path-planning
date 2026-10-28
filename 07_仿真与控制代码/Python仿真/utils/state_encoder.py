"""
状态编码工具
项目：基于深度强化学习的多无人机协同物流配送路径规划
"""

import numpy as np


class StateEncoder:
    """状态编码器"""

    def __init__(self, config):
        self.space_size = config.get('space_size', (100, 100, 50))
        self.v_max = config.get('v_max', 15.0)
        self.z_min = config.get('z_min', 10.0)
        self.z_max = config.get('z_max', 40.0)
        # 空域对角线距离，用于距离特征归一化
        self.max_dist = np.linalg.norm([
            self.space_size[0], self.space_size[1], self.z_max - self.z_min
        ])

    def normalize_position(self, position):
        """归一化位置到[-1, 1]"""
        normalized = np.array([
            position[0] / self.space_size[0] * 2 - 1,
            position[1] / self.space_size[1] * 2 - 1,
            (position[2] - self.z_min) / (self.z_max - self.z_min) * 2 - 1
        ])
        return normalized

    def normalize_velocity(self, velocity):
        """归一化速度到[-1, 1]"""
        return velocity / self.v_max

    def normalize_soc(self, soc):
        """归一化电量到[-1, 1]"""
        return soc * 2 - 1

    def normalize_distance(self, distance):
        """归一化距离到[-1, 1]，距离越小值越小"""
        # 处理无穷大值
        if np.isinf(distance):
            return 1.0  # 无穷大距离归一化为1
        return np.clip(distance / self.max_dist * 2 - 1, -1.0, 1.0)

    def normalize_delivery_progress(self, progress):
        """归一化配送进度到[-1, 1]，progress范围[0, 1]"""
        return np.clip(progress * 2 - 1, -1.0, 1.0)

    def encode_state(self, uav_pos, uav_vel, soc, d_goal, d_obs, d_uav, p_delivery):
        """
        编码完整11维状态

        维度说明：
        - 位置(x,y,z):      3维，归一化到[-1,1]
        - 速度(vx,vy,vz):    3维，归一化到[-1,1]
        - d_goal:            1维，到最近目标的距离
        - d_obs:             1维，到最近障碍物的距离
        - d_uav:             1维，到最近其他无人机的距离
        - E_remain:          1维，剩余电量
        - p_delivery:        1维，配送进度

        Args:
            uav_pos:    无人机位置 (3,)
            uav_vel:    无人机速度 (3,)
            soc:        剩余电量 (scalar, 0~1)
            d_goal:     到最近未完成目标的距离 (scalar)
            d_obs:      到最近障碍物的距离 (scalar)
            d_uav:      到最近其他无人机的距离 (scalar)
            p_delivery: 配送进度 (scalar, 0~1)

        Returns:
            state: 编码后的状态 (11,)
        """
        # 位置(3) + 速度(3)
        norm_pos = self.normalize_position(uav_pos)
        norm_vel = self.normalize_velocity(uav_vel)

        # d_goal(1) + d_obs(1) + d_uav(1)
        norm_d_goal = np.array([self.normalize_distance(d_goal)])
        norm_d_obs = np.array([self.normalize_distance(d_obs)])
        norm_d_uav = np.array([self.normalize_distance(d_uav)])

        # E_remain(1)
        norm_soc = np.array([self.normalize_soc(soc)])

        # p_delivery(1)
        norm_p = np.array([self.normalize_delivery_progress(p_delivery)])

        # 拼接11维状态: pos(3) + vel(3) + d_goal(1) + d_obs(1) + d_uav(1) + E_remain(1) + p_delivery(1)
        state = np.concatenate([
            norm_pos, norm_vel,
            norm_d_goal, norm_d_obs, norm_d_uav,
            norm_soc, norm_p
        ])

        return state.astype(np.float32)
