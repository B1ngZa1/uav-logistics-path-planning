"""
碰撞检测模块
项目：基于深度强化学习的多无人机协同物流配送路径规划
"""

import numpy as np


class CollisionDetector:
    """碰撞检测器"""

    def __init__(self, config):
        self.d_safe = config.get('d_safe', 5.0)       # 无人机间安全距离
        self.d_obs = config.get('d_obs', 3.0)          # 障碍物安全距离
        self.space_size = config.get('space_size', (100, 100, 50))

    def check_uav_collision(self, pos_i, pos_j):
        """
        检测两架无人机是否碰撞

        Args:
            pos_i: 无人机i位置
            pos_j: 无人机j位置

        Returns:
            collision: 是否碰撞
            distance: 距离
        """
        distance = np.linalg.norm(pos_i - pos_j)
        return distance < self.d_safe, distance

    def check_obstacle_collision(self, uav_pos, obs_pos, obs_radius):
        """
        检测无人机与障碍物是否碰撞

        Args:
            uav_pos: 无人机位置
            obs_pos: 障碍物中心位置
            obs_radius: 障碍物半径

        Returns:
            collision: 是否碰撞
            distance: 距离
        """
        distance = np.linalg.norm(uav_pos - obs_pos)
        return distance < obs_radius + self.d_obs, distance

    def check_boundary(self, position):
        """
        检测是否超出边界

        Args:
            position: 位置

        Returns:
            in_bounds: 是否在边界内
        """
        return (0 <= position[0] <= self.space_size[0] and
                0 <= position[1] <= self.space_size[1] and
                10 <= position[2] <= self.space_size[2])

    def detect_all(self, uav_positions, obstacle_positions, obstacle_radii):
        """
        检测所有碰撞

        Returns:
            uav_collisions: 无人机间碰撞列表
            obs_collisions: 障碍物碰撞列表
        """
        uav_collisions = []
        obs_collisions = []

        # 无人机间碰撞
        for i in range(len(uav_positions)):
            for j in range(i + 1, len(uav_positions)):
                collision, dist = self.check_uav_collision(uav_positions[i], uav_positions[j])
                if collision:
                    uav_collisions.append((i, j, dist))

        # 障碍物碰撞
        for i in range(len(uav_positions)):
            for k in range(len(obstacle_positions)):
                collision, dist = self.check_obstacle_collision(
                    uav_positions[i], obstacle_positions[k], obstacle_radii[k])
                if collision:
                    obs_collisions.append((i, k, dist))

        return uav_collisions, obs_collisions

    def compute_distance_field(self, grid_resolution=5.0):
        """
        预计算距离场（可选优化）

        Args:
            grid_resolution: 网格分辨率

        Returns:
            distance_field: 三维距离场
        """
        nx = int(self.space_size[0] / grid_resolution)
        ny = int(self.space_size[1] / grid_resolution)
        nz = int(self.space_size[2] / grid_resolution)

        # TODO: 实现距离场计算
        distance_field = np.ones((nx, ny, nz)) * np.inf
        return distance_field
