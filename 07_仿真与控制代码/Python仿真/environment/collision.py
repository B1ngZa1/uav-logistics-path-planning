"""
碰撞检测模块
项目：基于深度强化学习的多无人机协同物流配送路径规划
"""

import numpy as np
from typing import List, Tuple, Optional
from .airspace import Airspace3D, VoxelMap


class CollisionDetector:
    """碰撞检测器"""

    def __init__(self, airspace: Airspace3D, uav_radius: float = 0.5,
                 safe_distance: float = 2.0):
        """
        初始化碰撞检测器

        Args:
            airspace: 三维空域环境
            uav_radius: 无人机包围球半径（米）
            safe_distance: 无人机间最小安全距离（米）
        """
        self.airspace = airspace
        self.uav_radius = uav_radius
        self.safe_distance = safe_distance
        self.voxel_map = airspace.voxel_map
        self.resolution = airspace.resolution

    def check_point_obstacle(self, position: np.ndarray) -> bool:
        """
        检测点与障碍物的碰撞

        Args:
            position: 位置 [x, y, z]

        Returns:
            是否碰撞
        """
        x, y, z = position

        # 边界检测
        if (x < 0 or x > self.airspace.size[0] or
            y < 0 or y > self.airspace.size[1] or
                z < 0 or z > self.airspace.size[2]):
            return True

        # 体素碰撞检测
        i, j, k = self.voxel_map.world_to_voxel(x, y, z)
        return self.voxel_map.is_obstacle(i, j, k)

    def check_sphere_obstacle(self, position: np.ndarray) -> bool:
        """
        检测球形包围盒与障碍物的碰撞

        Args:
            position: 无人机位置 [x, y, z]

        Returns:
            是否碰撞
        """
        x, y, z = position
        radius = self.uav_radius

        # 边界检测
        if (x - radius < 0 or x + radius > self.airspace.size[0] or
            y - radius < 0 or y + radius > self.airspace.size[1] or
                z - radius < 0 or z + radius > self.airspace.size[2]):
            return True

        # 获取可能碰撞的体素范围
        i, j, k = self.voxel_map.world_to_voxel(x, y, z)
        check_range = max(1, int(np.ceil(radius / self.resolution)))

        for di in range(-check_range, check_range + 1):
            for dj in range(-check_range, check_range + 1):
                for dk in range(-check_range, check_range + 1):
                    ni, nj, nk = i + di, j + dj, k + dk
                    if self.voxel_map.is_obstacle(ni, nj, nk):
                        # 计算点到体素的最短距离
                        dist = self._point_to_voxel_distance(
                            position, ni, nj, nk
                        )
                        if dist < radius:
                            return True

        return False

    def check_uav_collision(self, pos1: np.ndarray,
                            pos2: np.ndarray) -> bool:
        """
        检测两架无人机之间的碰撞

        Args:
            pos1: 无人机1位置 [x, y, z]
            pos2: 无人机2位置 [x, y, z]

        Returns:
            是否碰撞（距离小于安全距离）
        """
        dist = np.linalg.norm(pos1 - pos2)
        return dist < self.safe_distance

    def check_multi_uav_collision(self, positions: np.ndarray) -> List[Tuple[int, int]]:
        """
        检测多架无人机之间的碰撞

        Args:
            positions: 无人机位置数组，形状为 (N, 3)

        Returns:
            碰撞的无人机对列表 [(i, j), ...]
        """
        n = len(positions)
        collisions = []
        for i in range(n):
            for j in range(i + 1, n):
                if self.check_uav_collision(positions[i], positions[j]):
                    collisions.append((i, j))
        return collisions

    def check_trajectory_collision(self, start: np.ndarray,
                                   end: np.ndarray,
                                   num_checks: int = 10) -> bool:
        """
        检测轨迹碰撞（线段与障碍物）

        Args:
            start: 起点 [x, y, z]
            end: 终点 [x, y, z]
            num_checks: 检查点数量

        Returns:
            轨迹是否碰撞
        """
        for t in np.linspace(0, 1, num_checks):
            position = start + t * (end - start)
            if self.check_sphere_obstacle(position):
                return True
        return False

    def _point_to_voxel_distance(self, point: np.ndarray,
                                 i: int, j: int, k: int) -> float:
        """
        计算点到体素的最短距离

        Args:
            point: 点坐标 [x, y, z]
            i, j, k: 体素索引

        Returns:
            最短距离（米）
        """
        # 体素中心坐标
        vx = (i + 0.5) * self.resolution
        vy = (j + 0.5) * self.resolution
        vz = (k + 0.5) * self.resolution
        half_res = self.resolution / 2

        # 计算点到体素各面的距离
        dx = max(0, abs(point[0] - vx) - half_res)
        dy = max(0, abs(point[1] - vy) - half_res)
        dz = max(0, abs(point[2] - vz) - half_res)

        return np.sqrt(dx * dx + dy * dy + dz * dz)

    def get_nearest_obstacle_info(self, position: np.ndarray) -> dict:
        """
        获取最近障碍物信息

        Args:
            position: 位置 [x, y, z]

        Returns:
            包含距离和方向的字典
        """
        x, y, z = position
        i, j, k = self.voxel_map.world_to_voxel(x, y, z)

        # 搜索范围
        search_range = 5
        min_dist = float('inf')
        nearest_pos = None

        for di in range(-search_range, search_range + 1):
            for dj in range(-search_range, search_range + 1):
                for dk in range(-search_range, search_range + 1):
                    ni, nj, nk = i + di, j + dj, k + dk
                    if self.voxel_map.is_obstacle(ni, nj, nk):
                        dist = self._point_to_voxel_distance(
                            position, ni, nj, nk
                        )
                        if dist < min_dist:
                            min_dist = dist
                            nearest_pos = self.voxel_map.voxel_to_world(
                                ni, nj, nk
                            )

        return {
            'distance': min_dist,
            'nearest_position': np.array(nearest_pos) if nearest_pos else None
        }

    def get_nearest_uav_distance(self, uav_index: int,
                                  positions: np.ndarray) -> float:
        """
        获取最近其他无人机的距离

        Args:
            uav_index: 当前无人机索引
            positions: 所有无人机位置数组

        Returns:
            最近距离
        """
        if len(positions) <= 1:
            return float('inf')

        pos = positions[uav_index]
        other_positions = np.delete(positions, uav_index, axis=0)
        distances = np.linalg.norm(other_positions - pos, axis=1)
        return float(np.min(distances))
