"""
三维空域环境模块
项目：基于深度强化学习的多无人机协同物流配送路径规划
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from enum import IntEnum


class VoxelState(IntEnum):
    """体素状态枚举"""
    FREE = 0        # 自由空间
    OBSTACLE = 1    # 静态障碍物
    NO_FLY = 2      # 禁飞区
    DYNAMIC = 3     # 动态障碍物区域


@dataclass
class BoxObstacle:
    """长方体障碍物"""
    min_point: Tuple[float, float, float]  # 西南角底面 (x_min, y_min, z_min)
    max_point: Tuple[float, float, float]  # 东北角顶面 (x_max, y_max, z_max)

    def contains_point(self, x: float, y: float, z: float) -> bool:
        """判断点是否在长方体内部"""
        return (self.min_point[0] <= x <= self.max_point[0] and
                self.min_point[1] <= y <= self.max_point[1] and
                self.min_point[2] <= z <= self.max_point[2])

    def get_dimensions(self) -> Tuple[float, float, float]:
        """获取长宽高"""
        return (
            self.max_point[0] - self.min_point[0],
            self.max_point[1] - self.min_point[1],
            self.max_point[2] - self.min_point[2]
        )


@dataclass
class CylinderObstacle:
    """圆柱体障碍物"""
    center: Tuple[float, float]    # 底面圆心 (cx, cy)
    radius: float                  # 底面半径
    height: float                  # 高度

    def contains_point(self, x: float, y: float, z: float) -> bool:
        """判断点是否在圆柱体内部"""
        dx = x - self.center[0]
        dy = y - self.center[1]
        return (dx * dx + dy * dy <= self.radius * self.radius and
                0 <= z <= self.height)


@dataclass
class NoFlyZone:
    """禁飞区"""
    vertices: List[Tuple[float, float]]  # 多边形顶点列表
    z_min: float                         # 最低高度
    z_max: float                         # 最高高度

    def contains_point_xy(self, x: float, y: float) -> bool:
        """射线法判断点是否在多边形内部（仅XY平面）"""
        n = len(self.vertices)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = self.vertices[i]
            xj, yj = self.vertices[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    def contains_point(self, x: float, y: float, z: float) -> bool:
        """判断点是否在禁飞区内部"""
        return (self.contains_point_xy(x, y) and
                self.z_min <= z <= self.z_max)


@dataclass
class DynamicObstacle:
    """动态障碍物"""
    position: np.ndarray      # 当前位置 [x, y, z]
    velocity: np.ndarray      # 速度 [vx, vy, vz]
    radius: float             # 安全半径

    def update(self, dt: float):
        """更新位置"""
        self.position += self.velocity * dt

    def contains_point(self, x: float, y: float, z: float) -> bool:
        """判断点是否在动态障碍物内部"""
        dx = x - self.position[0]
        dy = y - self.position[1]
        dz = z - self.position[2]
        return (dx * dx + dy * dy + dz * dz <= self.radius * self.radius)


@dataclass
class LandingPad:
    """起降点"""
    position: Tuple[float, float, float]  # 中心位置
    size: Tuple[float, float] = (10.0, 10.0)  # 起降场尺寸
    is_depot: bool = False                 # 是否为配送站

    def is_within(self, x: float, y: float, z: float, threshold: float = 5.0) -> bool:
        """判断位置是否在起降点范围内"""
        dx = abs(x - self.position[0])
        dy = abs(y - self.position[1])
        dz = abs(z - self.position[2])
        return dx <= threshold and dy <= threshold and dz <= 1.0


class VoxelMap:
    """三维体素地图"""

    def __init__(self, nx: int, ny: int, nz: int, resolution: float):
        """
        初始化体素地图

        Args:
            nx: X方向体素数量
            ny: Y方向体素数量
            nz: Z方向体素数量
            resolution: 体素分辨率（米）
        """
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.resolution = resolution
        self.data = np.zeros((nx, ny, nz), dtype=np.int8)
        self.distance_field = None

    def world_to_voxel(self, x: float, y: float, z: float) -> Tuple[int, int, int]:
        """世界坐标转体素索引（向下取整）"""
        i = min(int(x / self.resolution), self.nx - 1)
        j = min(int(y / self.resolution), self.ny - 1)
        k = min(int(z / self.resolution), self.nz - 1)
        return (max(i, 0), max(j, 0), max(k, 0))

    def voxel_to_world(self, i: int, j: int, k: int) -> Tuple[float, float, float]:
        """体素索引转世界坐标（体素中心点）"""
        x = (i + 0.5) * self.resolution
        y = (j + 0.5) * self.resolution
        z = (k + 0.5) * self.resolution
        return (x, y, z)

    def voxel_to_index(self, i: int, j: int, k: int) -> int:
        """三维索引转一维索引"""
        return i * (self.ny * self.nz) + j * self.nz + k

    def index_to_voxel(self, idx: int) -> Tuple[int, int, int]:
        """一维索引转三维索引"""
        i = idx // (self.ny * self.nz)
        j = (idx % (self.ny * self.nz)) // self.nz
        k = idx % self.nz
        return (i, j, k)

    def set_voxel(self, i: int, j: int, k: int, value: int):
        """设置体素值"""
        if 0 <= i < self.nx and 0 <= j < self.ny and 0 <= k < self.nz:
            self.data[i, j, k] = value

    def get_voxel(self, i: int, j: int, k: int) -> int:
        """获取体素值"""
        if 0 <= i < self.nx and 0 <= j < self.ny and 0 <= k < self.nz:
            return int(self.data[i, j, k])
        return -1  # 越界返回-1

    def is_free(self, i: int, j: int, k: int) -> bool:
        """检查体素是否为自由空间"""
        return self.get_voxel(i, j, k) == VoxelState.FREE

    def is_obstacle(self, i: int, j: int, k: int) -> bool:
        """检查体素是否为障碍物"""
        return self.get_voxel(i, j, k) in (VoxelState.OBSTACLE, VoxelState.NO_FLY)

    def get_neighbors(self, i: int, j: int, k: int,
                      connectivity: int = 26) -> List[Tuple[int, int, int]]:
        """
        获取邻域体素

        Args:
            i, j, k: 当前体素索引
            connectivity: 连通性（6, 18, 26）

        Returns:
            邻域体素索引列表
        """
        neighbors = []
        for di in range(-1, 2):
            for dj in range(-1, 2):
                for dk in range(-1, 2):
                    if di == 0 and dj == 0 and dk == 0:
                        continue

                    # 根据连通性过滤
                    dist = abs(di) + abs(dj) + abs(dk)
                    if connectivity == 6 and dist > 1:
                        continue
                    if connectivity == 18 and dist > 2:
                        continue

                    ni, nj, nk = i + di, j + dj, k + dk
                    if (0 <= ni < self.nx and
                        0 <= nj < self.ny and
                            0 <= nk < self.nz):
                        neighbors.append((ni, nj, nk))
        return neighbors

    def compute_distance_field(self) -> np.ndarray:
        """
        计算障碍物距离场

        Returns:
            距离场数组，每个体素存储到最近障碍物的距离（米）
        """
        from scipy.ndimage import distance_transform_edt

        # 创建二值障碍物掩码
        obstacle_mask = (self.data != VoxelState.FREE).astype(np.float32)

        # 使用欧氏距离变换
        # 注意：distance_transform_edt计算到0的距离，所以需要反转
        free_mask = (self.data == VoxelState.FREE).astype(np.float32)
        self.distance_field = distance_transform_edt(free_mask) * self.resolution

        return self.distance_field

    def get_distance_at(self, x: float, y: float, z: float) -> float:
        """获取指定世界坐标处的障碍物距离"""
        if self.distance_field is None:
            self.compute_distance_field()
        i, j, k = self.world_to_voxel(x, y, z)
        return float(self.distance_field[i, j, k])

    def fill_box(self, min_point: Tuple[float, float, float],
                 max_point: Tuple[float, float, float], value: int):
        """填充长方体区域"""
        i_min, j_min, k_min = self.world_to_voxel(*min_point)
        i_max, j_max, k_max = self.world_to_voxel(*max_point)

        for i in range(i_min, min(i_max + 1, self.nx)):
            for j in range(j_min, min(j_max + 1, self.ny)):
                for k in range(k_min, min(k_max + 1, self.nz)):
                    self.data[i, j, k] = value

    def fill_cylinder(self, center: Tuple[float, float], radius: float,
                      height: float, value: int):
        """填充圆柱体区域"""
        cx, cy = center
        resolution = self.resolution

        for i in range(self.nx):
            for j in range(self.ny):
                x = (i + 0.5) * resolution
                y = (j + 0.5) * resolution
                dx = x - cx
                dy = y - cy
                if dx * dx + dy * dy <= radius * radius:
                    k_max = min(int(height / resolution), self.nz)
                    for k in range(k_max):
                        self.data[i, j, k] = value

    def fill_polygon_xy(self, vertices: List[Tuple[float, float]],
                        z_min: float, z_max: float, value: int):
        """填充多边形区域（XY平面）"""
        nfz = NoFlyZone(vertices, z_min, z_max)
        resolution = self.resolution

        for i in range(self.nx):
            for j in range(self.ny):
                x = (i + 0.5) * resolution
                y = (j + 0.5) * resolution
                if nfz.contains_point_xy(x, y):
                    k_min = max(0, int(z_min / resolution))
                    k_max = min(int(z_max / resolution), self.nz)
                    for k in range(k_min, k_max):
                        self.data[i, j, k] = value


class Airspace3D:
    """三维空域环境"""

    def __init__(self, config: Dict = None):
        """
        初始化三维空域

        Args:
            config: 环境配置字典
        """
        if config is None:
            config = {}

        # 空域参数
        self.size = config.get('airspace_size', (100, 100, 50))
        self.resolution = config.get('resolution', 5)
        self.voxel_count = (
            int(self.size[0] / self.resolution),
            int(self.size[1] / self.resolution),
            int(self.size[2] / self.resolution)
        )

        # 体素地图
        self.voxel_map = VoxelMap(*self.voxel_count, self.resolution)

        # 障碍物列表
        self.box_obstacles: List[BoxObstacle] = []
        self.cylinder_obstacles: List[CylinderObstacle] = []
        self.no_fly_zones: List[NoFlyZone] = []
        self.dynamic_obstacles: List[DynamicObstacle] = []

        # 起降点
        self.depots: List[LandingPad] = []
        self.customers: List[LandingPad] = []

        # 是否已构建
        self._built = False

    def add_box_obstacle(self, min_point: Tuple[float, float, float],
                         max_point: Tuple[float, float, float]):
        """
        添加长方体障碍物

        Args:
            min_point: 西南角底面坐标 (x_min, y_min, z_min)
            max_point: 东北角顶面坐标 (x_max, y_max, z_max)
        """
        obstacle = BoxObstacle(min_point, max_point)
        self.box_obstacles.append(obstacle)
        self.voxel_map.fill_box(min_point, max_point, VoxelState.OBSTACLE)

    def add_cylinder_obstacle(self, center: Tuple[float, float],
                              radius: float, height: float):
        """
        添加圆柱体障碍物

        Args:
            center: 底面圆心坐标 (cx, cy)
            radius: 底面半径
            height: 高度
        """
        obstacle = CylinderObstacle(center, radius, height)
        self.cylinder_obstacles.append(obstacle)
        self.voxel_map.fill_cylinder(center, radius, height, VoxelState.OBSTACLE)

    def add_no_fly_zone(self, vertices: List[Tuple[float, float]],
                        z_min: float, z_max: float):
        """
        添加多边形禁飞区

        Args:
            vertices: 多边形顶点列表
            z_min: 最低高度
            z_max: 最高高度
        """
        nfz = NoFlyZone(vertices, z_min, z_max)
        self.no_fly_zones.append(nfz)
        self.voxel_map.fill_polygon_xy(vertices, z_min, z_max, VoxelState.NO_FLY)

    def add_dynamic_obstacle(self, position: np.ndarray,
                             velocity: np.ndarray, radius: float):
        """
        添加动态障碍物

        Args:
            position: 初始位置 [x, y, z]
            velocity: 速度 [vx, vy, vz]
            radius: 安全半径
        """
        obstacle = DynamicObstacle(
            position=np.array(position, dtype=np.float32),
            velocity=np.array(velocity, dtype=np.float32),
            radius=radius
        )
        self.dynamic_obstacles.append(obstacle)

    def add_depot(self, position: Tuple[float, float, float],
                  size: Tuple[float, float] = (10.0, 10.0)):
        """添加配送站"""
        depot = LandingPad(position, size, is_depot=True)
        self.depots.append(depot)

    def add_customer(self, position: Tuple[float, float, float],
                     size: Tuple[float, float] = (10.0, 10.0)):
        """添加客户点"""
        customer = LandingPad(position, size, is_depot=False)
        self.customers.append(customer)

    def build(self):
        """构建环境（计算距离场等）"""
        self.voxel_map.compute_distance_field()
        self._built = True

    def is_valid_position(self, position: np.ndarray,
                          radius: float = 0.5) -> bool:
        """
        检查位置是否有效（无碰撞）

        Args:
            position: 位置 [x, y, z]
            radius: 无人机半径

        Returns:
            是否有效
        """
        x, y, z = position

        # 边界检查
        if (x < 0 or x > self.size[0] or
            y < 0 or y > self.size[1] or
                z < 0 or z > self.size[2]):
            return False

        # 体素碰撞检查
        i, j, k = self.voxel_map.world_to_voxel(x, y, z)
        if self.voxel_map.is_obstacle(i, j, k):
            return False

        # 邻域检查（考虑无人机半径）
        if radius > 0:
            check_range = max(1, int(np.ceil(radius / self.resolution)))
            for di in range(-check_range, check_range + 1):
                for dj in range(-check_range, check_range + 1):
                    for dk in range(-check_range, check_range + 1):
                        ni, nj, nk = i + di, j + dj, k + dk
                        if self.voxel_map.is_obstacle(ni, nj, nk):
                            # 计算点到体素的距离
                            vx, vy, vz = self.voxel_map.voxel_to_world(ni, nj, nk)
                            dist = np.sqrt(
                                (x - vx) ** 2 + (y - vy) ** 2 + (z - vz) ** 2
                            )
                            if dist < radius + self.resolution * 0.5:
                                return False

        return True

    def get_nearest_obstacle_distance(self, position: np.ndarray) -> float:
        """获取最近障碍物距离"""
        if not self._built:
            self.build()
        return self.voxel_map.get_distance_at(*position)

    def update_dynamic_obstacles(self, dt: float):
        """更新动态障碍物位置"""
        for obs in self.dynamic_obstacles:
            obs.update(dt)

    def get_state(self) -> np.ndarray:
        """获取环境状态（体素地图副本）"""
        return self.voxel_map.data.copy()

    def get_obstacle_positions(self) -> np.ndarray:
        """获取所有障碍物体素位置"""
        positions = np.argwhere(self.voxel_map.data != VoxelState.FREE)
        return positions * self.resolution + self.resolution / 2

    def create_default_city_scene(self):
        """创建默认城市场景"""
        # 建筑物
        self.add_box_obstacle((20, 40, 0), (40, 55, 30))
        self.add_box_obstacle((60, 30, 0), (75, 50, 25))
        self.add_box_obstacle((10, 70, 0), (35, 80, 20))
        self.add_box_obstacle((70, 60, 0), (80, 90, 15))
        self.add_box_obstacle((30, 10, 0), (50, 30, 22))
        self.add_box_obstacle((75, 15, 0), (90, 30, 18))

        # 圆柱体建筑
        self.add_cylinder_obstacle((45, 70), 8, 35)
        self.add_cylinder_obstacle((55, 25), 5, 40)

        # 禁飞区
        self.add_no_fly_zone(
            [(35, 85), (45, 90), (40, 95), (30, 90)],
            z_min=0, z_max=50
        )

        # 配送站
        self.add_depot((90, 50, 0))
        self.add_depot((10, 50, 0))
        self.add_depot((50, 90, 0))

        # 客户点
        customer_positions = [
            (20, 80, 0), (30, 70, 0), (40, 85, 0),
            (60, 75, 0), (70, 80, 0), (80, 60, 0),
            (20, 30, 0), (35, 20, 0), (65, 25, 0),
            (75, 40, 0), (15, 55, 0), (45, 45, 0),
            (55, 60, 0), (25, 45, 0), (85, 30, 0)
        ]
        for pos in customer_positions:
            self.add_customer(pos)

        self.build()
