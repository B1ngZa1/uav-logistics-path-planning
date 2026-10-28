"""
三维可视化模块
项目：基于深度强化学习的多无人机协同物流配送路径规划
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import List, Tuple, Optional, Dict
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class AirspaceVisualizer:
    """三维空域可视化器"""

    def __init__(self, airspace):
        """
        初始化可视化器

        Args:
            airspace: Airspace3D实例
        """
        self.airspace = airspace
        self.fig = None
        self.ax = None

    def plot_static(self, show_obstacles: bool = True,
                    show_pads: bool = True,
                    show_paths: bool = False,
                    paths: Optional[List[np.ndarray]] = None,
                    figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
        """
        静态三维可视化

        Args:
            show_obstacles: 是否显示障碍物
            show_pads: 是否显示起降点
            show_paths: 是否显示路径
            paths: 路径列表
            figsize: 图形大小

        Returns:
            matplotlib图形对象
        """
        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot(111, projection='3d')

        # 设置坐标轴
        self.ax.set_xlim(0, self.airspace.size[0])
        self.ax.set_ylim(0, self.airspace.size[1])
        self.ax.set_zlim(0, self.airspace.size[2])
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.set_title('三维空域环境')

        # 绘制障碍物
        if show_obstacles:
            self._draw_obstacles()

        # 绘制起降点
        if show_pads:
            self._draw_landing_pads()

        # 绘制路径
        if show_paths and paths:
            self._draw_paths(paths)

        # 绘制边界
        self._draw_boundary()

        plt.tight_layout()
        return self.fig

    def _draw_obstacles(self):
        """绘制障碍物"""
        # 长方体障碍物
        for obs in self.airspace.box_obstacles:
            self._draw_box(obs.min_point, obs.max_point, color='gray', alpha=0.6)

        # 圆柱体障碍物
        for obs in self.airspace.cylinder_obstacles:
            self._draw_cylinder(
                obs.center, obs.radius, obs.height,
                color='brown', alpha=0.6
            )

        # 禁飞区
        for nfz in self.airspace.no_fly_zones:
            self._draw_no_fly_zone(
                nfz.vertices, nfz.z_min, nfz.z_max,
                color='red', alpha=0.3
            )

    def _draw_box(self, min_point: Tuple[float, float, float],
                  max_point: Tuple[float, float, float],
                  color: str = 'gray', alpha: float = 0.6):
        """绘制长方体"""
        x0, y0, z0 = min_point
        x1, y1, z1 = max_point

        # 定义8个顶点
        vertices = [
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]
        ]

        # 定义6个面
        faces = [
            [vertices[0], vertices[1], vertices[5], vertices[4]],
            [vertices[2], vertices[3], vertices[7], vertices[6]],
            [vertices[0], vertices[3], vertices[7], vertices[4]],
            [vertices[1], vertices[2], vertices[6], vertices[5]],
            [vertices[0], vertices[1], vertices[2], vertices[3]],
            [vertices[4], vertices[5], vertices[6], vertices[7]]
        ]

        collection = Poly3DCollection(faces, alpha=alpha, facecolor=color,
                                       edgecolor='black', linewidth=0.5)
        self.ax.add_collection3d(collection)

    def _draw_cylinder(self, center: Tuple[float, float], radius: float,
                       height: float, color: str = 'brown',
                       alpha: float = 0.6):
        """绘制圆柱体"""
        theta = np.linspace(0, 2 * np.pi, 30)
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)

        # 底面
        z_bottom = np.zeros_like(theta)
        self.ax.plot(x, y, z_bottom, color=color, alpha=alpha)

        # 顶面
        z_top = np.full_like(theta, height)
        self.ax.plot(x, y, z_top, color=color, alpha=alpha)

        # 侧面（采样绘制）
        for i in range(0, len(theta), 6):
            self.ax.plot(
                [x[i], x[i]], [y[i], y[i]], [0, height],
                color=color, alpha=alpha * 0.5
            )

        # 填充顶面和底面
        from matplotlib.patches import Circle
        import mpl_toolkits.mplot3d.art3d as art3d

        # 使用Poly3DCollection近似填充
        n_slices = 12
        for i in range(n_slices):
            angle1 = 2 * np.pi * i / n_slices
            angle2 = 2 * np.pi * (i + 1) / n_slices
            x1 = center[0] + radius * np.cos(angle1)
            y1 = center[1] + radius * np.sin(angle1)
            x2 = center[0] + radius * np.cos(angle2)
            y2 = center[1] + radius * np.sin(angle2)

            # 底面三角形
            verts_bottom = [
                [center[0], center[1], 0],
                [x1, y1, 0],
                [x2, y2, 0]
            ]
            # 顶面三角形
            verts_top = [
                [center[0], center[1], height],
                [x1, y1, height],
                [x2, y2, height]
            ]

            self.ax.add_collection3d(
                Poly3DCollection([verts_bottom], alpha=alpha * 0.3, facecolor=color)
            )
            self.ax.add_collection3d(
                Poly3DCollection([verts_top], alpha=alpha * 0.3, facecolor=color)
            )

    def _draw_no_fly_zone(self, vertices: List[Tuple[float, float]],
                          z_min: float, z_max: float,
                          color: str = 'red', alpha: float = 0.3):
        """绘制禁飞区"""
        n = len(vertices)

        # 绘制底面和顶面
        for z in [z_min, z_max]:
            verts = [[v[0], v[1], z] for v in vertices]
            self.ax.add_collection3d(
                Poly3DCollection([verts], alpha=alpha, facecolor=color,
                                 edgecolor='red', linewidth=1)
            )

        # 绘制侧面
        for i in range(n):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % n]
            verts = [
                [v1[0], v1[1], z_min],
                [v2[0], v2[1], z_min],
                [v2[0], v2[1], z_max],
                [v1[0], v1[1], z_max]
            ]
            self.ax.add_collection3d(
                Poly3DCollection([verts], alpha=alpha * 0.5, facecolor=color)
            )

    def _draw_landing_pads(self):
        """绘制起降点"""
        # 配送站（绿色）
        for depot in self.airspace.depots:
            x, y, z = depot.position
            size = depot.size[0] / 2
            verts = [
                [x - size, y - size, z],
                [x + size, y - size, z],
                [x + size, y + size, z],
                [x - size, y + size, z]
            ]
            self.ax.add_collection3d(
                Poly3DCollection([verts], alpha=0.8, facecolor='green',
                                 edgecolor='darkgreen', linewidth=2)
            )
            self.ax.text(x, y, z + 1, 'D', fontsize=8, color='darkgreen')

        # 客户点（蓝色）
        for customer in self.airspace.customers:
            x, y, z = customer.position
            size = customer.size[0] / 2
            verts = [
                [x - size, y - size, z],
                [x + size, y - size, z],
                [x + size, y + size, z],
                [x - size, y + size, z]
            ]
            self.ax.add_collection3d(
                Poly3DCollection([verts], alpha=0.8, facecolor='blue',
                                 edgecolor='darkblue', linewidth=2)
            )
            self.ax.text(x, y, z + 1, 'C', fontsize=8, color='darkblue')

    def _draw_paths(self, paths: List[np.ndarray]):
        """绘制飞行路径"""
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        for i, path in enumerate(paths):
            color = colors[i % len(colors)]
            self.ax.plot(path[:, 0], path[:, 1], path[:, 2],
                        color=color, linewidth=2, label=f'UAV {i+1}')

            # 绘制起点和终点
            self.ax.scatter(*path[0], color=color, s=100, marker='o')
            self.ax.scatter(*path[-1], color=color, s=100, marker='*')

        self.ax.legend()

    def _draw_boundary(self):
        """绘制空域边界"""
        x_max, y_max, z_max = self.airspace.size

        # 绘制底面边界
        boundary_x = [0, x_max, x_max, 0, 0]
        boundary_y = [0, 0, y_max, y_max, 0]
        boundary_z = [0, 0, 0, 0, 0]
        self.ax.plot(boundary_x, boundary_y, boundary_z, 'k-', linewidth=1)

        # 绘制顶面边界
        boundary_z_top = [z_max] * 5
        self.ax.plot(boundary_x, boundary_y, boundary_z_top, 'k-', linewidth=1)

        # 绘制垂直边
        for x, y in [(0, 0), (x_max, 0), (x_max, y_max), (0, y_max)]:
            self.ax.plot([x, x], [y, y], [0, z_max], 'k-', linewidth=1)

    def plot_voxel_map(self, threshold: int = 1,
                       figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
        """
        体素地图可视化

        Args:
            threshold: 显示的体素值阈值
            figsize: 图形大小

        Returns:
            matplotlib图形对象
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        # 获取障碍物体素位置
        obstacle_positions = np.argwhere(
            self.airspace.voxel_map.data >= threshold
        )

        if len(obstacle_positions) > 0:
            # 转换为世界坐标
            world_positions = (
                obstacle_positions * self.airspace.resolution +
                self.airspace.resolution / 2
            )

            # 按类型着色
            colors = []
            for pos in obstacle_positions:
                val = self.airspace.voxel_map.data[pos[0], pos[1], pos[2]]
                if val == 1:
                    colors.append('gray')    # 障碍物
                elif val == 2:
                    colors.append('red')     # 禁飞区
                else:
                    colors.append('orange')  # 其他

            ax.scatter(
                world_positions[:, 0],
                world_positions[:, 1],
                world_positions[:, 2],
                c=colors, alpha=0.6, s=50
            )

        ax.set_xlim(0, self.airspace.size[0])
        ax.set_ylim(0, self.airspace.size[1])
        ax.set_zlim(0, self.airspace.size[2])
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('体素地图可视化')

        plt.tight_layout()
        return fig

    def plot_distance_field(self, z_level: int = 5,
                            figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        距离场切片可视化

        Args:
            z_level: Z方向切片层数
            figsize: 图形大小

        Returns:
            matplotlib图形对象
        """
        if self.airspace.voxel_map.distance_field is None:
            self.airspace.voxel_map.compute_distance_field()

        fig, ax = plt.subplots(figsize=figsize)

        # 获取指定高度的距离场切片
        if z_level >= self.airspace.voxel_map.nz:
            z_level = self.airspace.voxel_map.nz - 1

        distance_slice = self.airspace.voxel_map.distance_field[:, :, z_level]

        im = ax.imshow(
            distance_slice.T,
            origin='lower',
            extent=[0, self.airspace.size[0], 0, self.airspace.size[1]],
            cmap='viridis'
        )

        plt.colorbar(im, ax=ax, label='距离 (m)')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        z_height = (z_level + 0.5) * self.airspace.resolution
        ax.set_title(f'障碍物距离场 (Z = {z_height:.1f} m)')

        plt.tight_layout()
        return fig

    def plot_top_view(self, z_level: int = 5,
                      figsize: Tuple[int, int] = (10, 10)) -> plt.Figure:
        """
        俯视图

        Args:
            z_level: Z方向切片层数
            figsize: 图形大小

        Returns:
            matplotlib图形对象
        """
        fig, ax = plt.subplots(figsize=figsize)

        # 绘制障碍物切片
        if z_level < self.airspace.voxel_map.nz:
            obstacle_slice = self.airspace.voxel_map.data[:, :, z_level]
            obstacle_mask = obstacle_slice > 0

            # 创建RGB图像
            img = np.ones((*obstacle_slice.shape, 3))
            img[obstacle_mask] = [0.5, 0.5, 0.5]  # 灰色障碍物

            ax.imshow(
                img.transpose(1, 0, 2),
                origin='lower',
                extent=[0, self.airspace.size[0], 0, self.airspace.size[1]]
            )

        # 绘制起降点
        for depot in self.airspace.depots:
            ax.plot(depot.position[0], depot.position[1], 'g^',
                    markersize=15, label='配送站')

        for customer in self.airspace.customers:
            ax.plot(customer.position[0], customer.position[1], 'bs',
                    markersize=10, label='客户点')

        ax.set_xlim(0, self.airspace.size[0])
        ax.set_ylim(0, self.airspace.size[1])
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        z_height = (z_level + 0.5) * self.airspace.resolution
        ax.set_title(f'俯视图 (Z = {z_height:.1f} m)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def save_figure(self, filename: str, dpi: int = 150):
        """保存图形"""
        if self.fig is not None:
            os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
            self.fig.savefig(filename, dpi=dpi, bbox_inches='tight')
            print(f"图形已保存至: {filename}")
