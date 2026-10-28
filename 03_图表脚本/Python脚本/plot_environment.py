"""
三维空域环境可视化
项目：基于深度强化学习的多无人机协同物流配送路径规划

输出图片：
  - environment_3d_view.png    三维空域环境图
  - environment_top_view.png   俯视图
  - environment_voxel_map.png  体素地图
  - environment_scenarios.png  场景配置对比图
"""

import sys
import os

# 将仿真代码目录加入路径，以便导入 airspace 模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '07_仿真与控制代码', 'Python仿真'))

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

# ==================== SCI 风格常量 ====================

SCI_PRIMARY   = '#4CAF50'
SCI_SECONDARY = '#81C784'
SCI_TERTIARY  = '#A5D6A7'
SCI_ACCENT    = '#2E7D32'
SCI_LIGHT     = '#E8F5E9'
SCI_BG        = '#F1F8E9'

SCI_SERIES = [
    '#4CAF50',  # 绿
    '#2196F3',  # 蓝
    '#FF9800',  # 橙
    '#9C27B0',  # 紫
    '#F44336',  # 红
    '#00BCD4',  # 青
]

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', '04_脚本图片输出', '论文插图'
)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _apply_sci_style():
    """应用 SCI 论文风格到当前 figure"""
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.labelcolor'] = '#333333'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.pad_inches'] = 0.1


# ==================== 环境数据定义 ====================

# 空域尺寸
SPACE_X, SPACE_Y, SPACE_Z = 100, 100, 50

# 建筑物 (BoxObstacle): (x_min, y_min, z_min, x_max, y_max, z_max)
BUILDINGS = [
    (20, 40, 0, 40, 55, 30),
    (60, 30, 0, 75, 50, 25),
    (10, 70, 0, 35, 80, 20),
    (70, 60, 0, 80, 90, 15),
    (30, 10, 0, 50, 30, 22),
    (75, 15, 0, 90, 30, 18),
]

# 圆柱体建筑: (cx, cy, radius, height)
CYLINDERS = [
    (45, 70, 8, 35),
    (55, 25, 5, 40),
]

# 禁飞区
NO_FLY_VERTICES = [(35, 85), (45, 90), (40, 95), (30, 90)]

# 配送站 (depot)
DEPOTS = [
    (90, 50, 0),
    (10, 50, 0),
    (50, 90, 0),
]

# 客户点 (customer / target)
CUSTOMERS = [
    (20, 80, 0), (30, 70, 0), (40, 85, 0),
    (60, 75, 0), (70, 80, 0), (80, 60, 0),
    (20, 30, 0), (35, 20, 0), (65, 25, 0),
    (75, 40, 0), (15, 55, 0), (45, 45, 0),
    (55, 60, 0), (25, 45, 0), (85, 30, 0),
]

# 无人机初始位置
UAV_STARTS = [
    (10, 50, 10),
    (50, 90, 10),
    (90, 50, 10),
]

# 示例目标 (用于路径规划示意的子集)
SAMPLE_TARGETS = [
    (20, 80, 10), (60, 75, 10), (80, 60, 10),
]

# 场景配置 S1-S6
SCENARIOS = {
    'S1': {'uavs': 3,  'targets': 5,  'obstacles': 0,  'label': 'S1 (基础)'},
    'S2': {'uavs': 3,  'targets': 5,  'obstacles': 5,  'label': 'S2 (少量障碍)'},
    'S3': {'uavs': 3,  'targets': 10, 'obstacles': 10, 'label': 'S3 (中等规模)'},
    'S4': {'uavs': 5,  'targets': 15, 'obstacles': 15, 'label': 'S4 (大规模)'},
    'S5': {'uavs': 6,  'targets': 20, 'obstacles': 20, 'label': 'S5 (密集)'},
    'S6': {'uavs': 5,  'targets': 15, 'obstacles': 15, 'label': 'S6 (动态障碍)'},
}


# ==================== 绘图函数 ====================

def _draw_box_obstacle_3d(ax, b, color='#81C784', alpha=0.45, edge='#2E7D32'):
    """在3D轴上绘制长方体障碍物"""
    x0, y0, z0, x1, y1, z1 = b
    # 6 个面
    verts = [
        [(x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0)],  # 底
        [(x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)],  # 顶
        [(x0,y0,z0),(x1,y0,z0),(x1,y0,z1),(x0,y0,z1)],  # 前
        [(x0,y1,z0),(x1,y1,z0),(x1,y1,z1),(x0,y1,z1)],  # 后
        [(x0,y0,z0),(x0,y1,z0),(x0,y1,z1),(x0,y0,z1)],  # 左
        [(x1,y0,z0),(x1,y1,z0),(x1,y1,z1),(x1,y0,z1)],  # 右
    ]
    pc = Poly3DCollection(verts, alpha=alpha, facecolor=color, edgecolor=edge, linewidth=0.6)
    ax.add_collection3d(pc)


def _draw_cylinder_3d(ax, c, radius, height, color='#A5D6A7', alpha=0.4, edge='#2E7D32'):
    """在3D轴上绘制圆柱体障碍物"""
    cx, cy = c
    theta = np.linspace(0, 2*np.pi, 30)
    z_levels = [0, height]
    for z in z_levels:
        xs = cx + radius * np.cos(theta)
        ys = cy + radius * np.sin(theta)
        ax.plot(xs, ys, zs=z, color=edge, linewidth=0.6, alpha=0.8)
    # 侧面网格线
    for t in np.linspace(0, 2*np.pi, 12, endpoint=False):
        x_edge = cx + radius * np.cos(t)
        y_edge = cy + radius * np.sin(t)
        ax.plot([x_edge, x_edge], [y_edge, y_edge], [0, height],
                color=edge, linewidth=0.4, alpha=0.5)
    # 填充顶面和底面
    xs = cx + radius * np.cos(theta)
    ys = cy + radius * np.sin(theta)
    for z in [0, height]:
        verts = [list(zip(xs, ys, [z]*len(xs)))]
        pc = Poly3DCollection(verts, alpha=alpha*0.5, facecolor=color, edgecolor='none')
        ax.add_collection3d(pc)


def plot_environment_3d():
    """生成三维空域环境图"""
    _apply_sci_style()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 建筑物
    for b in BUILDINGS:
        _draw_box_obstacle_3d(ax, b, color=SCI_SECONDARY, alpha=0.45, edge=SCI_ACCENT)

    # 圆柱体
    for c in CYLINDERS:
        _draw_cylinder_3d(ax, (c[0], c[1]), c[2], c[3], color=SCI_TERTIARY, alpha=0.4, edge=SCI_ACCENT)

    # 禁飞区
    nfz_x = [v[0] for v in NO_FLY_VERTICES] + [NO_FLY_VERTICES[0][0]]
    nfz_y = [v[1] for v in NO_FLY_VERTICES] + [NO_FLY_VERTICES[0][1]]
    for z in [0, SPACE_Z]:
        verts = [list(zip(nfz_x, nfz_y, [z]*len(nfz_x)))]
        pc = Poly3DCollection(verts, alpha=0.25, facecolor='#FF9800', edgecolor='#E65100', linewidth=1.0)
        ax.add_collection3d(pc)
    # 竖直边
    for v in NO_FLY_VERTICES:
        ax.plot([v[0], v[0]], [v[1], v[1]], [0, SPACE_Z],
                color='#E65100', linewidth=0.6, alpha=0.5)

    # 配送站
    for d in DEPOTS:
        ax.scatter(*d, marker='s', s=120, c=SCI_ACCENT, edgecolors='white',
                   linewidths=1.2, zorder=5, label='配送站 (Depot)')

    # 客户点
    for ct in CUSTOMERS[:6]:
        ax.scatter(*ct, marker='^', s=80, c='#2196F3', edgecolors='white',
                   linewidths=0.8, zorder=5, label='客户点 (Customer)')

    # 无人机
    for u in UAV_STARTS:
        ax.scatter(*u, marker='D', s=100, c='#F44336', edgecolors='white',
                   linewidths=1.2, zorder=6, label='UAV 起点')

    # 去重图例
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(),
              loc='upper left', fontsize=9, framealpha=0.9, edgecolor='#CCCCCC')

    ax.set_xlim(0, SPACE_X)
    ax.set_ylim(0, SPACE_Y)
    ax.set_zlim(0, SPACE_Z)
    ax.set_xlabel('X (m)', fontweight='bold', labelpad=8)
    ax.set_ylabel('Y (m)', fontweight='bold', labelpad=8)
    ax.set_zlabel('Z (m)', fontweight='bold', labelpad=8)
    ax.set_title('三维空域环境 (100×100×50 m)', fontweight='bold', pad=15)
    ax.view_init(elev=25, azim=-55)

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'environment_3d_view.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'已保存: {out}')


def plot_environment_top_view():
    """生成俯视图（XY平面投影）"""
    _apply_sci_style()
    fig, ax = plt.subplots(figsize=(8, 8))

    # 建筑物俯视投影
    for b in BUILDINGS:
        x0, y0, _, x1, y1, _ = b
        rect = mpatches.FancyBboxPatch(
            (x0, y0), x1-x0, y1-y0,
            boxstyle='round,pad=0.5', facecolor=SCI_SECONDARY,
            edgecolor=SCI_ACCENT, linewidth=1.0, alpha=0.6
        )
        ax.add_patch(rect)
        # 标注高度
        ax.text((x0+x1)/2, (y0+y1)/2, f'h={b[5]}m',
                ha='center', va='center', fontsize=7, color=SCI_ACCENT, fontweight='bold')

    # 圆柱体俯视投影
    for c in CYLINDERS:
        circle = plt.Circle((c[0], c[1]), c[2], facecolor=SCI_TERTIARY,
                             edgecolor=SCI_ACCENT, linewidth=1.0, alpha=0.6)
        ax.add_patch(circle)
        ax.text(c[0], c[1], f'h={c[3]}m', ha='center', va='center',
                fontsize=7, color=SCI_ACCENT, fontweight='bold')

    # 禁飞区
    nfz_patch = mpatches.Polygon(
        NO_FLY_VERTICES, closed=True, facecolor='#FF9800', edgecolor='#E65100',
        linewidth=1.5, alpha=0.3, label='禁飞区'
    )
    ax.add_patch(nfz_patch)

    # 配送站
    depots_arr = np.array(DEPOTS)
    ax.scatter(depots_arr[:, 0], depots_arr[:, 1], marker='s', s=150,
               c=SCI_ACCENT, edgecolors='white', linewidths=1.5,
               zorder=5, label='配送站')

    # 客户点
    cust_arr = np.array(CUSTOMERS)
    ax.scatter(cust_arr[:, 0], cust_arr[:, 1], marker='^', s=80,
               c='#2196F3', edgecolors='white', linewidths=0.8,
               zorder=5, label='客户点')

    # 无人机起点
    uav_arr = np.array(UAV_STARTS)
    ax.scatter(uav_arr[:, 0], uav_arr[:, 1], marker='D', s=120,
               c='#F44336', edgecolors='white', linewidths=1.5,
               zorder=6, label='UAV 起点')

    # 路径规划示意线（虚线）
    path_colors = ['#F44336', '#2196F3', '#FF9800']
    for i, uav in enumerate(UAV_STARTS):
        # 每架UAV到最近的两个客户点的示意路径
        targets_for_uav = [CUSTOMERS[i*3], CUSTOMERS[i*3+1]] if i*3+1 < len(CUSTOMERS) else [CUSTOMERS[i*3]]
        for t in targets_for_uav:
            ax.annotate('',
                        xy=(t[0], t[1]), xytext=(uav[0], uav[1]),
                        arrowprops=dict(arrowstyle='->', color=path_colors[i],
                                        lw=1.5, ls='--', alpha=0.6))

    # 边界
    ax.set_xlim(-2, SPACE_X + 2)
    ax.set_ylim(-2, SPACE_Y + 2)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)', fontweight='bold')
    ax.set_ylabel('Y (m)', fontweight='bold')
    ax.set_title('空域俯视图 (XY 平面投影)', fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3, linestyle='--')

    # 图例
    legend_handles = [
        mpatches.Patch(facecolor=SCI_SECONDARY, edgecolor=SCI_ACCENT, label='建筑物'),
        mpatches.Patch(facecolor='#FF9800', edgecolor='#E65100', alpha=0.3, label='禁飞区'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=SCI_ACCENT,
                   markersize=10, label='配送站'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='#2196F3',
                   markersize=10, label='客户点'),
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='#F44336',
                   markersize=10, label='UAV 起点'),
        plt.Line2D([0], [0], color=path_colors[0], lw=1.5, ls='--', label='路径示意'),
    ]
    ax.legend(handles=legend_handles, loc='upper left', fontsize=9,
              framealpha=0.9, edgecolor='#CCCCCC')

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'environment_top_view.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'已保存: {out}')


def plot_voxel_map():
    """生成体素地图和距离场可视化"""
    _apply_sci_style()

    resolution = 5  # 米
    nx = SPACE_X // resolution  # 20
    ny = SPACE_Y // resolution  # 20
    nz = SPACE_Z // resolution  # 10

    # 构建体素数据
    voxel_data = np.zeros((nx, ny, nz), dtype=np.int8)

    def world_to_voxel(x, y, z):
        return (min(int(x / resolution), nx-1),
                min(int(y / resolution), ny-1),
                min(int(z / resolution), nz-1))

    # 填充建筑物
    for b in BUILDINGS:
        i0, j0, k0 = world_to_voxel(b[0], b[1], b[2])
        i1, j1, k1 = world_to_voxel(b[3], b[4], b[5])
        voxel_data[i0:i1+1, j0:j1+1, k0:k1+1] = 1

    # 填充圆柱体
    for c in CYLINDERS:
        cx, cy, r, h = c
        for i in range(nx):
            for j in range(ny):
                x = (i + 0.5) * resolution
                y = (j + 0.5) * resolution
                if (x - cx)**2 + (y - cy)**2 <= r**2:
                    k_max = min(int(h / resolution), nz)
                    voxel_data[i, j, :k_max] = 1

    # 填充禁飞区
    from matplotlib.path import Path as MplPath
    nfz_path = MplPath(NO_FLY_VERTICES + [NO_FLY_VERTICES[0]])
    for i in range(nx):
        for j in range(ny):
            x = (i + 0.5) * resolution
            y = (j + 0.5) * resolution
            if nfz_path.contains_point((x, y)):
                voxel_data[i, j, :] = 2

    # 计算距离场（简化版：只在 z=2 层，即高度10m处做2D距离场）
    from scipy.ndimage import distance_transform_edt
    slice_z = 2  # z=10m
    free_slice = (voxel_data[:, :, slice_z] == 0).astype(np.float64)
    dist_field = distance_transform_edt(free_slice) * resolution

    fig = plt.figure(figsize=(14, 5))

    # --- 子图1: 体素3D散点图 ---
    ax1 = fig.add_subplot(131, projection='3d')
    # 障物体素
    obs_pts = np.argwhere(voxel_data == 1)
    if len(obs_pts) > 0:
        obs_world = obs_pts * resolution + resolution / 2
        ax1.scatter(obs_world[:, 0], obs_world[:, 1], obs_world[:, 2],
                    c=SCI_PRIMARY, marker='s', s=30, alpha=0.7, label='障碍物')
    # 禁飞区体素
    nfz_pts = np.argwhere(voxel_data == 2)
    if len(nfz_pts) > 0:
        nfz_world = nfz_pts * resolution + resolution / 2
        ax1.scatter(nfz_world[:, 0], nfz_world[:, 1], nfz_world[:, 2],
                    c='#FF9800', marker='s', s=20, alpha=0.5, label='禁飞区')

    ax1.set_xlim(0, SPACE_X)
    ax1.set_ylim(0, SPACE_Y)
    ax1.set_zlim(0, SPACE_Z)
    ax1.set_xlabel('X (m)', fontsize=8, fontweight='bold')
    ax1.set_ylabel('Y (m)', fontsize=8, fontweight='bold')
    ax1.set_zlabel('Z (m)', fontsize=8, fontweight='bold')
    ax1.set_title('(a) 体素标记', fontweight='bold', fontsize=10, pad=10)
    ax1.legend(fontsize=8, loc='upper left')
    ax1.view_init(elev=25, azim=-55)
    ax1.tick_params(labelsize=7)

    # --- 子图2: z=10m 切片的体素俯视图 ---
    ax2 = fig.add_subplot(132)
    slice_data = voxel_data[:, :, slice_z].T  # 转置使 Y 轴向上
    cmap = matplotlib.colors.ListedColormap(['#FFFFFF', SCI_PRIMARY, '#FF9800'])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    im = ax2.imshow(slice_data, origin='lower', cmap=cmap, norm=norm,
                    extent=[0, SPACE_X, 0, SPACE_Y], interpolation='nearest')
    ax2.set_xlabel('X (m)', fontweight='bold')
    ax2.set_ylabel('Y (m)', fontweight='bold')
    ax2.set_title(f'(b) z={slice_z*resolution}m 切片', fontweight='bold', fontsize=10, pad=10)
    # 标注起降点
    for d in DEPOTS:
        ax2.scatter(d[0], d[1], marker='s', s=60, c=SCI_ACCENT, edgecolors='white', linewidths=1, zorder=5)
    for ct in CUSTOMERS:
        ax2.scatter(ct[0], ct[1], marker='^', s=40, c='#2196F3', edgecolors='white', linewidths=0.5, zorder=5)
    legend_elements = [
        mpatches.Patch(facecolor=SCI_PRIMARY, label='障碍物'),
        mpatches.Patch(facecolor='#FF9800', label='禁飞区'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=SCI_ACCENT, markersize=8, label='配送站'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='#2196F3', markersize=8, label='客户点'),
    ]
    ax2.legend(handles=legend_elements, fontsize=7, loc='upper left')
    ax2.grid(True, alpha=0.2, linestyle='--')

    # --- 子图3: 距离场热力图 ---
    ax3 = fig.add_subplot(133)
    dist_display = np.ma.masked_where(voxel_data[:, :, slice_z].T != 0, dist_field.T)
    im3 = ax3.imshow(dist_display, origin='lower', cmap='YlGn',
                     extent=[0, SPACE_X, 0, SPACE_Y], interpolation='bilinear')
    # 障物区域覆盖灰色
    obs_overlay = np.ma.masked_where(voxel_data[:, :, slice_z].T == 0, voxel_data[:, :, slice_z].T)
    ax3.imshow(obs_overlay, origin='lower', cmap=matplotlib.colors.ListedColormap(['#555555']),
               extent=[0, SPACE_X, 0, SPACE_Y], interpolation='nearest', alpha=0.8)
    cbar = fig.colorbar(im3, ax=ax3, shrink=0.8, pad=0.02)
    cbar.set_label('距障碍物距离 (m)', fontsize=9, fontweight='bold')
    ax3.set_xlabel('X (m)', fontweight='bold')
    ax3.set_ylabel('Y (m)', fontweight='bold')
    ax3.set_title(f'(c) 距离场 (z={slice_z*resolution}m)', fontweight='bold', fontsize=10, pad=10)
    ax3.grid(True, alpha=0.2, linestyle='--')

    fig.suptitle('体素地图 (分辨率=5m)', fontweight='bold', fontsize=13, y=1.02)
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'environment_voxel_map.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'已保存: {out}')


def plot_scenarios():
    """生成场景配置对比图"""
    _apply_sci_style()

    names = list(SCENARIOS.keys())
    labels = [SCENARIOS[k]['label'] for k in names]
    uavs = [SCENARIOS[k]['uavs'] for k in names]
    targets = [SCENARIOS[k]['targets'] for k in names]
    obstacles = [SCENARIOS[k]['obstacles'] for k in names]

    x = np.arange(len(names))
    width = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- 子图1: 分组柱状图 ---
    ax1 = axes[0]
    bars1 = ax1.bar(x - width, uavs,      width, color=SCI_PRIMARY,  edgecolor='white', linewidth=0.5, label='UAV 数量')
    bars2 = ax1.bar(x,         targets,    width, color='#2196F3',    edgecolor='white', linewidth=0.5, label='目标数量')
    bars3 = ax1.bar(x + width, obstacles,  width, color='#FF9800',    edgecolor='white', linewidth=0.5, label='障碍物数量')

    # 数值标签
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., h + 0.3,
                     f'{int(h)}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel('数量', fontweight='bold')
    ax1.set_title('(a) 场景参数对比', fontweight='bold', pad=10)
    ax1.legend(fontsize=9, loc='upper left', framealpha=0.9, edgecolor='#CCCCCC')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_ylim(0, max(max(uavs), max(targets), max(obstacles)) + 3)

    # --- 子图2: 雷达图 ---
    ax2 = axes[1]
    categories = ['UAV 数量', '目标数量', '障碍物数量']
    N = len(categories)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    ax2 = fig.add_subplot(122, polar=True)

    for i, name in enumerate(names):
        values = [uavs[i], targets[i], obstacles[i]]
        values += values[:1]
        color = SCI_SERIES[i % len(SCI_SERIES)]
        ax2.plot(angles, values, 'o-', color=color, linewidth=1.8, markersize=5,
                 label=name, alpha=0.85)
        ax2.fill(angles, values, color=color, alpha=0.1)

    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(categories, fontsize=9, fontweight='bold')
    ax2.set_ylim(0, 25)
    ax2.set_yticks([5, 10, 15, 20, 25])
    ax2.set_yticklabels(['5', '10', '15', '20', '25'], fontsize=7)
    ax2.set_title('(b) 场景参数雷达图', fontweight='bold', pad=20, fontsize=11)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=9,
               framealpha=0.9, edgecolor='#CCCCCC')
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'environment_scenarios.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'已保存: {out}')


# ==================== 主入口 ====================

if __name__ == '__main__':
    print('=' * 50)
    print('三维空域环境可视化')
    print('=' * 50)

    plot_environment_3d()
    plot_environment_top_view()
    plot_voxel_map()
    plot_scenarios()

    print('=' * 50)
    print('全部图片生成完成。')
    print(f'输出目录: {OUTPUT_DIR}')
