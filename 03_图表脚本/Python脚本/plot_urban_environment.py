"""
城市空域环境可视化脚本
项目：基于深度强化学习的多无人机协同物流配送路径规划

输出图片：
  - urban_environment_3d.png      三维城市环境图
  - urban_environment_top.png     俯视图（含路径）
  - urban_environment_scenarios.png  场景配置对比图
  - urban_environment_legend.png  图例说明
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import os

# ==================== SCI 风格常量 ====================

SCI_PRIMARY   = '#4CAF50'
SCI_SECONDARY = '#81C784'
SCI_TERTIARY  = '#A5D6A7'
SCI_ACCENT    = '#2E7D32'
SCI_LIGHT     = '#E8F5E9'

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', '04_脚本图片输出', '论文插图'
)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _apply_sci_style():
    """应用 SCI 论文风格"""
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


# ==================== 城市环境数据 ====================

# 空域尺寸 (m)
SPACE_X, SPACE_Y, SPACE_Z = 2000, 2000, 200

# 建筑物: (中心x, 中心y, 宽度, 深度, 高度, 颜色, 名称)
BUILDINGS = [
    (200, 200, 80, 60, 45, '#E53935', '商业中心'),
    (500, 300, 100, 80, 60, '#1E88E5', '写字楼A'),
    (800, 200, 70, 50, 35, '#43A047', '住宅楼'),
    (300, 600, 90, 70, 55, '#FB8C00', '购物中心'),
    (600, 500, 110, 90, 70, '#8E24AA', '写字楼B'),
    (900, 700, 85, 65, 50, '#00ACC1', '酒店'),
    (400, 900, 95, 75, 65, '#F4511E', '医院'),
    (700, 800, 75, 55, 40, '#3949AB', '学校'),
    (1000, 400, 120, 100, 80, '#7B1FA2', '综合体'),
    (150, 800, 60, 45, 30, '#00897B', '图书馆'),
]

# 河流: (y位置, 宽度)
RIVER_Y = 1000
RIVER_WIDTH = 60

# 桥梁: (中心x, 中心y, 宽度, 长度)
BRIDGES = [
    (400, RIVER_Y, 40, 80),
    (1200, RIVER_Y, 40, 80),
]

# 高压线: (起点x, 起点y, 终点x, 终点y, 高度)
POWER_LINES = [
    (300, 500, 300, 700, 60),
    (800, 300, 800, 800, 60),
    (1200, 400, 1200, 900, 60),
]

# 树木区域: (中心x, 中心y, 宽度, 深度, 高度)
TREE_ZONES = [
    (150, 400, 100, 100, 15),
    (1500, 300, 120, 120, 15),
    (250, 1200, 80, 80, 15),
    (1600, 1400, 100, 100, 15),
]

# 动态飞鸟区: (中心x, 中心y, 半径, 高度)
BIRD_ZONE = (1000, 600, 100, 100)

# 禁飞区: (中心x, 中心y, 宽度, 深度, 名称)
NO_FLY_ZONES = [
    (1800, 1800, 200, 200, '政府机关'),
    (100, 1800, 150, 150, '军事设施'),
]

# GPS干扰区: (中心x, 中心y, 半径, 高度)
GPS_INTERFERENCE = (500, 1500, 150, 160)

# 风雨天气区: (中心x, 中心y, 半径, 高度)
WEATHER_ZONE = (1400, 800, 200, 200)

# 配送点: (编号, 中心x, 中心y, 高度, 是配送中心)
DELIVERY_POINTS = [
    (1, 100, 100, 10, True),    # 配送中心
    (2, 300, 250, 10, False),
    (3, 600, 150, 10, False),   # 特殊点
    (4, 900, 300, 10, False),
    (5, 1200, 200, 10, False),  # 特殊点
    (6, 400, 500, 10, False),
    (7, 700, 600, 10, False),
    (8, 1000, 500, 10, False),
    (9, 1300, 400, 10, False),
    (10, 500, 800, 10, False),
    (11, 800, 900, 10, False),
    (12, 1100, 700, 10, False), # 特殊点
]


# ==================== 绘图函数 ====================

def draw_box(ax, cx, cy, w, d, h, color, alpha=0.6, label=None):
    """绘制长方体"""
    x = [cx - w/2, cx + w/2]
    y = [cy - d/2, cy + d/2]
    z = [0, h]

    # 6个面
    verts = [
        # 底面
        [[x[0], y[0], z[0]], [x[1], y[0], z[0]], [x[1], y[1], z[0]], [x[0], y[1], z[0]]],
        # 顶面
        [[x[0], y[0], z[1]], [x[1], y[0], z[1]], [x[1], y[1], z[1]], [x[0], y[1], z[1]]],
        # 前面
        [[x[0], y[0], z[0]], [x[1], y[0], z[0]], [x[1], y[0], z[1]], [x[0], y[0], z[1]]],
        # 后面
        [[x[0], y[1], z[0]], [x[1], y[1], z[0]], [x[1], y[1], z[1]], [x[0], y[1], z[1]]],
        # 左面
        [[x[0], y[0], z[0]], [x[0], y[1], z[0]], [x[0], y[1], z[1]], [x[0], y[0], z[1]]],
        # 右面
        [[x[1], y[0], z[0]], [x[1], y[1], z[0]], [x[1], y[1], z[1]], [x[1], y[0], z[1]]],
    ]

    faces = Poly3DCollection(verts, alpha=alpha, facecolor=color, edgecolor='#333333', linewidth=0.5)
    ax.add_collection3d(faces)

    # 添加高度标注
    ax.text(cx, cy, h + 5, f'{h}m', ha='center', va='bottom', fontsize=6, color='#333333')


def draw_cylinder(ax, cx, cy, r, h, color, alpha=0.6):
    """绘制圆柱体"""
    theta = np.linspace(0, 2*np.pi, 20)
    x = cx + r * np.cos(theta)
    y = cy + r * np.sin(theta)

    # 底面
    verts_bottom = [list(zip(x, y, [0]*len(x)))]
    faces_bottom = Poly3DCollection(verts_bottom, alpha=alpha, facecolor=color, edgecolor='#333333', linewidth=0.5)
    ax.add_collection3d(faces_bottom)

    # 顶面
    verts_top = [list(zip(x, y, [h]*len(x)))]
    faces_top = Poly3DCollection(verts_top, alpha=alpha, facecolor=color, edgecolor='#333333', linewidth=0.5)
    ax.add_collection3d(faces_top)

    # 侧面
    for i in range(len(x) - 1):
        verts_side = [[
            [x[i], y[i], 0],
            [x[i+1], y[i+1], 0],
            [x[i+1], y[i+1], h],
            [x[i], y[i], h]
        ]]
        faces_side = Poly3DCollection(verts_side, alpha=alpha*0.8, facecolor=color, edgecolor='#333333', linewidth=0.3)
        ax.add_collection3d(faces_side)


def draw_zone_circle(ax, cx, cy, r, h, color, alpha=0.3, linestyle='--'):
    """绘制圆形区域（禁飞区、干扰区等）"""
    theta = np.linspace(0, 2*np.pi, 30)
    x = cx + r * np.cos(theta)
    y = cy + r * np.sin(theta)

    # 底面圆
    ax.plot(x, y, [0]*len(x), color=color, linewidth=2, linestyle=linestyle, alpha=0.8)

    # 顶面圆
    ax.plot(x, y, [h]*len(x), color=color, linewidth=1.5, linestyle=linestyle, alpha=0.6)

    # 垂直线
    for i in range(0, len(x), 5):
        ax.plot([x[i], x[i]], [y[i], y[i]], [0, h], color=color, linewidth=0.5, alpha=0.4)

    # 填充底面
    verts = [list(zip(x, y, [0]*len(x)))]
    faces = Poly3DCollection(verts, alpha=alpha, facecolor=color, edgecolor='none')
    ax.add_collection3d(faces)


def plot_3d_urban_environment():
    """绘制三维城市环境"""
    _apply_sci_style()

    fig = plt.figure(figsize=(16, 12), dpi=150)
    ax = fig.add_subplot(111, projection='3d')

    # 设置坐标轴范围
    ax.set_xlim(0, SPACE_X)
    ax.set_ylim(0, SPACE_Y)
    ax.set_zlim(0, SPACE_Z)

    # ==================== 绘制建筑物 ====================
    for bx, by, bw, bd, bh, color, name in BUILDINGS:
        draw_box(ax, bx, by, bw, bd, bh, color, alpha=0.7)
        ax.text(bx, by, 0, name, ha='center', va='top', fontsize=5, color='#333333', zorder=10)

    # ==================== 绘制河流 ====================
    river_x = [0, SPACE_X, SPACE_X, 0]
    river_y = [RIVER_Y - RIVER_WIDTH/2, RIVER_Y - RIVER_WIDTH/2,
                RIVER_Y + RIVER_WIDTH/2, RIVER_Y + RIVER_WIDTH/2]
    river_z = [0, 0, 0, 0]
    verts_river = [list(zip(river_x, river_y, river_z))]
    faces_river = Poly3DCollection(verts_river, alpha=0.5, facecolor='#2196F3', edgecolor='#1565C0', linewidth=1)
    ax.add_collection3d(faces_river)
    ax.text(1000, RIVER_Y, 5, '河流', ha='center', va='center', fontsize=8, color='#1565C0', fontweight='bold')

    # ==================== 绘制桥梁 ====================
    for bx, by, bw, bd in BRIDGES:
        draw_box(ax, bx, by, bw, bd, 20, '#795548', alpha=0.8)
        ax.text(bx, by, 25, '桥梁', ha='center', va='bottom', fontsize=6, color='#795548')

    # ==================== 绘制高压线塔 ====================
    for x1, y1, x2, y2, h in POWER_LINES:
        # 线塔
        draw_cylinder(ax, x1, y1, 5, h, '#FFD600', alpha=0.8)
        draw_cylinder(ax, x2, y2, 5, h, '#FFD600', alpha=0.8)
        # 连接线
        ax.plot([x1, x2], [y1, y2], [h, h], color='#FFD600', linewidth=2, alpha=0.6)

    # ==================== 绘制树木区域 ====================
    for tx, ty, tw, td, th in TREE_ZONES:
        draw_box(ax, tx, ty, tw, td, th, '#2E7D32', alpha=0.5)

    # ==================== 绘制动态飞鸟区 ====================
    bx, by, br, bh = BIRD_ZONE
    draw_zone_circle(ax, bx, by, br, bh, '#FF7043', alpha=0.2)
    ax.text(bx, by, bh + 10, '飞鸟区', ha='center', va='bottom', fontsize=7, color='#FF7043')

    # ==================== 绘制禁飞区 ====================
    for nx, ny, nw, nd, name in NO_FLY_ZONES:
        draw_zone_circle(ax, nx, ny, max(nw, nd)/2, SPACE_Z, '#D32F2F', alpha=0.15)
        ax.text(nx, ny, SPACE_Z + 5, f'禁飞区\n{name}', ha='center', va='bottom',
                fontsize=7, color='#D32F2F', fontweight='bold')

    # ==================== 绘制GPS干扰区 ====================
    gx, gy, gr, gh = GPS_INTERFERENCE
    draw_zone_circle(ax, gx, gy, gr, gh, '#FF9800', alpha=0.15, linestyle='-.')
    ax.text(gx, gy, gh + 10, 'GPS干扰区', ha='center', va='bottom', fontsize=7, color='#FF9800')

    # ==================== 绘制风雨天气区 ====================
    wx, wy, wr, wh = WEATHER_ZONE
    draw_zone_circle(ax, wx, wy, wr, wh, '#78909C', alpha=0.1, linestyle=':')
    ax.text(wx, wy, wh + 10, '风雨区', ha='center', va='bottom', fontsize=7, color='#78909C')

    # ==================== 绘制配送点 ====================
    for pid, px, py, pz, is_depot in DELIVERY_POINTS:
        if is_depot:
            ax.scatter(px, py, pz, s=300, c=SCI_PRIMARY, marker='s',
                      edgecolors='white', linewidth=3, zorder=100)
            ax.text(px, py, pz + 15, f'配送中心', ha='center', va='bottom',
                    fontsize=8, color=SCI_PRIMARY, fontweight='bold')
        else:
            ax.scatter(px, py, pz, s=150, c='#E53935', marker='^',
                      edgecolors='white', linewidth=2, zorder=100)
            ax.text(px, py, pz + 10, f'点{pid}', ha='center', va='bottom',
                    fontsize=7, color='#E53935')

    # ==================== 绘制示例路径 ====================
    # 生成一条示例路径
    path_points = [
        (100, 100, 50),   # 配送中心
        (300, 250, 50),   # 点2
        (600, 150, 50),   # 点3
        (900, 300, 50),   # 点4
        (400, 500, 50),   # 点6
        (700, 600, 50),   # 点7
        (100, 100, 50),   # 返回配送中心
    ]

    # 路径颜色分级（绿色=安全，黄色=低风险，橙色=中风险，红色=高风险）
    path_colors = ['#4CAF50', '#4CAF50', '#FF9800', '#FF9800', '#4CAF50', '#4CAF50']

    for i in range(len(path_points) - 1):
        p1 = path_points[i]
        p2 = path_points[i + 1]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                color=path_colors[i], linewidth=2.5, alpha=0.8)

    # ==================== 设置标签 ====================
    ax.set_xlabel('X (m)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('Y (m)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_zlabel('高度 (m)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title('城市空域环境三维视图', fontsize=16, fontweight='bold', pad=20)

    # 设置视角
    ax.view_init(elev=25, azim=45)

    # 添加图例
    legend_items = [
        mpatches.Patch(color='#E53935', label='建筑物'),
        mpatches.Patch(color='#2196F3', label='河流'),
        mpatches.Patch(color='#795548', label='桥梁'),
        mpatches.Patch(color='#FFD600', label='高压线'),
        mpatches.Patch(color='#2E7D32', label='树木区'),
        mpatches.Patch(color='#FF7043', label='飞鸟区'),
        mpatches.Patch(color='#D32F2F', label='禁飞区'),
        mpatches.Patch(color='#FF9800', label='GPS干扰区'),
        mpatches.Patch(color='#78909C', label='风雨区'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=SCI_PRIMARY,
                   markersize=12, label='配送中心'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='#E53935',
                   markersize=12, label='配送点'),
        plt.Line2D([0], [0], color='#4CAF50', linewidth=2.5, label='安全路径'),
        plt.Line2D([0], [0], color='#FF9800', linewidth=2.5, label='风险路径'),
    ]
    ax.legend(handles=legend_items, loc='upper left', fontsize=8, frameon=True, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'urban_environment_3d.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("保存: urban_environment_3d.png", flush=True)


def plot_top_view():
    """绘制俯视图"""
    _apply_sci_style()

    fig, ax = plt.subplots(figsize=(14, 14), dpi=150)

    # 设置坐标轴
    ax.set_xlim(0, SPACE_X)
    ax.set_ylim(0, SPACE_Y)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2, linestyle='--')

    # ==================== 绘制建筑物 ====================
    for bx, by, bw, bd, bh, color, name in BUILDINGS:
        rect = plt.Rectangle((bx - bw/2, by - bd/2), bw, bd,
                            facecolor=color, edgecolor='#333333', linewidth=1, alpha=0.7)
        ax.add_patch(rect)
        ax.text(bx, by, f'{name}\n{bh}m', ha='center', va='center', fontsize=6, color='white', fontweight='bold')

    # ==================== 绘制河流 ====================
    river_rect = plt.Rectangle((0, RIVER_Y - RIVER_WIDTH/2), SPACE_X, RIVER_WIDTH,
                              facecolor='#2196F3', edgecolor='#1565C0', linewidth=1, alpha=0.5)
    ax.add_patch(river_rect)
    ax.text(SPACE_X/2, RIVER_Y, '河流', ha='center', va='center', fontsize=10, color='#1565C0', fontweight='bold')

    # ==================== 绘制桥梁 ====================
    for bx, by, bw, bd in BRIDGES:
        rect = plt.Rectangle((bx - bw/2, by - bd/2), bw, bd,
                            facecolor='#795548', edgecolor='#5D4037', linewidth=2, alpha=0.9)
        ax.add_patch(rect)
        ax.text(bx, by, '桥梁', ha='center', va='center', fontsize=7, color='white', fontweight='bold')

    # ==================== 绘制高压线 ====================
    for x1, y1, x2, y2, h in POWER_LINES:
        ax.plot([x1, x2], [y1, y2], color='#FFD600', linewidth=3, alpha=0.7, linestyle='-')
        ax.scatter([x1, x2], [y1, y2], s=100, c='#FFD600', marker='s', edgecolors='#F57F17', linewidth=1, zorder=5)

    # ==================== 绘制树木区域 ====================
    for tx, ty, tw, td, th in TREE_ZONES:
        rect = plt.Rectangle((tx - tw/2, ty - td/2), tw, td,
                            facecolor='#2E7D32', edgecolor='#1B5E20', linewidth=1, alpha=0.5)
        ax.add_patch(rect)
        ax.text(tx, ty, f'树木区\n{th}m', ha='center', va='center', fontsize=6, color='white')

    # ==================== 绘制特殊区域 ====================
    # 飞鸟区
    bx, by, br, bh = BIRD_ZONE
    circle = plt.Circle((bx, by), br, facecolor='#FF7043', edgecolor='#E64A19',
                        linewidth=2, alpha=0.2, linestyle='--')
    ax.add_patch(circle)
    ax.text(bx, by, f'飞鸟区\n{bh}m', ha='center', va='center', fontsize=7, color='#E64A19')

    # 禁飞区
    for nx, ny, nw, nd, name in NO_FLY_ZONES:
        rect = plt.Rectangle((nx - nw/2, ny - nd/2), nw, nd,
                            facecolor='#D32F2F', edgecolor='#B71C1C', linewidth=2, alpha=0.2, linestyle='--')
        ax.add_patch(rect)
        ax.text(nx, ny, f'禁飞区\n{name}', ha='center', va='center', fontsize=7, color='#B71C1C', fontweight='bold')

    # GPS干扰区
    gx, gy, gr, gh = GPS_INTERFERENCE
    circle = plt.Circle((gx, gy), gr, facecolor='#FF9800', edgecolor='#E65100',
                        linewidth=2, alpha=0.15, linestyle='-.')
    ax.add_patch(circle)
    ax.text(gx, gy, f'GPS干扰区\n{gh}m', ha='center', va='center', fontsize=7, color='#E65100')

    # 风雨区
    wx, wy, wr, wh = WEATHER_ZONE
    circle = plt.Circle((wx, wy), wr, facecolor='#78909C', edgecolor='#546E7A',
                        linewidth=2, alpha=0.1, linestyle=':')
    ax.add_patch(circle)
    ax.text(wx, wy, f'风雨区\n{wh}m', ha='center', va='center', fontsize=7, color='#546E7A')

    # ==================== 绘制配送点 ====================
    for pid, px, py, pz, is_depot in DELIVERY_POINTS:
        if is_depot:
            ax.scatter(px, py, s=400, c=SCI_PRIMARY, marker='s',
                      edgecolors='white', linewidth=4, zorder=100)
            ax.text(px, py - 30, '配送中心', ha='center', va='top',
                    fontsize=9, color=SCI_PRIMARY, fontweight='bold')
        else:
            ax.scatter(px, py, s=200, c='#E53935', marker='^',
                      edgecolors='white', linewidth=2, zorder=100)
            ax.text(px, py + 20, f'点{pid}', ha='center', va='bottom',
                    fontsize=8, color='#E53935', fontweight='bold')

    # ==================== 绘制示例路径 ====================
    path_points = [
        (100, 100), (300, 250), (600, 150), (900, 300),
        (400, 500), (700, 600), (100, 100)
    ]

    path_colors = ['#4CAF50', '#4CAF50', '#FF9800', '#FF9800', '#4CAF50', '#4CAF50']
    risk_labels = ['安全', '安全', '中风险', '中风险', '安全', '安全']

    for i in range(len(path_points) - 1):
        p1 = path_points[i]
        p2 = path_points[i + 1]
        ax.annotate('', xy=p2, xytext=p1,
                   arrowprops=dict(arrowstyle='->', color=path_colors[i], lw=2.5, alpha=0.8))

    # ==================== 设置标签 ====================
    ax.set_xlabel('X (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
    ax.set_title('城市空域环境俯视图（含路径规划）', fontsize=16, fontweight='bold')

    # 添加图例
    legend_items = [
        mpatches.Patch(color='#E53935', label='建筑物', alpha=0.7),
        mpatches.Patch(color='#2196F3', label='河流', alpha=0.5),
        mpatches.Patch(color='#795548', label='桥梁'),
        plt.Line2D([0], [0], color='#FFD600', linewidth=3, label='高压线'),
        mpatches.Patch(color='#2E7D32', label='树木区', alpha=0.5),
        mpatches.Patch(color='#FF7043', label='飞鸟区', alpha=0.3),
        mpatches.Patch(color='#D32F2F', label='禁飞区', alpha=0.3),
        mpatches.Patch(color='#FF9800', label='GPS干扰区', alpha=0.3),
        mpatches.Patch(color='#78909C', label='风雨区', alpha=0.3),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=SCI_PRIMARY,
                   markersize=14, label='配送中心'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='#E53935',
                   markersize=14, label='配送点'),
        plt.Line2D([0], [0], color='#4CAF50', linewidth=2.5, label='安全路径'),
        plt.Line2D([0], [0], color='#FF9800', linewidth=2.5, label='风险路径'),
    ]
    ax.legend(handles=legend_items, loc='upper left', fontsize=9, frameon=True, framealpha=0.9, ncol=2)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'urban_environment_top.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("保存: urban_environment_top.png", flush=True)


def plot_scenario_comparison():
    """绘制场景配置对比图"""
    _apply_sci_style()

    scenarios = [
        {'name': 'S1: 基础场景', 'uavs': 1, 'targets': 1, 'obstacles': '简单'},
        {'name': 'S2: 简单障碍', 'uavs': 1, 'targets': 1, 'obstacles': '3个'},
        {'name': 'S3: 多机协同', 'uavs': 3, 'targets': 1, 'obstacles': '5个'},
        {'name': 'S4: 完整任务', 'uavs': 3, 'targets': 12, 'obstacles': '全部'},
        {'name': 'S5: 动态障碍', 'uavs': 3, 'targets': 12, 'obstacles': '全部+动态'},
        {'name': 'S6: 恶劣天气', 'uavs': 3, 'targets': 12, 'obstacles': '全部+天气'},
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=150)

    for idx, scenario in enumerate(scenarios):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]

        ax.set_xlim(0, SPACE_X)
        ax.set_ylim(0, SPACE_Y)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.15, linestyle='--')

        # 简化建筑物
        for bx, by, bw, bd, bh, color, name in BUILDINGS[:4]:
            rect = plt.Rectangle((bx - bw/2, by - bd/2), bw, bd,
                                facecolor=color, edgecolor='#333333', linewidth=0.5, alpha=0.6)
            ax.add_patch(rect)

        # 河流
        river_rect = plt.Rectangle((0, RIVER_Y - RIVER_WIDTH/2), SPACE_X, RIVER_WIDTH,
                                  facecolor='#2196F3', edgecolor='none', alpha=0.3)
        ax.add_patch(river_rect)

        # 禁飞区
        for nx, ny, nw, nd, name in NO_FLY_ZONES:
            rect = plt.Rectangle((nx - nw/2, ny - nd/2), nw, nd,
                                facecolor='#D32F2F', edgecolor='none', alpha=0.1)
            ax.add_patch(rect)

        # 配送中心
        ax.scatter(100, 100, s=200, c=SCI_PRIMARY, marker='s', edgecolors='white', linewidth=2, zorder=5)

        # 配送点
        num_targets = scenario['targets']
        for i in range(min(num_targets, len(DELIVERY_POINTS) - 1)):
            pid, px, py, pz, _ = DELIVERY_POINTS[i + 1]
            ax.scatter(px, py, s=100, c='#E53935', marker='^', edgecolors='white', linewidth=1, zorder=5)

        # UAV
        num_uavs = scenario['uavs']
        for i in range(num_uavs):
            offset = i * 30
            ax.scatter(100 + offset, 100, s=150, c=SCI_PRIMARY, marker='o', edgecolors='white', linewidth=2, zorder=6)

        ax.set_title(f'{scenario["name"]}\n{scenario["uavs"]}UAV, {scenario["targets"]}目标, {scenario["obstacles"]}障碍',
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('X (m)', fontsize=9)
        ax.set_ylabel('Y (m)', fontsize=9)

    plt.suptitle('实验场景配置对比', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'urban_environment_scenarios.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("保存: urban_environment_scenarios.png", flush=True)


# ==================== 主函数 ====================

if __name__ == '__main__':
    print("=" * 60, flush=True)
    print("城市空域环境可视化", flush=True)
    print("=" * 60, flush=True)

    print("\n1. 生成三维环境图...", flush=True)
    plot_3d_urban_environment()

    print("\n2. 生成俯视图...", flush=True)
    plot_top_view()

    print("\n3. 生成场景对比图...", flush=True)
    plot_scenario_comparison()

    print("\n" + "=" * 60, flush=True)
    print("所有图表生成完成!", flush=True)
    print(f"输出目录: {OUTPUT_DIR}", flush=True)
    print("=" * 60, flush=True)
