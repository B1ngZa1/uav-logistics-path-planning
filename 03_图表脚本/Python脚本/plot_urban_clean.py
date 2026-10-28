"""
城市空域环境纯净版可视化（无路径，突出建筑物）
项目：基于深度强化学习的多无人机协同物流配送路径规划
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import os

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', '04_脚本图片输出', '论文插图'
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

SPACE_X, SPACE_Y, SPACE_Z = 2000, 2000, 200


def _apply_style():
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.pad_inches'] = 0.15


# ==================== 环境数据 ====================

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

# 道路（降低宽度）
ROADS = [
    (0, 500, 2000, 500, 12, 'main'),
    (0, 1000, 2000, 1000, 12, 'main'),
    (0, 1500, 2000, 1500, 10, 'main'),
    (500, 0, 500, 2000, 12, 'main'),
    (1000, 0, 1000, 2000, 12, 'main'),
    (1500, 0, 1500, 2000, 10, 'main'),
    (0, 250, 2000, 250, 6, 'secondary'),
    (0, 750, 2000, 750, 6, 'secondary'),
    (0, 1250, 2000, 1250, 6, 'secondary'),
    (0, 1750, 2000, 1750, 6, 'secondary'),
    (250, 0, 250, 2000, 6, 'secondary'),
    (750, 0, 750, 2000, 6, 'secondary'),
    (1250, 0, 1250, 2000, 6, 'secondary'),
    (1750, 0, 1750, 2000, 6, 'secondary'),
]

RIVER_Y = 1000
RIVER_WIDTH = 80

BRIDGES = [
    (500, RIVER_Y, 50, 100),
    (1000, RIVER_Y, 50, 100),
    (1500, RIVER_Y, 50, 100),
]

POWER_LINES = [
    (300, 500, 300, 700, 60),
    (800, 300, 800, 800, 60),
    (1200, 400, 1200, 900, 60),
]

TREE_ZONES = [
    (150, 400, 120, 120, 15),
    (1500, 300, 140, 140, 15),
    (250, 1200, 100, 100, 15),
    (1600, 1400, 120, 120, 15),
]

BIRD_ZONE = (1000, 600, 120, 100)

NO_FLY_ZONES = [
    (1800, 1800, 250, 250, '政府机关'),
    (100, 1800, 200, 200, '军事设施'),
]

GPS_INTERFERENCE = (500, 1500, 180, 160)
WEATHER_ZONE = (1400, 800, 250, 200)

DELIVERY_POINTS = [
    (1, 100, 100, 10, True),
    (2, 300, 250, 10, False),
    (3, 600, 150, 10, False),
    (4, 900, 300, 10, False),
    (5, 1200, 200, 10, False),
    (6, 400, 500, 10, False),
    (7, 700, 600, 10, False),
    (8, 1000, 500, 10, False),
    (9, 1300, 400, 10, False),
    (10, 500, 800, 10, False),
    (11, 800, 900, 10, False),
    (12, 1100, 700, 10, False),
]


# ==================== 绘图函数 ====================

def draw_box(ax, cx, cy, w, d, h, color, alpha=0.9):
    """绘制长方体（建筑物）"""
    x = [cx - w/2, cx + w/2]
    y = [cy - d/2, cy + d/2]
    z = [0, h]

    verts = [
        [[x[0], y[0], z[0]], [x[1], y[0], z[0]], [x[1], y[1], z[0]], [x[0], y[1], z[0]]],
        [[x[0], y[0], z[1]], [x[1], y[0], z[1]], [x[1], y[1], z[1]], [x[0], y[1], z[1]]],
        [[x[0], y[0], z[0]], [x[1], y[0], z[0]], [x[1], y[0], z[1]], [x[0], y[0], z[1]]],
        [[x[0], y[1], z[0]], [x[1], y[1], z[0]], [x[1], y[1], z[1]], [x[0], y[1], z[1]]],
        [[x[0], y[0], z[0]], [x[0], y[1], z[0]], [x[0], y[1], z[1]], [x[0], y[0], z[1]]],
        [[x[1], y[0], z[0]], [x[1], y[1], z[0]], [x[1], y[1], z[1]], [x[1], y[0], z[1]]],
    ]

    faces = Poly3DCollection(verts, alpha=alpha, facecolor=color, edgecolor='#212121', linewidth=1.2)
    ax.add_collection3d(faces)

    # 高度标注
    ax.text(cx, cy, h + 10, f'{h}m', ha='center', va='bottom', fontsize=8,
            color='#212121', fontweight='bold',
            path_effects=[pe.withStroke(linewidth=2, foreground='white')])


def draw_road(ax, x1, y1, x2, y2, width, road_type):
    """绘制道路（低透明度）"""
    w = width / 2
    color = '#FFD54F' if road_type == 'main' else '#FFE082'
    alpha = 0.25 if road_type == 'main' else 0.15  # 降低透明度

    dx, dy = x2 - x1, y2 - y1
    length = np.sqrt(dx**2 + dy**2)
    if length == 0:
        return

    nx = -dy / length * w
    ny = dx / length * w

    verts = [[[x1 + nx, y1 + ny, 0], [x2 + nx, y2 + ny, 0],
              [x2 - nx, y2 - ny, 0], [x1 - nx, y1 - ny, 0]]]

    faces = Poly3DCollection(verts, alpha=alpha, facecolor=color, edgecolor='none')
    ax.add_collection3d(faces)


def draw_bridge(ax, cx, cy, w, h, elevation=15):
    """绘制桥梁"""
    x = [cx - w/2, cx + w/2]
    y = [cy - h/2, cy + h/2]
    z = [elevation, elevation + 15]

    verts = [
        [[x[0], y[0], z[0]], [x[1], y[0], z[0]], [x[1], y[1], z[0]], [x[0], y[1], z[0]]],
        [[x[0], y[0], z[1]], [x[1], y[0], z[1]], [x[1], y[1], z[1]], [x[0], y[1], z[1]]],
        [[x[0], y[0], z[0]], [x[1], y[0], z[0]], [x[1], y[0], z[1]], [x[0], y[0], z[1]]],
        [[x[0], y[1], z[0]], [x[1], y[1], z[0]], [x[1], y[1], z[1]], [x[0], y[1], z[1]]],
        [[x[0], y[0], z[0]], [x[0], y[1], z[0]], [x[0], y[1], z[1]], [x[0], y[0], z[1]]],
        [[x[1], y[0], z[0]], [x[1], y[1], z[0]], [x[1], y[1], z[1]], [x[1], y[0], z[1]]],
    ]

    faces = Poly3DCollection(verts, alpha=0.95, facecolor='#8D6E63', edgecolor='#4E342E', linewidth=1.5)
    ax.add_collection3d(faces)

    ax.text(cx, cy, z[1] + 8, '桥梁', ha='center', va='bottom', fontsize=9,
            color='#4E342E', fontweight='bold',
            path_effects=[pe.withStroke(linewidth=2, foreground='white')])


def draw_zone(ax, cx, cy, r, h, color, alpha=0.15, label=None):
    """绘制区域"""
    theta = np.linspace(0, 2*np.pi, 40)
    x = cx + r * np.cos(theta)
    y = cy + r * np.sin(theta)

    verts = [list(zip(x, y, [0]*len(x)))]
    faces = Poly3DCollection(verts, alpha=alpha, facecolor=color, edgecolor='none')
    ax.add_collection3d(faces)

    ax.plot(x, y, [0]*len(x), color=color, linewidth=2.5, alpha=0.7)
    ax.plot(x, y, [h]*len(x), color=color, linewidth=1.5, alpha=0.4)

    for i in range(0, len(x), 8):
        ax.plot([x[i], x[i]], [y[i], y[i]], [0, h], color=color, linewidth=0.8, alpha=0.2)

    if label:
        ax.text(cx, cy, h + 15, label, ha='center', va='bottom', fontsize=9,
                color=color, fontweight='bold',
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])


def plot_3d_clean():
    """绘制纯净版3D环境图（无路径）"""
    _apply_style()

    fig = plt.figure(figsize=(18, 14), dpi=150)
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlim(0, SPACE_X)
    ax.set_ylim(0, SPACE_Y)
    ax.set_zlim(0, SPACE_Z)

    # 先绘制道路（低透明度，作为背景）
    for x1, y1, x2, y2, w, rtype in ROADS:
        draw_road(ax, x1, y1, x2, y2, w, rtype)

    # 绘制河流
    river_x = [0, SPACE_X, SPACE_X, 0]
    river_y = [RIVER_Y - RIVER_WIDTH/2, RIVER_Y - RIVER_WIDTH/2,
                RIVER_Y + RIVER_WIDTH/2, RIVER_Y + RIVER_WIDTH/2]
    verts_river = [list(zip(river_x, river_y, [0, 0, 0, 0]))]
    faces_river = Poly3DCollection(verts_river, alpha=0.5, facecolor='#1E88E5', edgecolor='#1565C0', linewidth=2)
    ax.add_collection3d(faces_river)
    ax.text(1000, RIVER_Y, 5, '河流 (宽80m)', ha='center', va='center', fontsize=11,
            color='#0D47A1', fontweight='bold',
            path_effects=[pe.withStroke(linewidth=3, foreground='white')])

    # 绘制桥梁
    for bx, by, bw, bh in BRIDGES:
        draw_bridge(ax, bx, by, bw, bh)

    # 绘制建筑物（最突出）
    for bx, by, bw, bd, bh, color, name in BUILDINGS:
        draw_box(ax, bx, by, bw, bd, bh, color, alpha=0.92)
        ax.text(bx, by, 0, name, ha='center', va='top', fontsize=7, color='#212121',
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])

    # 绘制高压线塔
    for x1, y1, x2, y2, h in POWER_LINES:
        draw_box(ax, x1, y1, 8, 8, h, '#FFD600', alpha=0.9)
        draw_box(ax, x2, y2, 8, 8, h, '#FFD600', alpha=0.9)
        ax.plot([x1, x2], [y1, y2], [h, h], color='#F57F17', linewidth=3, alpha=0.7)
        ax.text((x1+x2)/2, (y1+y2)/2, h + 8, '高压线', ha='center', va='bottom',
                fontsize=8, color='#E65100', fontweight='bold')

    # 绘制树木区域
    for tx, ty, tw, td, th in TREE_ZONES:
        draw_box(ax, tx, ty, tw, td, th, '#2E7D32', alpha=0.65)
        ax.text(tx, ty, th + 8, '树木区', ha='center', va='bottom', fontsize=8, color='#1B5E20')

    # 绘制特殊区域
    bx, by, br, bh = BIRD_ZONE
    draw_zone(ax, bx, by, br, bh, '#FF7043', alpha=0.12, label='飞鸟区')

    for nx, ny, nw, nd, name in NO_FLY_ZONES:
        draw_zone(ax, nx, ny, max(nw, nd)/2, SPACE_Z, '#D32F2F', alpha=0.08,
                 label=f'禁飞区({name})')

    gx, gy, gr, gh = GPS_INTERFERENCE
    draw_zone(ax, gx, gy, gr, gh, '#FF9800', alpha=0.08, label='GPS干扰区')

    wx, wy, wr, wh = WEATHER_ZONE
    draw_zone(ax, wx, wy, wr, wh, '#78909C', alpha=0.06, label='风雨区')

    # 绘制配送点
    for pid, px, py, pz, is_depot in DELIVERY_POINTS:
        if is_depot:
            ax.scatter(px, py, pz, s=500, c='#4CAF50', marker='s',
                      edgecolors='white', linewidth=4, zorder=200)
            ax.text(px, py, pz + 20, '配送中心', ha='center', va='bottom',
                    fontsize=11, color='#2E7D32', fontweight='bold',
                    path_effects=[pe.withStroke(linewidth=3, foreground='white')])
        else:
            ax.scatter(px, py, pz, s=250, c='#E53935', marker='^',
                      edgecolors='white', linewidth=3, zorder=200)
            ax.text(px, py, pz + 15, f'点{pid}', ha='center', va='bottom',
                    fontsize=9, color='#C62828', fontweight='bold',
                    path_effects=[pe.withStroke(linewidth=2, foreground='white')])

    # 标签
    ax.set_xlabel('X (m)', fontsize=13, fontweight='bold', labelpad=15)
    ax.set_ylabel('Y (m)', fontsize=13, fontweight='bold', labelpad=15)
    ax.set_zlabel('高度 (m)', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_title('城市空域环境三维视图', fontsize=18, fontweight='bold', pad=25)
    ax.view_init(elev=30, azim=45)

    # 图例
    legend_items = [
        mpatches.Patch(color='#E53935', label='建筑物', alpha=0.92),
        mpatches.Patch(color='#1E88E5', label='河流', alpha=0.5),
        mpatches.Patch(color='#8D6E63', label='桥梁', alpha=0.95),
        plt.Line2D([0], [0], color='#F57F17', linewidth=3, label='高压线'),
        mpatches.Patch(color='#2E7D32', label='树木区', alpha=0.65),
        mpatches.Patch(color='#FF7043', label='飞鸟区', alpha=0.3),
        mpatches.Patch(color='#D32F2F', label='禁飞区', alpha=0.3),
        mpatches.Patch(color='#FF9800', label='GPS干扰区', alpha=0.3),
        mpatches.Patch(color='#78909C', label='风雨区', alpha=0.3),
        mpatches.Patch(color='#FFD54F', label='主干道', alpha=0.25),
        mpatches.Patch(color='#FFE082', label='次干道', alpha=0.15),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#4CAF50',
                   markersize=14, label='配送中心'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='#E53935',
                   markersize=14, label='配送点'),
    ]
    ax.legend(handles=legend_items, loc='upper left', fontsize=9, frameon=True,
              framealpha=0.95, edgecolor='#333333', ncol=2)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'urban_env_3d_clean.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  保存: urban_env_3d_clean.png", flush=True)


def plot_top_clean():
    """绘制纯净版俯视图（无路径）"""
    _apply_style()

    fig, ax = plt.subplots(figsize=(16, 16), dpi=150)

    ax.set_xlim(0, SPACE_X)
    ax.set_ylim(0, SPACE_Y)
    ax.set_aspect('equal')
    ax.set_facecolor('#FAFAFA')

    # 道路（低透明度）
    for x1, y1, x2, y2, w, rtype in ROADS:
        if rtype == 'main':
            ax.plot([x1, x2], [y1, y2], color='#FFD54F', linewidth=w/2, alpha=0.2, solid_capstyle='round')
        else:
            ax.plot([x1, x2], [y1, y2], color='#FFE082', linewidth=w/2.5, alpha=0.12, solid_capstyle='round')

    # 河流
    river_rect = plt.Rectangle((0, RIVER_Y - RIVER_WIDTH/2), SPACE_X, RIVER_WIDTH,
                              facecolor='#1E88E5', edgecolor='#1565C0', linewidth=2, alpha=0.5)
    ax.add_patch(river_rect)
    ax.text(SPACE_X/2, RIVER_Y, '河流 (宽80m)', ha='center', va='center',
            fontsize=13, color='#0D47A1', fontweight='bold',
            path_effects=[pe.withStroke(linewidth=3, foreground='white')])

    # 桥梁
    for bx, by, bw, bh in BRIDGES:
        rect = plt.Rectangle((bx - bw/2, by - bh/2), bw, bh,
                            facecolor='#8D6E63', edgecolor='#4E342E', linewidth=3, alpha=0.95, zorder=10)
        ax.add_patch(rect)
        ax.text(bx, by, '桥梁', ha='center', va='center', fontsize=11,
                color='white', fontweight='bold', zorder=11)

    # 建筑物（最突出）
    for bx, by, bw, bd, bh, color, name in BUILDINGS:
        # 阴影
        shadow = plt.Rectangle((bx - bw/2 + 8, by - bd/2 - 8), bw, bd,
                              facecolor='#757575', edgecolor='none', alpha=0.25)
        ax.add_patch(shadow)

        # 建筑物主体
        rect = plt.Rectangle((bx - bw/2, by - bd/2), bw, bd,
                            facecolor=color, edgecolor='#212121', linewidth=2, alpha=0.92, zorder=5)
        ax.add_patch(rect)

        # 标注
        ax.text(bx, by, f'{name}\n{bh}m', ha='center', va='center', fontsize=8,
                color='white', fontweight='bold', zorder=6,
                path_effects=[pe.withStroke(linewidth=2.5, foreground='#212121')])

    # 高压线
    for x1, y1, x2, y2, h in POWER_LINES:
        ax.plot([x1, x2], [y1, y2], color='#F57F17', linewidth=4, alpha=0.7)
        ax.scatter([x1, x2], [y1, y2], s=250, c='#FFD600', marker='s',
                  edgecolors='#E65100', linewidth=2, zorder=8)
        ax.text((x1+x2)/2, (y1+y2)/2, f'高压线 {h}m', ha='center', va='center',
                fontsize=8, color='#E65100', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#E65100'))

    # 树木区域
    for tx, ty, tw, td, th in TREE_ZONES:
        rect = plt.Rectangle((tx - tw/2, ty - td/2), tw, td,
                            facecolor='#2E7D32', edgecolor='#1B5E20', linewidth=1.5, alpha=0.55, zorder=4)
        ax.add_patch(rect)
        ax.text(tx, ty, f'树木区\n{th}m', ha='center', va='center', fontsize=8, color='white')

    # 特殊区域
    bx, by, br, bh = BIRD_ZONE
    circle = plt.Circle((bx, by), br, facecolor='#FF7043', edgecolor='#E64A19',
                        linewidth=2.5, alpha=0.15, linestyle='--', zorder=3)
    ax.add_patch(circle)
    ax.text(bx, by, f'飞鸟区\n半径{br}m', ha='center', va='center',
            fontsize=9, color='#BF360C', fontweight='bold')

    for nx, ny, nw, nd, name in NO_FLY_ZONES:
        rect = plt.Rectangle((nx - nw/2, ny - nd/2), nw, nd,
                            facecolor='#D32F2F', edgecolor='#B71C1C', linewidth=2.5,
                            alpha=0.15, linestyle='--', zorder=3)
        ax.add_patch(rect)
        ax.text(nx, ny, f'禁飞区\n{name}', ha='center', va='center',
                fontsize=10, color='#B71C1C', fontweight='bold')

    gx, gy, gr, gh = GPS_INTERFERENCE
    circle = plt.Circle((gx, gy), gr, facecolor='#FF9800', edgecolor='#E65100',
                        linewidth=2.5, alpha=0.12, linestyle='-.', zorder=3)
    ax.add_patch(circle)
    ax.text(gx, gy, f'GPS干扰区\n半径{gr}m', ha='center', va='center',
            fontsize=9, color='#E65100', fontweight='bold')

    wx, wy, wr, wh = WEATHER_ZONE
    circle = plt.Circle((wx, wy), wr, facecolor='#78909C', edgecolor='#546E7A',
                        linewidth=2.5, alpha=0.08, linestyle=':', zorder=3)
    ax.add_patch(circle)
    ax.text(wx, wy, f'风雨区\n半径{wr}m', ha='center', va='center',
            fontsize=9, color='#37474F', fontweight='bold')

    # 配送点
    for pid, px, py, pz, is_depot in DELIVERY_POINTS:
        if is_depot:
            ax.scatter(px, py, s=600, c='#4CAF50', marker='s',
                      edgecolors='white', linewidth=5, zorder=100)
            ax.text(px, py - 45, '配送中心', ha='center', va='top',
                    fontsize=12, color='#2E7D32', fontweight='bold',
                    path_effects=[pe.withStroke(linewidth=3, foreground='white')])
        else:
            ax.scatter(px, py, s=300, c='#E53935', marker='^',
                      edgecolors='white', linewidth=3, zorder=100)
            ax.text(px, py + 30, f'点{pid}', ha='center', va='bottom',
                    fontsize=10, color='#C62828', fontweight='bold',
                    path_effects=[pe.withStroke(linewidth=2, foreground='white')])

    # 标签
    ax.set_xlabel('X (m)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Y (m)', fontsize=14, fontweight='bold')
    ax.set_title('城市空域环境俯视图', fontsize=18, fontweight='bold')
    ax.grid(True, alpha=0.1, linestyle='--', color='#9E9E9E')

    # 图例
    legend_items = [
        mpatches.Patch(color='#E53935', label='建筑物', alpha=0.92),
        mpatches.Patch(color='#1E88E5', label='河流', alpha=0.5),
        mpatches.Patch(color='#8D6E63', label='桥梁', alpha=0.95),
        plt.Line2D([0], [0], color='#F57F17', linewidth=4, label='高压线'),
        mpatches.Patch(color='#2E7D32', label='树木区', alpha=0.55),
        mpatches.Patch(color='#FF7043', label='飞鸟区', alpha=0.3),
        mpatches.Patch(color='#D32F2F', label='禁飞区', alpha=0.3),
        mpatches.Patch(color='#FF9800', label='GPS干扰区', alpha=0.3),
        mpatches.Patch(color='#78909C', label='风雨区', alpha=0.3),
        mpatches.Patch(color='#FFD54F', label='主干道', alpha=0.2),
        mpatches.Patch(color='#FFE082', label='次干道', alpha=0.12),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#4CAF50',
                   markersize=16, label='配送中心'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='#E53935',
                   markersize=16, label='配送点'),
    ]
    ax.legend(handles=legend_items, loc='upper left', fontsize=10, frameon=True,
              framealpha=0.95, edgecolor='#333333', ncol=2)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'urban_env_top_clean.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  保存: urban_env_top_clean.png", flush=True)


if __name__ == '__main__':
    print("=" * 60, flush=True)
    print("城市空域环境纯净版可视化（无路径）", flush=True)
    print("=" * 60, flush=True)

    print("\n1. 生成三维环境图...", flush=True)
    plot_3d_clean()

    print("\n2. 生成俯视图...", flush=True)
    plot_top_clean()

    print("\n完成!", flush=True)
