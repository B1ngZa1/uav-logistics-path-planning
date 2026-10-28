"""
武汉光谷无人机配送环境示意图生成脚本。

输出：
  - wuhan_environment_3d.png
  - wuhan_zone_top_view.png
  - wuhan_building_height_heatmap.png
  - wuhan_obstacle_legend.png
  - wuhan_environment_overview.png

说明：
  这是可复现的程序化脱敏示意场景，不是 GIS 实测地图，不提供现实配送航线，
  也不能用于填充环境统计。未来导入 OSM/DEM 后，可保留本脚本的绘图接口并
  替换 make_scene() 的数据来源。
"""

import argparse
import math
from pathlib import Path as FsPath

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.patches import (
    Ellipse,
    FancyArrowPatch,
    FancyBboxPatch,
    Patch,
    Polygon as MplPolygon,
    Rectangle,
)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


SCENE_W_M = 8000.0
SCENE_H_M = 6000.0
SCENE_Z_M = 220.0
Z_EXAGGERATION = 5.0
DEFAULT_SEED = 20260703

ZONE_COLORS = {
    "光谷核心区": "#FCE8E6",
    "关山产业区": "#FFF1D6",
    "南湖沿岸区": "#E8F4FD",
    "东湖南缘区": "#E8F5E9",
}

BUILDING_COLORS = {
    "低层住宅": "#D8C3A5",
    "多层住宅": "#F4A261",
    "高层住宅": "#E76F51",
    "商业综合体": "#9B5DE5",
    "办公塔楼": "#277DA1",
    "工业厂房": "#78909C",
    "公共建筑": "#00A6A6",
    "景区建筑": "#5E9F67",
}

RISK_STYLES = {
    "硬禁飞区": ("#D32F2F", "///"),
    "GPS多路径": ("#8E44AD", "..."),
    "通信阴影": ("#546E7A", "\\\\\\"),
    "湖面侧风": ("#039BE5", "xx"),
    "降雨低能见度": ("#607D8B", "oo"),
    "临时施工区": ("#FB8C00", "++"),
}

WATER_COLOR = "#58A6E7"
WATER_EDGE = "#1976B9"
FOREST_COLOR = "#3F7F4C"
ROAD_MAIN = "#ECE3D1"
ROAD_SECONDARY = "#F5F0E6"
BRIDGE_COLOR = "#795548"
POWER_COLOR = "#E5A500"
PATH_PLAN = "#1565C0"
PATH_EXEC = "#43A047"
PATH_RETURN = "#6D4C41"
TEXT_DARK = "#263238"
GRID_COLOR = "#CFD8DC"


def apply_style():
    """应用适合论文与答辩展示的中文绘图风格。"""
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.edgecolor"] = "#455A64"
    plt.rcParams["axes.labelcolor"] = TEXT_DARK
    plt.rcParams["xtick.color"] = TEXT_DARK
    plt.rcParams["ytick.color"] = TEXT_DARK
    plt.rcParams["savefig.facecolor"] = "white"
    plt.rcParams["figure.facecolor"] = "white"


def terrain_height(x, y):
    """程序化丘陵地形；输入可为标量或数组。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    hill_1 = 58.0 * np.exp(-(((x - 900.0) / 1300.0) ** 2 + ((y - 5050.0) / 950.0) ** 2))
    hill_2 = 35.0 * np.exp(-(((x - 2850.0) / 1000.0) ** 2 + ((y - 5200.0) / 750.0) ** 2))
    hill_3 = 16.0 * np.exp(-(((x - 4100.0) / 1700.0) ** 2 + ((y - 3500.0) / 1300.0) ** 2))
    undulation = 4.0 * np.sin(x / 1150.0) * np.cos(y / 900.0)
    return np.maximum(3.0, 7.0 + hill_1 + hill_2 + hill_3 + undulation)


def rotate_points(points, angle_deg, center):
    angle = math.radians(angle_deg)
    c, s = math.cos(angle), math.sin(angle)
    pts = np.asarray(points, dtype=float) - np.asarray(center)
    rot = np.array([[c, -s], [s, c]])
    return pts @ rot.T + np.asarray(center)


def make_footprint(cx, cy, width, depth, variant, angle_deg=0.0):
    """生成矩形、切角或 L 型建筑轮廓。"""
    x0, x1 = cx - width / 2, cx + width / 2
    y0, y1 = cy - depth / 2, cy + depth / 2
    if variant == 0:
        points = [
            (x0, y0), (x1, y0), (x1, y1), (x0, y1)
        ]
    elif variant == 1:
        cut = min(width, depth) * 0.22
        points = [
            (x0 + cut, y0), (x1 - cut, y0), (x1, y0 + cut),
            (x1, y1), (x0, y1), (x0, y0 + cut)
        ]
    else:
        notch_x = x0 + width * 0.46
        notch_y = y0 + depth * 0.48
        points = [
            (x0, y0), (x1, y0), (x1, notch_y),
            (notch_x, notch_y), (notch_x, y1), (x0, y1)
        ]
    return rotate_points(points, angle_deg, (cx, cy))


def poly_centroid(poly):
    """简单多边形质心；退化时使用顶点均值。"""
    p = np.asarray(poly, dtype=float)
    x, y = p[:, 0], p[:, 1]
    x2, y2 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y2 - x2 * y
    area2 = cross.sum()
    if abs(area2) < 1e-9:
        return p.mean(axis=0)
    cx = ((x + x2) * cross).sum() / (3.0 * area2)
    cy = ((y + y2) * cross).sum() / (3.0 * area2)
    return np.array([cx, cy])


def irregular_lake(cx, cy, rx, ry, phase=0.0, points=90):
    theta = np.linspace(0.0, 2.0 * np.pi, points, endpoint=False)
    radial = (
        1.0
        + 0.10 * np.sin(3.0 * theta + phase)
        + 0.06 * np.cos(5.0 * theta - phase)
        + 0.035 * np.sin(8.0 * theta)
    )
    return np.column_stack([
        cx + rx * radial * np.cos(theta),
        cy + ry * radial * np.sin(theta),
    ])


def point_in_polygon(point, polygon):
    """射线法点在多边形内判断，避免额外 GIS 依赖。"""
    x, y = point
    poly = np.asarray(polygon)
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def make_buildings(rng, waters):
    """按四分区生成八类、非统一方盒的建筑簇。"""
    specs = [
        # zone, type, center x/y, rows, cols, spacing x/y, height range, footprint range, angle
        ("光谷核心区", "商业综合体", 6100, 1250, 2, 4, 430, 420, (55, 105), (210, 330), 8),
        ("光谷核心区", "办公塔楼", 6900, 2100, 3, 4, 360, 360, (85, 180), (120, 220), -6),
        ("光谷核心区", "高层住宅", 4800, 1050, 3, 4, 310, 300, (80, 135), (105, 175), 2),
        ("光谷核心区", "多层住宅", 5050, 2250, 2, 5, 280, 250, (35, 70), (95, 160), 0),
        ("关山产业区", "办公塔楼", 5050, 4150, 3, 4, 390, 370, (60, 150), (130, 220), -4),
        ("关山产业区", "工业厂房", 6900, 4550, 3, 3, 480, 390, (18, 42), (260, 420), 0),
        ("关山产业区", "公共建筑", 6500, 3450, 2, 4, 420, 330, (25, 65), (190, 300), 5),
        ("南湖沿岸区", "高层住宅", 3000, 2200, 2, 4, 310, 300, (65, 105), (115, 170), -5),
        ("南湖沿岸区", "多层住宅", 950, 2050, 2, 6, 250, 250, (35, 65), (95, 150), 3),
        ("南湖沿岸区", "低层住宅", 3150, 650, 2, 4, 260, 240, (8, 20), (80, 140), 7),
        ("东湖南缘区", "景区建筑", 2850, 4650, 3, 4, 300, 260, (8, 24), (90, 160), -8),
        ("东湖南缘区", "低层住宅", 3100, 5600, 2, 5, 250, 220, (8, 18), (75, 130), 5),
        ("东湖南缘区", "公共建筑", 1150, 3650, 2, 4, 360, 310, (18, 45), (150, 240), 0),
    ]

    buildings = []
    building_id = 1
    for zone, btype, cx0, cy0, rows, cols, sx, sy, hrange, frange, angle in specs:
        for row in range(rows):
            for col in range(cols):
                cx = cx0 + (col - (cols - 1) / 2) * sx + rng.uniform(-45, 45)
                cy = cy0 + (row - (rows - 1) / 2) * sy + rng.uniform(-40, 40)
                width = rng.uniform(frange[0], frange[1])
                depth = rng.uniform(frange[0] * 0.70, frange[1] * 0.88)
                variant = int(rng.integers(0, 3))
                footprint = make_footprint(
                    cx, cy, width, depth, variant, angle + rng.uniform(-4, 4)
                )
                centroid = poly_centroid(footprint)
                if any(point_in_polygon(centroid, water["polygon"]) for water in waters):
                    continue
                height = float(rng.uniform(hrange[0], hrange[1]))
                buildings.append({
                    "id": "BLD-%03d" % building_id,
                    "zone": zone,
                    "type": btype,
                    "footprint": footprint,
                    "height_m": height,
                    "ground_m": float(terrain_height(centroid[0], centroid[1])),
                    "estimated": bool(rng.random() < 0.30),
                    "roof": ["平屋顶", "坡屋顶", "设备屋顶"][variant],
                })
                building_id += 1
    return buildings


def choose_building(buildings, zone, btype, index):
    candidates = [b for b in buildings if b["zone"] == zone and b["type"] == btype]
    candidates.sort(key=lambda item: item["id"])
    return candidates[index % len(candidates)]


def make_delivery_points(buildings):
    """创建 12 个脱敏任务点；屋顶点从程序化建筑中选取。"""
    specs = [
        (1, "配送中心", "关山产业区", None, None, (5150, 5200)),
        (2, "商业楼顶", "光谷核心区", "商业综合体", 1, None),
        (3, "办公园区", "光谷核心区", "办公塔楼", 3, None),
        (4, "商业楼顶", "光谷核心区", "商业综合体", 6, None),
        (5, "办公园区", "关山产业区", "办公塔楼", 2, None),
        (6, "办公园区", "关山产业区", "公共建筑", 4, None),
        (7, "住宅区", "南湖沿岸区", "高层住宅", 2, None),
        (8, "住宅区", "南湖沿岸区", "多层住宅", 6, None),
        (9, "湖岸设施", "南湖沿岸区", None, None, (3520, 1180)),
        (10, "湖岸设施", "东湖南缘区", None, None, (2550, 4050)),
        (11, "丘陵边缘", "东湖南缘区", None, None, (1050, 5050)),
        (12, "景区建筑", "东湖南缘区", "景区建筑", 5, None),
    ]

    points = []
    for pid, site_type, zone, btype, index, fixed_xy in specs:
        if btype is not None:
            building = choose_building(buildings, zone, btype, index)
            xy = poly_centroid(building["footprint"])
            z = building["ground_m"] + building["height_m"] + 3.0
        else:
            xy = np.asarray(fixed_xy, dtype=float)
            z = float(terrain_height(xy[0], xy[1]) + (8.0 if pid == 1 else 5.0))
        points.append({
            "id": pid,
            "name": "P%d" % pid,
            "site_type": site_type,
            "zone": zone,
            "xy": np.asarray(xy, dtype=float),
            "z_m": z,
        })
    return points


def make_scene(seed=DEFAULT_SEED):
    rng = np.random.default_rng(seed)
    south_lake = irregular_lake(1550, 930, 1420, 700, phase=0.4)
    east_lake = irregular_lake(850, 5150, 1550, 900, phase=1.3)
    east_lake[:, 0] = np.clip(east_lake[:, 0], -350, 2650)
    east_lake[:, 1] = np.clip(east_lake[:, 1], 4100, 6300)
    waters = [
        {"name": "南湖水域边缘", "polygon": south_lake, "level_m": 5.5},
        {"name": "东湖水域边缘", "polygon": east_lake, "level_m": 8.0},
    ]

    buildings = make_buildings(rng, waters)
    deliveries = make_delivery_points(buildings)

    roads = [
        {"class": "main", "points": np.array([[3900, 200], [4300, 5900]])},
        {"class": "main", "points": np.array([[3900, 2900], [7900, 2900]])},
        {"class": "main", "points": np.array([[4050, 3850], [7900, 3850]])},
        {"class": "main", "points": np.array([[5550, 0], [5550, 6000]])},
        {"class": "main", "points": np.array([[7200, 0], [7200, 6000]])},
        {"class": "secondary", "points": np.array([[3900, 750], [8000, 750]])},
        {"class": "secondary", "points": np.array([[3900, 1650], [8000, 1650]])},
        {"class": "secondary", "points": np.array([[3900, 4700], [8000, 4700]])},
        {"class": "secondary", "points": np.array([[4700, 3000], [4700, 6000]])},
        {"class": "secondary", "points": np.array([[6400, 3000], [6400, 6000]])},
        {"class": "shore", "points": np.array([
            [250, 1850], [900, 2000], [1700, 2050], [2500, 2000], [3500, 1750]
        ])},
        {"class": "shore", "points": np.array([
            [2300, 3850], [2650, 4300], [3000, 4800], [3500, 5350]
        ])},
    ]

    bridges = [
        {"name": "BR-01", "points": np.array([[1450, 150], [1700, 1750]])},
        {"name": "BR-02", "points": np.array([[500, 980], [2950, 980]])},
        {"name": "BR-03", "points": np.array([[300, 4450], [1900, 5650]])},
    ]

    forests = [
        {"center": (700, 3550), "width": 1100, "height": 720, "angle": 8},
        {"center": (2500, 5200), "width": 1250, "height": 900, "angle": -12},
        {"center": (3200, 4050), "width": 780, "height": 620, "angle": 18},
    ]

    power_lines = [
        np.array([[4350, 3500, 55], [5200, 3650, 62], [6100, 3500, 58]]),
        np.array([[6100, 5450, 65], [6900, 5300, 70], [7750, 5450, 65]]),
    ]

    risks = [
        {"kind": "硬禁飞区", "center": (7450, 3400), "width": 650, "height": 780, "angle": 8},
        {"kind": "GPS多路径", "center": (6500, 1850), "width": 1450, "height": 1200, "angle": -8},
        {"kind": "通信阴影", "center": (1500, 4550), "width": 1500, "height": 950, "angle": 15},
        {"kind": "湖面侧风", "center": (1600, 950), "width": 2700, "height": 1200, "angle": 0},
        {"kind": "降雨低能见度", "center": (4300, 3350), "width": 1150, "height": 980, "angle": -20},
        {"kind": "临时施工区", "center": (6650, 4550), "width": 780, "height": 650, "angle": 0},
    ]

    dynamic_entities = [
        {"kind": "车辆", "xy": (5750, 2900), "marker": "s", "color": "#455A64"},
        {"kind": "其他无人机", "xy": (5900, 2500), "marker": "D", "color": "#EC407A"},
        {"kind": "飞鸟群", "xy": (2450, 3950), "marker": "v", "color": "#5D4037"},
        {"kind": "船只", "xy": (1250, 900), "marker": ">", "color": "#0D47A1"},
        {"kind": "施工吊臂", "xy": (6650, 4550), "marker": "P", "color": "#F57C00"},
    ]

    route_order = [1, 5, 6, 4, 3, 2, 7, 8, 9, 10, 11, 12, 1]
    by_id = {p["id"]: p for p in deliveries}
    planned_path = np.array([by_id[i]["xy"] for i in route_order])
    executed_path = planned_path.copy()
    phase = np.linspace(0, 2 * np.pi, len(executed_path))
    executed_path[:, 0] += 80 * np.sin(phase)
    executed_path[:, 1] += 55 * np.cos(phase * 1.3)
    return_path = np.array([
        by_id[12]["xy"],
        [3900, 5200],
        by_id[1]["xy"],
    ])

    return {
        "seed": seed,
        "waters": waters,
        "buildings": buildings,
        "deliveries": deliveries,
        "roads": roads,
        "bridges": bridges,
        "forests": forests,
        "power_lines": power_lines,
        "risks": risks,
        "dynamic_entities": dynamic_entities,
        "planned_path": planned_path,
        "executed_path": executed_path,
        "return_path": return_path,
    }


def add_zone_background(ax, labels=True):
    zones = [
        ("南湖沿岸区", 0, 0, 4000, 3000),
        ("光谷核心区", 4000, 0, 4000, 3000),
        ("东湖南缘区", 0, 3000, 4000, 3000),
        ("关山产业区", 4000, 3000, 4000, 3000),
    ]
    for name, x, y, w, h in zones:
        ax.add_patch(Rectangle(
            (x, y), w, h,
            facecolor=ZONE_COLORS[name],
            edgecolor="#90A4AE",
            linewidth=1.2,
            linestyle=(0, (5, 4)),
            zorder=0,
        ))
        if labels:
            ax.text(
                x + 170, y + h - 230, name,
                fontsize=12, fontweight="bold", color="#455A64", zorder=20,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=2.5),
            )


def add_map_frame(ax, title, subtitle=None):
    ax.set_xlim(0, SCENE_W_M)
    ax.set_ylim(0, SCENE_H_M)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("东向 E (km)", fontweight="bold")
    ax.set_ylabel("北向 N (km)", fontweight="bold")
    ticks_x = np.arange(0, SCENE_W_M + 1, 1000)
    ticks_y = np.arange(0, SCENE_H_M + 1, 1000)
    ax.set_xticks(ticks_x)
    ax.set_yticks(ticks_y)
    ax.set_xticklabels(["%.0f" % (v / 1000) for v in ticks_x])
    ax.set_yticklabels(["%.0f" % (v / 1000) for v in ticks_y])
    ax.grid(color=GRID_COLOR, linewidth=0.5, alpha=0.45, zorder=-5)
    ax.set_title(title, fontsize=16, fontweight="bold", pad=14)
    if subtitle:
        ax.text(
            0.5, 1.005, subtitle,
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=9, color="#607D8B",
        )


def draw_terrain_2d(ax, alpha=0.28):
    x = np.linspace(0, SCENE_W_M, 90)
    y = np.linspace(0, SCENE_H_M, 70)
    xx, yy = np.meshgrid(x, y)
    zz = terrain_height(xx, yy)
    ax.contourf(
        xx, yy, zz,
        levels=np.linspace(float(zz.min()), float(zz.max()), 14),
        cmap="Greys",
        alpha=alpha,
        zorder=0.3,
    )
    ax.contour(
        xx, yy, zz,
        levels=np.arange(10, 81, 10),
        colors="#8D9A9E",
        linewidths=0.45,
        alpha=0.45,
        zorder=0.4,
    )


def draw_waters_2d(ax, scene, labels=True):
    for water in scene["waters"]:
        ax.add_patch(MplPolygon(
            water["polygon"], closed=True,
            facecolor=WATER_COLOR, edgecolor=WATER_EDGE,
            linewidth=1.8, alpha=0.84, zorder=2,
        ))
        if labels:
            c = water["polygon"].mean(axis=0)
            ax.text(
                c[0], c[1], water["name"],
                ha="center", va="center", color="white",
                fontsize=10, fontweight="bold", zorder=15,
                path_effects=[pe.withStroke(linewidth=2.4, foreground=WATER_EDGE)],
            )


def draw_forests_2d(ax, scene, labels=False):
    for forest in scene["forests"]:
        patch = Ellipse(
            forest["center"], forest["width"], forest["height"],
            angle=forest["angle"],
            facecolor=FOREST_COLOR, edgecolor="#1B5E20",
            alpha=0.38, hatch="..", linewidth=1.3, zorder=3,
        )
        ax.add_patch(patch)
        if labels:
            ax.text(
                forest["center"][0], forest["center"][1], "林地",
                ha="center", va="center", color="#1B5E20",
                fontweight="bold", fontsize=8, zorder=15,
            )


def draw_roads_2d(ax, scene):
    for road in scene["roads"]:
        points = road["points"]
        if road["class"] == "main":
            color, width, z = ROAD_MAIN, 8.0, 4
        elif road["class"] == "secondary":
            color, width, z = ROAD_SECONDARY, 4.5, 4
        else:
            color, width, z = "#C8BCA8", 3.2, 4
        ax.plot(
            points[:, 0], points[:, 1],
            color="#9E9E9E", linewidth=width + 1.6,
            solid_capstyle="round", alpha=0.55, zorder=z,
        )
        ax.plot(
            points[:, 0], points[:, 1],
            color=color, linewidth=width,
            solid_capstyle="round", alpha=0.95, zorder=z + 0.1,
        )

    for bridge in scene["bridges"]:
        points = bridge["points"]
        ax.plot(
            points[:, 0], points[:, 1],
            color=BRIDGE_COLOR, linewidth=10,
            solid_capstyle="round", alpha=0.96, zorder=6,
        )
        ax.plot(
            points[:, 0], points[:, 1],
            color="#D7CCC8", linewidth=5.3,
            solid_capstyle="round", alpha=1.0, zorder=6.1,
        )


def draw_buildings_2d(ax, scene, by_height=False, show_estimated=False):
    norm = mcolors.Normalize(vmin=0, vmax=180)
    cmap = plt.get_cmap("viridis")
    for building in scene["buildings"]:
        if by_height:
            face = cmap(norm(building["height_m"]))
        else:
            face = BUILDING_COLORS[building["type"]]
        hatch = "///" if show_estimated and building["estimated"] else None
        ax.add_patch(MplPolygon(
            building["footprint"], closed=True,
            facecolor=face, edgecolor="#37474F",
            linewidth=0.45, alpha=0.92,
            hatch=hatch, zorder=7,
        ))
    return norm, cmap


def draw_power_lines_2d(ax, scene):
    for line in scene["power_lines"]:
        ax.plot(
            line[:, 0], line[:, 1],
            color=POWER_COLOR, linewidth=1.6,
            linestyle=(0, (6, 3)), zorder=9,
        )
        ax.scatter(
            line[:, 0], line[:, 1],
            marker="|", s=95, linewidths=2,
            color="#7A5A00", zorder=10,
        )


def draw_risks_2d(ax, scene, label=True):
    for risk in scene["risks"]:
        color, hatch = RISK_STYLES[risk["kind"]]
        patch = Ellipse(
            risk["center"], risk["width"], risk["height"],
            angle=risk["angle"],
            facecolor=mcolors.to_rgba(color, 0.12),
            edgecolor=color, linewidth=1.6,
            linestyle=(0, (5, 3)), hatch=hatch, zorder=10,
        )
        ax.add_patch(patch)
        if label:
            ax.text(
                risk["center"][0],
                risk["center"][1] + risk["height"] * 0.38,
                risk["kind"],
                ha="center", va="center", fontsize=7.5,
                color=color, fontweight="bold", zorder=17,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=1.5),
            )


def draw_paths_2d(ax, scene, labels=True):
    planned = scene["planned_path"]
    executed = scene["executed_path"]
    return_path = scene["return_path"]
    ax.plot(
        planned[:, 0], planned[:, 1],
        color=PATH_PLAN, linewidth=2.6, alpha=0.82,
        linestyle="-", zorder=12,
    )
    ax.plot(
        executed[:, 0], executed[:, 1],
        color=PATH_EXEC, linewidth=1.8, alpha=0.92,
        linestyle=(0, (5, 2)), zorder=13,
    )
    ax.plot(
        return_path[:, 0], return_path[:, 1],
        color=PATH_RETURN, linewidth=2.0, alpha=0.85,
        linestyle=(0, (2, 2)), zorder=12,
    )
    if labels:
        ax.text(
            5900, 5650, "路径仅为分级表达示意",
            color=PATH_PLAN, fontsize=8, fontweight="bold",
            bbox=dict(facecolor="white", edgecolor=PATH_PLAN, alpha=0.88, boxstyle="round,pad=0.25"),
            zorder=18,
        )


def draw_delivery_points_2d(ax, scene, labels=True):
    for point in scene["deliveries"]:
        xy = point["xy"]
        is_depot = point["id"] == 1
        marker = "*" if is_depot else "o"
        size = 160 if is_depot else 55
        color = "#2E7D32" if is_depot else "#FFFFFF"
        edge = "#1B5E20" if is_depot else "#D81B60"
        ax.scatter(
            [xy[0]], [xy[1]],
            marker=marker, s=size,
            facecolor=color, edgecolor=edge,
            linewidth=1.5, zorder=16,
        )
        if labels:
            ax.annotate(
                point["name"],
                xy=xy, xytext=(5, 5), textcoords="offset points",
                fontsize=7.5, fontweight="bold",
                color="#1B5E20" if is_depot else "#AD1457",
                zorder=18,
                path_effects=[pe.withStroke(linewidth=2.2, foreground="white")],
            )


def draw_dynamic_entities_2d(ax, scene):
    for entity in scene["dynamic_entities"]:
        x, y = entity["xy"]
        ax.scatter(
            [x], [y], marker=entity["marker"], s=62,
            facecolor=entity["color"], edgecolor="white",
            linewidth=0.8, zorder=16,
        )


def add_cartographic_elements(ax):
    # 比例尺
    x0, y0, length = 520, 260, 2000
    ax.plot([x0, x0 + length], [y0, y0], color=TEXT_DARK, linewidth=4, zorder=25)
    for x in [x0, x0 + length / 2, x0 + length]:
        ax.plot([x, x], [y0 - 45, y0 + 45], color=TEXT_DARK, linewidth=1.3, zorder=25)
    ax.text(x0, y0 + 85, "0", ha="center", fontsize=8, zorder=25)
    ax.text(x0 + length / 2, y0 + 85, "1", ha="center", fontsize=8, zorder=25)
    ax.text(x0 + length, y0 + 85, "2 km", ha="center", fontsize=8, zorder=25)

    # 指北针
    ax.annotate(
        "N", xy=(7600, 5650), xytext=(7600, 5150),
        ha="center", va="center", fontsize=11, fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color=TEXT_DARK, lw=1.8),
        zorder=25,
    )


def schematic_footer(fig, text=None):
    text = text or "程序化脱敏示意 · 非 GIS 实测统计 · 非现实配送航线 · seed=20260703"
    fig.text(
        0.5, 0.012, text,
        ha="center", va="bottom",
        fontsize=8, color="#607D8B",
    )


def draw_top_view(ax, scene, compact=False):
    add_zone_background(ax, labels=not compact)
    draw_terrain_2d(ax, alpha=0.16 if compact else 0.22)
    draw_waters_2d(ax, scene, labels=not compact)
    draw_forests_2d(ax, scene, labels=not compact)
    draw_roads_2d(ax, scene)
    draw_buildings_2d(ax, scene, by_height=False)
    draw_power_lines_2d(ax, scene)
    draw_risks_2d(ax, scene, label=not compact)
    draw_paths_2d(ax, scene, labels=not compact)
    draw_delivery_points_2d(ax, scene, labels=True)
    draw_dynamic_entities_2d(ax, scene)
    add_map_frame(
        ax,
        "武汉光谷无人机配送环境分区俯视图" if not compact else "分区俯视图",
        None if compact else "8 × 6 km 脱敏 ENU 示意；建筑按类型着色，风险区为半透明纹理",
    )
    if not compact:
        add_cartographic_elements(ax)


def extruded_polygon_faces(poly, z0, z1):
    poly = np.asarray(poly)
    bottom = [(x, y, z0) for x, y in poly]
    top = [(x, y, z1) for x, y in poly]
    faces = [bottom, top]
    for i in range(len(poly)):
        j = (i + 1) % len(poly)
        faces.append([
            (poly[i, 0], poly[i, 1], z0),
            (poly[j, 0], poly[j, 1], z0),
            (poly[j, 0], poly[j, 1], z1),
            (poly[i, 0], poly[i, 1], z1),
        ])
    return faces


def draw_risk_cylinder_3d(ax, risk, z_top_m):
    color, _ = RISK_STYLES[risk["kind"]]
    theta = np.linspace(0, 2 * np.pi, 60)
    angle = math.radians(risk["angle"])
    x0 = risk["width"] / 2 * np.cos(theta)
    y0 = risk["height"] / 2 * np.sin(theta)
    x = risk["center"][0] + x0 * math.cos(angle) - y0 * math.sin(angle)
    y = risk["center"][1] + x0 * math.sin(angle) + y0 * math.cos(angle)
    ground = terrain_height(x, y) * Z_EXAGGERATION
    top = np.full_like(x, z_top_m * Z_EXAGGERATION)
    ax.plot(x, y, ground, color=color, lw=1.1, alpha=0.65, linestyle="--")
    ax.plot(x, y, top, color=color, lw=0.9, alpha=0.48, linestyle="--")
    for idx in range(0, len(theta), 12):
        ax.plot([x[idx], x[idx]], [y[idx], y[idx]], [ground[idx], top[idx]],
                color=color, lw=0.55, alpha=0.34)


def draw_3d(ax, scene, compact=False):
    x = np.linspace(0, SCENE_W_M, 52)
    y = np.linspace(0, SCENE_H_M, 40)
    xx, yy = np.meshgrid(x, y)
    zz = terrain_height(xx, yy) * Z_EXAGGERATION
    ax.plot_surface(
        xx, yy, zz,
        cmap="terrain", alpha=0.42,
        linewidth=0, antialiased=True, shade=True, zorder=0,
    )

    for water in scene["waters"]:
        z = water["level_m"] * Z_EXAGGERATION
        verts = [[(px, py, z) for px, py in water["polygon"]]]
        ax.add_collection3d(Poly3DCollection(
            verts, facecolor=WATER_COLOR, edgecolor=WATER_EDGE,
            linewidth=0.8, alpha=0.82,
        ))

    for road in scene["roads"]:
        p = road["points"]
        zroad = terrain_height(p[:, 0], p[:, 1]) * Z_EXAGGERATION + 2
        color = "#B0A58F" if road["class"] == "main" else "#CFC7B7"
        width = 1.5 if road["class"] == "main" else 0.8
        ax.plot(p[:, 0], p[:, 1], zroad, color=color, lw=width, alpha=0.8)

    for bridge in scene["bridges"]:
        p = bridge["points"]
        zbridge = np.maximum(terrain_height(p[:, 0], p[:, 1]) + 18, 25) * Z_EXAGGERATION
        ax.plot(p[:, 0], p[:, 1], zbridge, color=BRIDGE_COLOR, lw=4.2, alpha=0.95)

    for forest in scene["forests"]:
        count = 55 if not compact else 30
        theta = np.linspace(0, 2 * np.pi, count, endpoint=False)
        radial = np.sqrt(np.linspace(0.04, 1.0, count))
        fx = forest["center"][0] + forest["width"] * 0.47 * radial * np.cos(theta)
        fy = forest["center"][1] + forest["height"] * 0.47 * radial * np.sin(theta)
        fz = terrain_height(fx, fy) * Z_EXAGGERATION + 7
        ax.scatter(fx, fy, fz, s=7, color=FOREST_COLOR, alpha=0.55, marker="^")

    for building in scene["buildings"]:
        z0 = building["ground_m"] * Z_EXAGGERATION
        z1 = (building["ground_m"] + building["height_m"]) * Z_EXAGGERATION
        faces = extruded_polygon_faces(building["footprint"], z0, z1)
        ax.add_collection3d(Poly3DCollection(
            faces,
            facecolor=BUILDING_COLORS[building["type"]],
            edgecolor="#37474F",
            linewidth=0.25,
            alpha=0.90,
        ))

    for line in scene["power_lines"]:
        ax.plot(
            line[:, 0], line[:, 1], line[:, 2] * Z_EXAGGERATION,
            color=POWER_COLOR, lw=1.3, linestyle="--", alpha=0.9,
        )
        for p in line:
            ground = float(terrain_height(p[0], p[1])) * Z_EXAGGERATION
            ax.plot([p[0], p[0]], [p[1], p[1]], [ground, p[2] * Z_EXAGGERATION],
                    color="#7A5A00", lw=0.8, alpha=0.8)

    for risk in scene["risks"]:
        top = 190 if risk["kind"] == "硬禁飞区" else 145
        draw_risk_cylinder_3d(ax, risk, top)

    path = scene["planned_path"]
    path_z = (115 + 15 * np.sin(np.linspace(0, 2 * np.pi, len(path)))) * Z_EXAGGERATION
    ax.plot(path[:, 0], path[:, 1], path_z, color=PATH_PLAN, lw=2.3, alpha=0.88)
    executed = scene["executed_path"]
    ax.plot(
        executed[:, 0], executed[:, 1], path_z + 8 * Z_EXAGGERATION,
        color=PATH_EXEC, lw=1.5, linestyle="--", alpha=0.9,
    )

    for point in scene["deliveries"]:
        x0, y0 = point["xy"]
        z0 = point["z_m"] * Z_EXAGGERATION
        is_depot = point["id"] == 1
        ax.scatter(
            [x0], [y0], [z0],
            marker="*" if is_depot else "o",
            s=70 if is_depot else 22,
            color="#2E7D32" if is_depot else "#D81B60",
            edgecolors="white", linewidths=0.55, depthshade=False,
        )
        if not compact or point["id"] in (1, 4, 9, 11):
            ax.text(
                x0, y0, z0 + 12 * Z_EXAGGERATION, point["name"],
                fontsize=6.5, color="#AD1457", fontweight="bold",
            )

    ax.set_xlim(0, SCENE_W_M)
    ax.set_ylim(0, SCENE_H_M)
    ax.set_zlim(0, SCENE_Z_M * Z_EXAGGERATION)
    ax.set_xlabel("E (km)", labelpad=8)
    ax.set_ylabel("N (km)", labelpad=8)
    ax.set_zlabel("高度 (m)", labelpad=7)
    ax.set_xticks(np.arange(0, SCENE_W_M + 1, 2000))
    ax.set_yticks(np.arange(0, SCENE_H_M + 1, 2000))
    ax.set_xticklabels(["%.0f" % (v / 1000) for v in np.arange(0, SCENE_W_M + 1, 2000)])
    ax.set_yticklabels(["%.0f" % (v / 1000) for v in np.arange(0, SCENE_H_M + 1, 2000)])
    zticks = np.arange(0, SCENE_Z_M + 1, 50)
    ax.set_zticks(zticks * Z_EXAGGERATION)
    ax.set_zticklabels(["%d" % v for v in zticks])
    ax.view_init(elev=31, azim=-60)
    try:
        ax.set_box_aspect((8, 6, 2.8))
    except AttributeError:
        pass
    ax.grid(True, alpha=0.3)
    ax.set_title(
        "武汉光谷无人机配送三维环境示意" if not compact else "三维环境示意",
        fontsize=16 if not compact else 13,
        fontweight="bold",
        pad=16,
    )
    if not compact:
        ax.text2D(
            0.5, 0.97,
            "地形、八类建筑、水域、风险体、配送点与分级路径（垂直方向 ×5）",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=9, color="#607D8B",
        )


def draw_height_heatmap(ax, scene, compact=False):
    add_zone_background(ax, labels=not compact)
    draw_terrain_2d(ax, alpha=0.16)
    draw_waters_2d(ax, scene, labels=not compact)
    draw_forests_2d(ax, scene, labels=False)
    norm, cmap = draw_buildings_2d(
        ax, scene, by_height=True, show_estimated=True
    )
    draw_delivery_points_2d(ax, scene, labels=not compact)
    add_map_frame(
        ax,
        "建筑高度热力图" if not compact else "建筑高度热力图",
        None if compact else "连续色带表示建筑高度；斜线建筑为程序化补全高度",
    )
    return norm, cmap


def legend_items():
    building_items = [
        (name, Patch(facecolor=color, edgecolor="#37474F"))
        for name, color in BUILDING_COLORS.items()
    ]
    environment_items = [
        ("水域", Patch(facecolor=WATER_COLOR, edgecolor=WATER_EDGE)),
        ("丘陵地形", Patch(facecolor="#C8B89A", edgecolor="#8D9A9E")),
        ("森林/树冠", Patch(facecolor=FOREST_COLOR, edgecolor="#1B5E20", alpha=0.55, hatch="..")),
        ("主干道", Line2D([0], [0], color=ROAD_MAIN, lw=7, markeredgecolor="#9E9E9E")),
        ("桥梁", Line2D([0], [0], color=BRIDGE_COLOR, lw=7)),
        ("高压线", Line2D([0], [0], color=POWER_COLOR, lw=2, linestyle="--")),
    ]
    risk_items = [
        (name, Patch(
            facecolor=mcolors.to_rgba(style[0], 0.14),
            edgecolor=style[0], hatch=style[1],
        ))
        for name, style in RISK_STYLES.items()
    ]
    task_items = [
        ("配送中心 P1", Line2D([0], [0], marker="*", color="none", markerfacecolor="#2E7D32",
                            markeredgecolor="#1B5E20", markersize=12)),
        ("配送点 P2–P12", Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
                               markeredgecolor="#D81B60", markersize=8)),
        ("规划路径", Line2D([0], [0], color=PATH_PLAN, lw=2.4)),
        ("执行路径", Line2D([0], [0], color=PATH_EXEC, lw=2.0, linestyle="--")),
        ("安全返航路径", Line2D([0], [0], color=PATH_RETURN, lw=2.0, linestyle=":")),
        ("动态对象", Line2D([0], [0], marker="D", color="none", markerfacecolor="#EC407A",
                           markeredgecolor="white", markersize=8)),
    ]
    return building_items, environment_items, risk_items, task_items


def add_legend_section(ax, x, y, width, title, items, columns=2, row_height=0.085):
    rows = int(math.ceil(len(items) / columns))
    height = 0.12 + rows * row_height
    box = FancyBboxPatch(
        (x, y - height), width, height,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        transform=ax.transAxes,
        facecolor="#FAFAFA", edgecolor="#CFD8DC",
        linewidth=1.1,
    )
    ax.add_patch(box)
    ax.text(
        x + 0.02, y - 0.045, title,
        transform=ax.transAxes, fontsize=11,
        fontweight="bold", color=TEXT_DARK, va="center",
    )
    col_width = (width - 0.04) / columns
    for idx, (label, handle) in enumerate(items):
        row = idx // columns
        col = idx % columns
        hx = x + 0.025 + col * col_width
        hy = y - 0.115 - row * row_height
        if isinstance(handle, Patch):
            rect = Rectangle(
                (hx, hy - 0.018), 0.038, 0.036,
                transform=ax.transAxes,
                facecolor=handle.get_facecolor(),
                edgecolor=handle.get_edgecolor(),
                hatch=handle.get_hatch(),
                linewidth=1.0,
            )
            ax.add_patch(rect)
        else:
            marker = handle.get_marker()
            if marker and marker != "None":
                ax.plot(
                    [hx + 0.019], [hy],
                    transform=ax.transAxes,
                    marker=marker,
                    markersize=handle.get_markersize(),
                    markerfacecolor=handle.get_markerfacecolor(),
                    markeredgecolor=handle.get_markeredgecolor(),
                    linestyle="none",
                )
            else:
                ax.plot(
                    [hx, hx + 0.04], [hy, hy],
                    transform=ax.transAxes,
                    color=handle.get_color(),
                    linewidth=handle.get_linewidth(),
                    linestyle=handle.get_linestyle(),
                )
        ax.text(
            hx + 0.052, hy, label,
            transform=ax.transAxes,
            fontsize=8.5, color=TEXT_DARK,
            va="center", ha="left",
        )
    return y - height - 0.025


def add_legend_card(ax, x, y, width, height, title, items, columns=2, compact=False):
    """在固定尺寸卡片中绘制图例，避免组合图中纵向溢出。"""
    box = FancyBboxPatch(
        (x, y - height), width, height,
        boxstyle="round,pad=0.010,rounding_size=0.015",
        transform=ax.transAxes,
        facecolor="#FAFAFA", edgecolor="#CFD8DC",
        linewidth=0.9 if compact else 1.1,
    )
    ax.add_patch(box)
    ax.text(
        x + 0.018, y - 0.045, title,
        transform=ax.transAxes,
        fontsize=9.2 if compact else 11,
        fontweight="bold", color=TEXT_DARK, va="center",
    )

    rows = int(math.ceil(len(items) / columns))
    usable_height = height - 0.105
    row_height = usable_height / max(rows, 1)
    col_width = (width - 0.035) / columns
    for idx, (label, handle) in enumerate(items):
        row = idx // columns
        col = idx % columns
        hx = x + 0.020 + col * col_width
        hy = y - 0.105 - row * row_height - row_height * 0.32
        symbol_w = 0.030 if compact else 0.036
        if isinstance(handle, Patch):
            rect = Rectangle(
                (hx, hy - 0.015), symbol_w, 0.030,
                transform=ax.transAxes,
                facecolor=handle.get_facecolor(),
                edgecolor=handle.get_edgecolor(),
                hatch=handle.get_hatch(),
                linewidth=0.8,
            )
            ax.add_patch(rect)
        else:
            marker = handle.get_marker()
            if marker and marker != "None":
                ax.plot(
                    [hx + symbol_w / 2], [hy],
                    transform=ax.transAxes,
                    marker=marker,
                    markersize=6.5 if compact else handle.get_markersize(),
                    markerfacecolor=handle.get_markerfacecolor(),
                    markeredgecolor=handle.get_markeredgecolor(),
                    linestyle="none",
                )
            else:
                ax.plot(
                    [hx, hx + symbol_w], [hy, hy],
                    transform=ax.transAxes,
                    color=handle.get_color(),
                    linewidth=max(1.3, handle.get_linewidth() * (0.72 if compact else 1.0)),
                    linestyle=handle.get_linestyle(),
                )
        ax.text(
            hx + symbol_w + 0.010, hy, label,
            transform=ax.transAxes,
            fontsize=6.6 if compact else 8.5,
            color=TEXT_DARK, va="center", ha="left",
        )


def draw_legend_panel(ax, compact=False):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    title_size = 14 if compact else 17
    ax.text(
        0.5, 0.975, "环境要素与障碍物图例",
        ha="center", va="top", fontsize=title_size,
        fontweight="bold", color=TEXT_DARK,
        transform=ax.transAxes,
    )
    if not compact:
        ax.text(
            0.5, 0.93,
            "颜色表达建筑类型；纹理表达风险语义；高度热力图另用连续色带",
            ha="center", va="top", fontsize=9,
            color="#607D8B", transform=ax.transAxes,
        )
    building_items, environment_items, risk_items, task_items = legend_items()
    top_y = 0.89 if not compact else 0.91
    card_h = 0.34 if not compact else 0.36
    lower_y = 0.47 if not compact else 0.48
    add_legend_card(
        ax, 0.025, top_y, 0.46, card_h,
        "八类建筑", building_items, columns=2, compact=compact,
    )
    add_legend_card(
        ax, 0.515, top_y, 0.46, card_h,
        "地表与静态障碍", environment_items, columns=2, compact=compact,
    )
    add_legend_card(
        ax, 0.025, lower_y, 0.46, card_h,
        "风险层", risk_items, columns=2, compact=compact,
    )
    add_legend_card(
        ax, 0.515, lower_y, 0.46, card_h,
        "任务与动态对象", task_items, columns=2, compact=compact,
    )
    ax.text(
        0.5, 0.055,
        "软风险不等于实体碰撞；硬禁飞区采用连续穿越检测。",
        ha="center", va="bottom", fontsize=8.5,
        color="#546E7A", transform=ax.transAxes,
    )


def save_figure(fig, output_dir, filename, dpi):
    path = output_dir / filename
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("已保存: %s" % path)
    return path


def plot_top_view(scene, output_dir, dpi):
    fig, ax = plt.subplots(figsize=(14, 10.8))
    draw_top_view(ax, scene, compact=False)
    handles = [
        Patch(facecolor=BUILDING_COLORS["商业综合体"], edgecolor="#37474F", label="建筑（按类型）"),
        Patch(facecolor=WATER_COLOR, edgecolor=WATER_EDGE, label="水域"),
        Patch(facecolor=FOREST_COLOR, edgecolor="#1B5E20", alpha=0.55, label="林地"),
        Line2D([0], [0], color=BRIDGE_COLOR, lw=6, label="桥梁"),
        Line2D([0], [0], color=POWER_COLOR, lw=2, linestyle="--", label="高压线"),
        Line2D([0], [0], color=PATH_PLAN, lw=2.4, label="规划路径"),
        Line2D([0], [0], color=PATH_EXEC, lw=2, linestyle="--", label="执行路径"),
    ]
    ax.legend(
        handles=handles, loc="lower right", ncol=2,
        framealpha=0.94, edgecolor="#B0BEC5", fontsize=8.5,
    )
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.08, top=0.91)
    schematic_footer(fig)
    return save_figure(fig, output_dir, "wuhan_zone_top_view.png", dpi)


def plot_3d(scene, output_dir, dpi):
    fig = plt.figure(figsize=(14.5, 10.8))
    ax = fig.add_subplot(111, projection="3d")
    draw_3d(ax, scene, compact=False)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.06, top=0.94)
    schematic_footer(fig)
    return save_figure(fig, output_dir, "wuhan_environment_3d.png", dpi)


def plot_height_heatmap(scene, output_dir, dpi):
    fig, ax = plt.subplots(figsize=(14, 10.8))
    norm, cmap = draw_height_heatmap(ax, scene, compact=False)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.025)
    cbar.set_label("建筑高度 (m)", fontweight="bold")
    estimated = Patch(
        facecolor="white", edgecolor="#37474F", hatch="///",
        label="程序化补全高度",
    )
    ax.legend(handles=[estimated], loc="lower right", framealpha=0.94, fontsize=9)
    add_cartographic_elements(ax)
    fig.subplots_adjust(left=0.08, right=0.92, bottom=0.08, top=0.91)
    schematic_footer(fig)
    return save_figure(fig, output_dir, "wuhan_building_height_heatmap.png", dpi)


def plot_legend(output_dir, dpi):
    fig, ax = plt.subplots(figsize=(13, 9.5))
    draw_legend_panel(ax, compact=False)
    schematic_footer(fig, "程序化脱敏示意图例 · 与场景绘图使用同一颜色和线型规范")
    fig.subplots_adjust(left=0.04, right=0.96, bottom=0.055, top=0.98)
    return save_figure(fig, output_dir, "wuhan_obstacle_legend.png", dpi)


def plot_overview(scene, output_dir, dpi):
    fig = plt.figure(figsize=(20, 14))
    grid = fig.add_gridspec(
        2, 2,
        width_ratios=[1.18, 1.0],
        height_ratios=[1.0, 1.0],
        hspace=0.20, wspace=0.13,
    )
    ax_top = fig.add_subplot(grid[0, 0])
    ax_3d = fig.add_subplot(grid[0, 1], projection="3d")
    ax_heat = fig.add_subplot(grid[1, 0])
    ax_legend = fig.add_subplot(grid[1, 1])

    draw_top_view(ax_top, scene, compact=True)
    draw_3d(ax_3d, scene, compact=True)
    norm, cmap = draw_height_heatmap(ax_heat, scene, compact=True)
    draw_legend_panel(ax_legend, compact=True)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_heat, fraction=0.035, pad=0.025)
    cbar.set_label("建筑高度 (m)", fontsize=9)

    fig.suptitle(
        "武汉光谷—南湖—东湖南缘无人机配送环境总览",
        fontsize=20, fontweight="bold", y=0.985,
    )
    fig.text(
        0.5, 0.958,
        "约 8 × 6 km 脱敏 ENU 程序化示意：四分区、八类建筑、静动态障碍、风险层与 12 个配送点",
        ha="center", va="center", fontsize=10, color="#607D8B",
    )
    schematic_footer(fig)
    fig.subplots_adjust(left=0.045, right=0.975, bottom=0.045, top=0.93)
    return save_figure(fig, output_dir, "wuhan_environment_overview.png", dpi)


def parse_args():
    script_dir = FsPath(__file__).resolve().parent
    default_output = script_dir.parents[1] / "outputs" / "环境示意图"
    parser = argparse.ArgumentParser(description="生成武汉光谷无人机配送环境示意图")
    parser.add_argument("--output-dir", type=FsPath, default=default_output)
    parser.add_argument("--dpi", type=int, default=260)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main():
    args = parse_args()
    apply_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scene = make_scene(seed=args.seed)

    print("场景模式: 程序化脱敏示意（非 GIS 统计）")
    print("随机种子: %d" % args.seed)
    print("示意建筑数: %d" % len(scene["buildings"]))
    print("配送点数: %d" % len(scene["deliveries"]))

    paths = [
        plot_3d(scene, args.output_dir, args.dpi),
        plot_top_view(scene, args.output_dir, args.dpi),
        plot_height_heatmap(scene, args.output_dir, args.dpi),
        plot_legend(args.output_dir, args.dpi),
        plot_overview(scene, args.output_dir, args.dpi),
    ]
    print("生成完成，共 %d 张图。" % len(paths))


if __name__ == "__main__":
    main()
