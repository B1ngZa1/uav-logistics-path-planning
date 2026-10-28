"""
场景热力图与建筑分布 / Scene Heatmap and Building Distribution

职责 / Responsibility: 生成四区建筑分布热力图、任务点标注图、分区统计图
参考 / Reference: scene_2d_manifest.json 真实场景数据 + matplotlib
领域适配 / Domain Adaptation: 武汉四区OSM场景数据→无人机配送环境可视化

使用方式 / Usage:
    python plot_scene_heatmap.py
    python plot_scene_heatmap.py --manifest ../outputs/环境示意图_2D审核/scene_2d_manifest.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# ============================================================
# 路径设置 / Path setup
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "04_脚本图片输出" / "科研图表"
MANIFEST_PATH = PROJECT_ROOT / "outputs" / "环境示意图_2D审核" / "scene_2d_manifest.json"


def setup_plot_style():
    """配置全局绘图风格 / Configure global plot style"""
    font_candidates = ["SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei", "DejaVu Sans"]
    available_fonts = [f.name for f in matplotlib.font_manager.fontManager.ttflist]
    chosen_font = "DejaVu Sans"
    for candidate in font_candidates:
        if candidate in available_fonts:
            chosen_font = candidate
            break

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [chosen_font, "DejaVu Sans", "Arial"],
        "font.size": 11,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 0.8,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        "axes.unicode_minus": False,
    })


# ============================================================
# 数据加载 / Data loading
# ============================================================

def load_manifest(manifest_path: Path) -> dict:
    """加载场景清单 / Load scene manifest"""
    with open(str(manifest_path), "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# 模拟建筑位置数据 / Simulated building position data
# ============================================================

def generate_building_positions(zone: dict, num_buildings: int = None) -> np.ndarray:
    """根据场景清单生成建筑分布点（模拟）。

    / Generate building distribution points based on scene manifest (simulated).

    使用核密度估计的思想：在场景范围内随机撒点，
    建筑密度与 feature_counts.building 成正比。

    Args:
        zone: 场景清单中的zone数据 / Zone data from manifest
        num_buildings: 建筑数量（默认从清单读取）

    Returns:
        (N, 2) 数组，建筑中心坐标 / (N, 2) array of building centers
    """
    if num_buildings is None:
        num_buildings = zone["feature_counts"].get("building", 100)

    extent = zone["extent_m"]
    rng = np.random.RandomState(hash(zone["zone_id"]) % 2**31)

    # 模拟建筑分布：中心密集，边缘稀疏
    # / Simulated distribution: dense center, sparse edges
    center_x, center_y = extent[0] / 2, extent[1] / 2

    # 使用高斯混合模拟建筑群 / Gaussian mixture for building clusters
    n_clusters = max(3, num_buildings // 80)
    cluster_centers = rng.uniform(0.2, 0.8, (n_clusters, 2)) * np.array(extent)
    cluster_stds = rng.uniform(0.05, 0.15, (n_clusters, 2)) * np.array(extent)

    points = []
    points_per_cluster = num_buildings // n_clusters
    for i in range(n_clusters):
        n = points_per_cluster if i < n_clusters - 1 else num_buildings - len(points)
        cx, cy = cluster_centers[i]
        sx, sy = cluster_stds[i]
        px = rng.normal(cx, sx, n)
        py = rng.normal(cy, sy, n)
        # 裁剪到场景范围 / Clip to extent
        px = np.clip(px, 0, extent[0])
        py = np.clip(py, 0, extent[1])
        points.append(np.column_stack([px, py]))

    return np.vstack(points)


# ============================================================
# 图表4a: 四区建筑分布热力图 / Building density heatmap (4 zones)
# ============================================================

def plot_building_heatmap(manifest: dict, output_dir: Path):
    """生成2x2热力图面板，展示四区建筑密度分布。

    / Generate 2x2 heatmap panel showing building density across 4 zones.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # 自定义色图 / Custom colormap
    cmap = LinearSegmentedColormap.from_list(
        "building_density",
        ["#f7fbff", "#6baed6", "#2171b5", "#08306b"],
        N=256,
    )

    zone_colors = {"A": "#e41a1c", "B": "#377eb8", "C": "#4daf4a", "D": "#984ea3"}

    for idx, zone in enumerate(manifest["zones"]):
        ax = axes[idx // 2, idx % 2]
        zone_id = zone["zone_id"]
        extent = zone["extent_m"]

        # 生成建筑位置 / Generate building positions
        buildings = generate_building_positions(zone)

        # 2D直方图热力图 / 2D histogram heatmap
        hb = ax.hexbin(
            buildings[:, 0], buildings[:, 1],
            gridsize=30, cmap=cmap, mincnt=1,
            extent=(0, extent[0], 0, extent[1]),
        )

        # 标注任务点 / Mark task points
        for tp in zone["task_points"]:
            tx, ty = tp["local_enu_m"]
            marker = "^" if tp["site_type"] == "配送中心" or tp["site_type"] == "校园后勤点" else "o"
            color = zone_colors[zone_id]
            ax.plot(tx, ty, marker=marker, color=color, markersize=14,
                    markeredgecolor="white", markeredgewidth=2, zorder=10)
            ax.annotate(f"T{tp['id']}", xy=(tx, ty), fontsize=9, fontweight="bold",
                        color="white", ha="center", va="center", zorder=11,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.8))

        ax.set_xlim(0, extent[0])
        ax.set_ylim(0, extent[1])
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(
            f"分区 {zone_id}: {zone['title']}\n"
            f"Zone {zone_id}: {zone['feature_counts']['building']} buildings, "
            f"{len(zone['task_points'])} task points",
            fontsize=12,
        )
        ax.set_aspect("equal")

        # 色条 / Color bar
        cb = fig.colorbar(hb, ax=ax, shrink=0.8, pad=0.02)
        cb.set_label("建筑密度 / Building Density", fontsize=9)

    plt.suptitle(
        "武汉四区建筑分布热力图\nWuhan Four-Zone Building Distribution Heatmap",
        fontsize=16, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    _save(output_dir, "fig4a_building_heatmap")
    plt.close(fig)
    print("  [4a] 四区热力图已生成")


# ============================================================
# 图表4b: 分区统计对比柱状图 / Zone statistics comparison
# ============================================================

def plot_zone_statistics(manifest: dict, output_dir: Path):
    """生成四区要素统计对比图 / Generate four-zone feature statistics comparison"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    zones = manifest["zones"]
    zone_ids = [z["zone_id"] for z in zones]
    zone_labels = [f"分区{z['zone_id']}\nZone {z['zone_id']}" for z in zones]
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
    x = np.arange(len(zones))

    # 1: 建筑数量 / Building counts
    ax = axes[0]
    building_counts = [z["feature_counts"]["building"] for z in zones]
    bars = ax.bar(x, building_counts, 0.6, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(zone_labels, fontsize=10)
    ax.set_ylabel("建筑数量 / Building Count")
    ax.set_title("(a) 建筑数量对比 / Building Count Comparison")
    for bar, val in zip(bars, building_counts):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 10,
                str(val), ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.25, axis="y")

    # 2: 场景面积 / Scene area
    ax = axes[1]
    areas = [z["extent_m"][0] * z["extent_m"][1] / 1e6 for z in zones]  # km²
    bars = ax.bar(x, areas, 0.6, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(zone_labels, fontsize=10)
    ax.set_ylabel("场景面积 (km²) / Scene Area (km²)")
    ax.set_title("(b) 场景面积对比 / Scene Area Comparison")
    for bar, val in zip(bars, areas):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.1,
                f"{val:.1f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.25, axis="y")

    # 3: 建筑密度 / Building density (buildings/km²)
    ax = axes[2]
    densities = [b / a for b, a in zip(building_counts, areas)]
    bars = ax.bar(x, densities, 0.6, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(zone_labels, fontsize=10)
    ax.set_ylabel("建筑密度 (栋/km²) / Density (bld/km²)")
    ax.set_title("(c) 建筑密度对比 / Building Density Comparison")
    for bar, val in zip(bars, densities):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 5,
                f"{val:.0f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.25, axis="y")

    plt.suptitle(
        "四区场景特征统计对比\nFour-Zone Scene Feature Statistics Comparison",
        fontsize=16, fontweight="bold", y=1.03,
    )
    plt.tight_layout()
    _save(output_dir, "fig4b_zone_statistics")
    plt.close(fig)
    print("  [4b] 分区统计图已生成")


# ============================================================
# 图表4c: 风险类型与任务点分布 / Risk types and task point distribution
# ============================================================

def plot_risk_and_tasks(manifest: dict, output_dir: Path):
    """生成风险类型和任务点分布图 / Generate risk type and task point distribution"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    zones = manifest["zones"]

    # 左图：各分区风险类型矩阵 / Left: Risk type matrix
    ax = axes[0]
    all_risks = set()
    for z in zones:
        all_risks.update(z["risk_types"])
    risk_list = sorted(all_risks)

    risk_matrix = np.zeros((len(zones), len(risk_list)))
    for i, z in enumerate(zones):
        for j, r in enumerate(risk_list):
            if r in z["risk_types"]:
                risk_matrix[i, j] = 1

    # 绘制矩阵热力图 / Draw matrix heatmap
    cmap = LinearSegmentedColormap.from_list("risk", ["#ffffff", "#d62728"], N=2)
    im = ax.imshow(risk_matrix, cmap=cmap, aspect="auto", interpolation="nearest")

    ax.set_xticks(range(len(risk_list)))
    ax.set_xticklabels(risk_list, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(len(zones)))
    ax.set_yticklabels([f"分区{z['zone_id']}" for z in zones], fontsize=11)
    ax.set_title("(a) 各分区风险类型分布\nRisk Type Distribution by Zone", fontsize=13)

    # 添加标记 / Add markers
    for i in range(len(zones)):
        for j in range(len(risk_list)):
            if risk_matrix[i, j] > 0:
                ax.text(j, i, "●", ha="center", va="center",
                        color="white", fontsize=14, fontweight="bold")

    # 右图：任务点类型分布 / Right: Task point type distribution
    ax = axes[1]
    task_types = {}
    for z in zones:
        for tp in z["task_points"]:
            t = tp["site_type"]
            task_types[t] = task_types.get(t, 0) + 1

    type_names = list(task_types.keys())
    type_counts = list(task_types.values())
    type_colors = plt.cm.Set3(np.linspace(0, 1, len(type_names)))

    wedges, texts, autotexts = ax.pie(
        type_counts, labels=None, autopct="%1.0f%%",
        colors=type_colors, startangle=90,
        pctdistance=0.8, textprops={"fontsize": 10},
    )
    ax.legend(wedges, type_names, loc="center left", bbox_to_anchor=(1, 0.5),
              fontsize=9, framealpha=0.9)
    ax.set_title("(b) 任务点类型分布\nTask Point Type Distribution", fontsize=13)

    plt.suptitle(
        "风险与任务点分布分析\nRisk and Task Point Distribution Analysis",
        fontsize=16, fontweight="bold", y=1.03,
    )
    plt.tight_layout()
    _save(output_dir, "fig4c_risk_and_tasks")
    plt.close(fig)
    print("  [4c] 风险与任务点分布图已生成")


# ============================================================
# 工具函数 / Utility
# ============================================================

def _save(output_dir: Path, name: str):
    """保存PNG和PDF / Save PNG and PDF"""
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{name}.png"
    plt.savefig(str(png_path), dpi=300, bbox_inches="tight", pad_inches=0.1)
    print(f"    PNG: {png_path}")
    pdf_path = output_dir / f"{name}.pdf"
    plt.savefig(str(pdf_path), format="pdf", bbox_inches="tight", pad_inches=0.1)
    print(f"    PDF: {pdf_path}")


# ============================================================
# 主入口 / Main entry
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="场景热力图与建筑分布 / Scene Heatmap and Building Distribution"
    )
    parser.add_argument("--manifest", type=str, default=None,
                        help="场景清单JSON路径 / Scene manifest JSON path")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="输出目录 / Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    manifest_path = Path(args.manifest) if args.manifest else MANIFEST_PATH

    setup_plot_style()

    if manifest_path.exists():
        print(f"加载场景清单 / Loading manifest: {manifest_path}")
        manifest = load_manifest(manifest_path)
    else:
        print(f"警告：未找到场景清单，使用内置数据 / Warning: manifest not found, using built-in data")
        manifest = _builtin_manifest()

    print(f"\n输出目录 / Output directory: {output_dir}\n")

    plot_building_heatmap(manifest, output_dir)
    plot_zone_statistics(manifest, output_dir)
    plot_risk_and_tasks(manifest, output_dir)

    print(f"\n所有图表已保存到 / All figures saved to: {output_dir}")
    print("完成! / Done!")


def _builtin_manifest() -> dict:
    """内置场景数据（当无法读取JSON时使用）/ Built-in scene data fallback"""
    return {
        "zones": [
            {
                "zone_id": "A", "title": "分区 A｜环岛高密核心区",
                "extent_m": [2400.0, 1800.0],
                "feature_counts": {"building": 837, "road": 499, "green": 40, "railway": 14, "water": 4},
                "task_points": [
                    {"id": 1, "site_type": "配送中心", "local_enu_m": [408.0, 324.0]},
                    {"id": 2, "site_type": "商业裙楼屋顶", "local_enu_m": [1149.4, 1073.39]},
                    {"id": 3, "site_type": "商业综合体平台", "local_enu_m": [2012.57, 490.92]},
                ],
                "risk_types": ["GPS多路径", "动态交通", "轨道硬隔离", "通信阴影"],
            },
            {
                "zone_id": "B", "title": "分区 B｜湖滨校园生态区",
                "extent_m": [2400.0, 1800.0],
                "feature_counts": {"building": 43, "road": 56, "green": 8, "water": 8},
                "task_points": [
                    {"id": 4, "site_type": "校园后勤点", "local_enu_m": [1789.52, 588.58]},
                    {"id": 5, "site_type": "公共建筑屋顶", "local_enu_m": [945.27, 1010.52]},
                    {"id": 6, "site_type": "湖岸维护设施", "local_enu_m": [1160.0, 990.0]},
                ],
                "risk_types": ["湖面侧风", "飞鸟活动", "人群密集", "通信阴影"],
            },
            {
                "zone_id": "C", "title": "分区 C｜跨江桥头复合区",
                "extent_m": [2800.0, 2000.0],
                "feature_counts": {"building": 538, "road": 494, "green": 63, "railway": 20, "water": 5},
                "task_points": [
                    {"id": 7, "site_type": "桥头服务设施", "local_enu_m": [2138.54, 1188.99]},
                    {"id": 8, "site_type": "老城配送平台", "local_enu_m": [2363.41, 438.23]},
                    {"id": 9, "site_type": "滨水公共设施", "local_enu_m": [1929.2, 463.26]},
                ],
                "risk_types": ["江面阵风", "桥体硬隔离", "船舶走廊", "人群密集", "通信阴影"],
            },
            {
                "zone_id": "D", "title": "分区 D｜窄河水岸商办区",
                "extent_m": [2400.0, 1800.0],
                "feature_counts": {"building": 592, "road": 830, "green": 65, "railway": 14, "water": 8},
                "task_points": [
                    {"id": 10, "site_type": "商业裙楼屋顶", "local_enu_m": [714.39, 1047.26]},
                    {"id": 11, "site_type": "办公塔楼平台", "local_enu_m": [1248.84, 642.43]},
                    {"id": 12, "site_type": "社区公共建筑", "local_enu_m": [2027.94, 688.83]},
                ],
                "risk_types": ["河道风廊", "GPS多路径", "通信阴影", "动态交通", "临时施工"],
            },
        ]
    }


if __name__ == "__main__":
    main()
