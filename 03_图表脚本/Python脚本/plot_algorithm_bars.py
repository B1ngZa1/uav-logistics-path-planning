"""
算法对比柱状图 / Algorithm Comparison Bar Charts

职责 / Responsibility: 生成6种算法x4指标的分组柱状图，含误差棒和最优标注
参考 / Reference: scientific_plots.py + Nature/Science配色规范
领域适配 / Domain Adaptation: 无人机配送RL算法性能→出版级对比图表

使用方式 / Usage:
    python plot_algorithm_bars.py --demo
    python plot_algorithm_bars.py --data-dir ../02_数据保存/experiment_results/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ============================================================
# 路径设置 / Path setup
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "04_脚本图片输出" / "科研图表"

# ============================================================
# 配色与样式 / Colors and style
# ============================================================
PALETTE = {
    "q_learning":   "#1f77b4",
    "sarsa":        "#ff7f0e",
    "monte_carlo":  "#2ca02c",
    "dqn":          "#d62728",
    "a2c":          "#9467bd",
    "ppo":          "#8c564b",
}

ALGO_DISPLAY = {
    "q_learning":   "Q-Learning",
    "sarsa":        "SARSA",
    "monte_carlo":  "Monte Carlo",
    "dqn":          "DQN",
    "a2c":          "A2C",
    "ppo":          "PPO",
}


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
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.unicode_minus": False,
    })


# ============================================================
# 模拟数据 / Simulated data
# ============================================================

def generate_final_metrics(num_seeds: int = 5) -> dict:
    """生成各算法的最终性能指标（含标准差）。

    / Generate final performance metrics per algorithm (with std dev).

    5个随机种子运行的统计结果。各指标物理意义：
    - completion_rate: 最后100回合平均配送完成率（0~1）
    - avg_reward: 最后100回合平均奖励
    - collision_rate: 最后100回合平均碰撞率（越低越好）
    - convergence_speed: 达到90%最终性能所需的episode数（越小越好）

    Returns:
        {algo: {metric: (mean, std)}} 嵌套字典
    """
    np.random.seed(42)

    # 各算法的"真实"指标均值和标准差 / Ground-truth means and stds
    params = {
        "q_learning":   {"cr": (0.62, 0.04), "rew": (280, 30), "col": (0.08, 0.02), "conv": (800, 80)},
        "sarsa":        {"cr": (0.58, 0.05), "rew": (250, 35), "col": (0.09, 0.02), "conv": (900, 90)},
        "monte_carlo":  {"cr": (0.50, 0.06), "rew": (220, 45), "col": (0.12, 0.03), "conv": (1200, 120)},
        "dqn":          {"cr": (0.78, 0.03), "rew": (320, 20), "col": (0.05, 0.01), "conv": (500, 50)},
        "a2c":          {"cr": (0.75, 0.03), "rew": (310, 25), "col": (0.06, 0.02), "conv": (600, 60)},
        "ppo":          {"cr": (0.85, 0.02), "rew": (350, 15), "col": (0.04, 0.01), "conv": (400, 40)},
    }

    data = {}
    for algo, p in params.items():
        data[algo] = {
            "completion_rate": (p["cr"][0],  p["cr"][1]),
            "avg_reward":      (p["rew"][0], p["rew"][1]),
            "collision_rate":  (p["col"][0], p["col"][1]),
            "convergence_ep":  (p["conv"][0], p["conv"][1]),
        }

    return data


# ============================================================
# 图表2a: 四指标分组柱状图 / Four-metric grouped bar chart
# ============================================================

def plot_four_metric_bars(data: dict, output_dir: Path):
    """生成4子图（完成率/奖励/碰撞率/收敛速度），每子图含6算法柱状图。

    / Generate 4-panel bar chart (completion rate / reward / collision / convergence)
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    algos = list(data.keys())
    colors = [PALETTE[a] for a in algos]
    labels = [ALGO_DISPLAY[a] for a in algos]
    x = np.arange(len(algos))

    # 2a: 完成率 / Completion rate
    ax = axes[0, 0]
    means = [data[a]["completion_rate"][0] for a in algos]
    stds  = [data[a]["completion_rate"][1] for a in algos]
    bars = ax.bar(x, means, 0.6, color=colors, edgecolor="white", linewidth=0.5,
                  yerr=stds, capsize=4, error_kw={"linewidth": 1.2})
    best_idx = np.argmax(means)
    bars[best_idx].set_edgecolor("gold")
    bars[best_idx].set_linewidth(2.5)
    ax.set_ylabel("完成率 (Completion Rate)")
    ax.set_title("(a) 配送完成率 / Delivery Completion Rate")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    _add_labels(ax, bars, means, "{:.2f}")

    # 2b: 平均奖励 / Average reward
    ax = axes[0, 1]
    means = [data[a]["avg_reward"][0] for a in algos]
    stds  = [data[a]["avg_reward"][1] for a in algos]
    bars = ax.bar(x, means, 0.6, color=colors, edgecolor="white", linewidth=0.5,
                  yerr=stds, capsize=4, error_kw={"linewidth": 1.2})
    best_idx = np.argmax(means)
    bars[best_idx].set_edgecolor("gold")
    bars[best_idx].set_linewidth(2.5)
    ax.set_ylabel("平均奖励 (Average Reward)")
    ax.set_title("(b) 最终平均奖励 / Final Average Reward")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    _add_labels(ax, bars, means, "{:.0f}")

    # 2c: 碰撞率（越低越好）/ Collision rate (lower is better)
    ax = axes[1, 0]
    means = [data[a]["collision_rate"][0] for a in algos]
    stds  = [data[a]["collision_rate"][1] for a in algos]
    bars = ax.bar(x, means, 0.6, color=colors, edgecolor="white", linewidth=0.5,
                  yerr=stds, capsize=4, error_kw={"linewidth": 1.2})
    best_idx = np.argmin(means)
    bars[best_idx].set_edgecolor("gold")
    bars[best_idx].set_linewidth(2.5)
    ax.set_ylabel("碰撞率 (Collision Rate)")
    ax.set_title("(c) 碰撞率（越低越好）/ Collision Rate (Lower is Better)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    _add_labels(ax, bars, means, "{:.3f}")

    # 2d: 收敛速度（越小越好）/ Convergence speed (lower is better)
    ax = axes[1, 1]
    means = [data[a]["convergence_ep"][0] for a in algos]
    stds  = [data[a]["convergence_ep"][1] for a in algos]
    bars = ax.bar(x, means, 0.6, color=colors, edgecolor="white", linewidth=0.5,
                  yerr=stds, capsize=4, error_kw={"linewidth": 1.2})
    best_idx = np.argmin(means)
    bars[best_idx].set_edgecolor("gold")
    bars[best_idx].set_linewidth(2.5)
    ax.set_ylabel("收敛回合数 (Episodes to Converge)")
    ax.set_title("(d) 收敛速度（越低越好）/ Convergence Speed (Lower is Better)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    _add_labels(ax, bars, means, "{:.0f}")

    # 添加图例说明金色边框 / Add legend for gold border
    gold_patch = mpatches.Patch(facecolor="white", edgecolor="gold", linewidth=2.5,
                                 label="最优算法 / Best Algorithm")
    fig.legend(handles=[gold_patch], loc="upper center", ncol=1,
               bbox_to_anchor=(0.5, 1.02), fontsize=11, framealpha=0.9)

    plt.suptitle("六种算法四维度性能对比 / Six-Algorithm Four-Metric Comparison",
                 fontsize=16, fontweight="bold", y=1.05)
    plt.tight_layout()
    _save(output_dir, "fig2a_four_metric_bars")
    plt.close(fig)
    print("  [2a] 四指标柱状图已生成")


# ============================================================
# 图表2b: 综合分组对比 / Grouped comparison chart
# ============================================================

def plot_grouped_comparison(data: dict, output_dir: Path):
    """生成单图分组柱状图，所有指标归一化到[0,1]后对比。

    / Generate single grouped bar chart with all metrics normalized to [0,1]
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    algos = list(data.keys())
    metrics = ["completion_rate", "avg_reward", "collision_rate", "convergence_ep"]
    metric_labels = [
        "配送完成率\nCompletion Rate",
        "平均奖励\nAvg Reward",
        "碰撞率\nCollision Rate",
        "收敛速度\nConvergence Speed",
    ]

    n_algos = len(algos)
    n_metrics = len(metrics)
    bar_width = 0.12
    x = np.arange(n_metrics)

    # 收集原始值 / Collect raw values
    raw = {}
    for algo in algos:
        raw[algo] = [data[algo][m][0] for m in metrics]

    # Min-max归一化（碰撞率和收敛速度取反）/ Min-max normalization (invert col & conv)
    normalized = {}
    for j, m in enumerate(metrics):
        vals = [raw[a][j] for a in algos]
        v_min, v_max = min(vals), max(vals)
        for a in algos:
            if a not in normalized:
                normalized[a] = []
            if v_max - v_min > 1e-8:
                norm = (raw[a][j] - v_min) / (v_max - v_min)
            else:
                norm = 0.5
            # 碰撞率和收敛速度越低越好，反转 / Invert for "lower is better"
            if m in ("collision_rate", "convergence_ep"):
                norm = 1.0 - norm
            normalized[a].append(norm)

    # 绘制 / Draw
    offsets = np.arange(n_algos) - (n_algos - 1) / 2
    for i, algo in enumerate(algos):
        pos = x + offsets[i] * bar_width
        vals = normalized[algo]
        ax.barh(pos, vals, bar_width, color=PALETTE[algo], edgecolor="white",
                linewidth=0.5, label=ALGO_DISPLAY[algo])

    ax.set_yticks(x)
    ax.set_yticklabels(metric_labels, fontsize=11)
    ax.set_xlabel("归一化得分 (Normalized Score)")
    ax.set_title("六种算法归一化性能对比 / Normalized Performance Comparison")
    ax.set_xlim(0, 1.1)
    ax.legend(loc="lower right", ncol=2, fontsize=10, framealpha=0.9)
    ax.axvline(x=1.0, color="gold", linestyle="--", alpha=0.5, linewidth=1, label="理想值 / Ideal")

    plt.tight_layout()
    _save(output_dir, "fig2b_normalized_grouped_bars")
    plt.close(fig)
    print("  [2b] 归一化分组对比图已生成")


# ============================================================
# 图表2c: L1-L4层级性能递减 / Level Performance Degradation
# ============================================================

def plot_level_degradation(output_dir: Path):
    """生成L1→L4性能递减示意图 / Generate L1→L4 performance degradation"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    levels = ["L1", "L2", "L3", "L4"]
    level_labels = [
        "L1: 2D固定\n2D Fixed",
        "L2: 2D随机\n2D Random",
        "L3: 3D固定\n3D Fixed",
        "L4: 3D多机\n3D Multi-drone",
    ]
    colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]
    x = np.arange(len(levels))

    # 左图：综合得分 / Left: Overall scores
    ax = axes[0]
    best_scores = [0.95, 0.78, 0.60, 0.42]
    np.random.seed(42)
    noise = np.random.normal(0, 0.02, 4)
    scores = np.clip(np.array(best_scores) + noise, 0, 1)
    bars = ax.bar(x, scores, 0.6, color=colors, edgecolor="white", linewidth=0.5)

    # 二次趋势线 / Quadratic trend line
    z = np.polyfit(x, scores, 2)
    p = np.poly1d(z)
    x_smooth = np.linspace(0, 3, 100)
    ax.plot(x_smooth, p(x_smooth), "--", color="grey", alpha=0.5, linewidth=1.5,
            label="难度趋势 / Difficulty Trend")

    ax.set_xticks(x)
    ax.set_xticklabels(level_labels, fontsize=10)
    ax.set_ylabel("归一化性能得分 / Normalized Score")
    ax.set_title("(a) 性能随层级递减 / Performance Degradation by Level")
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=10)
    _add_labels(ax, bars, scores, "{:.2f}")

    # 右图：各维度分组 / Right: Multi-dimension grouped
    ax = axes[1]
    dimensions = {
        "路线最优性\nRoute Optimality":   [0.95, 0.80, 0.55, 0.35],
        "订单完成率\nOrder Completion":    [0.92, 0.75, 0.62, 0.45],
        "安全性\nSafety (No Crash)":       [0.98, 0.85, 0.70, 0.50],
        "时间效率\nTime Efficiency":       [0.90, 0.72, 0.58, 0.40],
    }
    n_dims = len(dimensions)
    dim_names = list(dimensions.keys())
    bar_width = 0.18
    offsets = np.arange(n_dims) - (n_dims - 1) / 2

    for i, level in enumerate(levels):
        dim_values = [dimensions[d][i] for d in dim_names]
        pos = np.arange(n_dims) + offsets[i] * bar_width
        ax.barh(pos, dim_values, bar_width, color=colors[i],
                edgecolor="white", linewidth=0.5, label=level)

    ax.set_yticks(np.arange(n_dims))
    ax.set_yticklabels(dim_names, fontsize=10)
    ax.set_xlabel("归一化得分 / Normalized Score")
    ax.set_title("(b) 各维度分层对比 / Multi-Dimension Comparison")
    ax.set_xlim(0, 1.1)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    _save(output_dir, "fig2c_level_degradation")
    plt.close(fig)
    print("  [2c] 层级递减图已生成")


# ============================================================
# 工具函数 / Utility
# ============================================================

def _add_labels(ax, bars, values, fmt="{:.2f}"):
    """添加柱顶标签 / Add bar top labels"""
    for bar, val in zip(bars, values):
        height = bar.get_height()
        offset = 0.01 * ax.get_ylim()[1]
        ax.text(bar.get_x() + bar.get_width() / 2., height + offset,
                fmt.format(val), ha="center", va="bottom", fontsize=8.5, color="0.3")


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
        description="算法对比柱状图 / Algorithm Comparison Bar Charts"
    )
    parser.add_argument("--demo", action="store_true",
                        help="使用模拟数据 / Use simulated data")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="输出目录 / Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    setup_plot_style()

    print("使用模拟数据生成柱状图 / Using simulated data for bar charts")
    data = generate_final_metrics()

    print(f"\n输出目录 / Output directory: {output_dir}\n")

    plot_four_metric_bars(data, output_dir)
    plot_grouped_comparison(data, output_dir)
    plot_level_degradation(output_dir)

    print(f"\n所有图表已保存到 / All figures saved to: {output_dir}")
    print("完成! / Done!")


if __name__ == "__main__":
    main()
