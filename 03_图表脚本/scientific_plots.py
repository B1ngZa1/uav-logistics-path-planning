"""
科研绘图脚本 / Scientific Publication Plotting Script

职责 / Responsibility: 生成SCI/Nature标准的高质量实验图表，支持从CSV读取
    真实数据或生成模拟演示数据。
参考 / Reference: matplotlib + seaborn best practices, SCI/Nature guidelines
领域适配 / Domain Adaptation: 无人机配送实验指标→出版级图表

使用方式 / Usage:
    # 从真实数据生成图表
    python scientific_plots.py --data-dir 02_数据保存/experiment_results/

    # 使用模拟数据演示
    python scientific_plots.py --demo

    # 指定输出目录
    python scientific_plots.py --demo --output-dir 04_脚本图片输出/科研图表/

    # 仅生成指定图表
    python scientific_plots.py --demo --figures convergence,bar,radar,levels
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # 无头渲染 / Headless rendering

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ============================================================
# 路径设置 / Path setup
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # 无人机项目设计/

# ============================================================
# 全局绘图配置 / Global plot configuration
# ============================================================

# 色盲友好调色板（tab10扩展）/ Colorblind-friendly palette
PALETTE = {
    "q_learning":   "#1f77b4",   # 蓝 / Blue
    "sarsa":        "#ff7f0e",   # 橙 / Orange
    "monte_carlo":  "#2ca02c",   # 绿 / Green
    "dqn":          "#d62728",   # 红 / Red
    "a2c":          "#9467bd",   # 紫 / Purple
    "ppo":          "#8c564b",   # 棕 / Brown
}

# 算法显示名称 / Algorithm display names
ALGO_DISPLAY = {
    "q_learning":   "Q-Learning",
    "sarsa":        "SARSA",
    "monte_carlo":  "Monte Carlo",
    "dqn":          "DQN",
    "a2c":          "A2C",
    "ppo":          "PPO",
}

# 指标显示名称 / Metric display names
METRIC_DISPLAY = {
    "completion_rate":  "Completion Rate",
    "avg_reward":       "Average Reward",
    "collision_rate":   "Collision Rate",
    "convergence_speed": "Convergence Speed",
}

# 层级显示名称 / Level display names
LEVEL_DISPLAY = {
    "L1": "L1: 2D Fixed",
    "L2": "L2: 2D Random",
    "L3": "L3: 3D Fixed",
    "L4": "L4: 3D Multi-drone",
}


def setup_plot_style():
    """配置全局绘图风格（SCI标准）。

    / Configure global plot style (SCI standard).
    """
    # 中文字体降级策略 / Chinese font fallback chain
    font_candidates = ["SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei", "DejaVu Sans"]
    available_fonts = [f.name for f in matplotlib.font_manager.fontManager.ttflist]
    chosen_font = "DejaVu Sans"
    for candidate in font_candidates:
        if candidate in available_fonts:
            chosen_font = candidate
            break

    plt.rcParams.update({
        # 字体 / Font
        "font.family": "sans-serif",
        "font.sans-serif": [chosen_font, "DejaVu Sans", "Arial"],
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,

        # 线条与标记 / Lines and markers
        "lines.linewidth": 1.8,
        "lines.markersize": 5,

        # 坐标轴 / Axes
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,

        # 图例 / Legend
        "legend.framealpha": 0.8,
        "legend.edgecolor": "0.8",

        # 保存 / Save
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,

        # 风格 / Style
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    # 尝试设置seaborn风格 / Try setting seaborn style
    try:
        import seaborn as sns
        sns.set_style("whitegrid", {
            "grid.color": "0.85",
            "grid.linewidth": 0.5,
        })
    except ImportError:
        pass  # seaborn不可用时使用默认风格 / Use default style


# ============================================================
# 数据加载与模拟 / Data loading and simulation
# ============================================================

def generate_demo_data(num_algos: int = 6, num_episodes: int = 2000
                       ) -> Dict[str, Dict[str, np.ndarray]]:
    """生成模拟训练数据（用于演示）。

    / Generate simulated training data (for demonstration).

    不同算法使用不同的学习曲线特征模拟真实行为：
    - Q-learning/SARSA: 中等收敛速度，较高方差
    - Monte Carlo: 慢收敛，低方差
    - DQN/A2C/PPO: 快收敛，低方差

    Returns:
        {algo: {metric: np.ndarray}} 嵌套字典 / Nested dict
    """
    np.random.seed(42)
    algos = ["q_learning", "sarsa", "monte_carlo", "dqn", "a2c", "ppo"][:num_algos]
    episodes = np.arange(1, num_episodes + 1)

    # 每个算法的收敛特征 / Convergence characteristics per algo
    algo_params = {
        "q_learning":   {"asymptote": 280, "rate": 0.003, "noise": 80,  "conv_ep": 800},
        "sarsa":        {"asymptote": 250, "rate": 0.0025,"noise": 70,  "conv_ep": 900},
        "monte_carlo":  {"asymptote": 220, "rate": 0.002, "noise": 100, "conv_ep": 1200},
        "dqn":          {"asymptote": 320, "rate": 0.005, "noise": 50,  "conv_ep": 500},
        "a2c":          {"asymptote": 310, "rate": 0.004, "noise": 60,  "conv_ep": 600},
        "ppo":          {"asymptote": 350, "rate": 0.006, "noise": 40,  "conv_ep": 400},
    }

    data = {}
    for algo in algos:
        params = algo_params.get(algo, algo_params["q_learning"])
        # 收敛曲线 = 初始值 + (渐近值 - 初始值) * (1 - exp(-rate * ep))
        # / Convergence curve = init + (asymptote - init) * (1 - exp(-rate * ep))
        base_reward = -100 + (params["asymptote"] + 100) * (
            1 - np.exp(-params["rate"] * episodes)
        )
        # 加入噪声 / Add noise
        noise_scale = params["noise"] * np.exp(-0.001 * episodes) + params["noise"] * 0.2
        reward = base_reward + np.random.normal(0, 1, num_episodes) * noise_scale

        # 完成率 / Completion rate
        base_cr = 0.1 + 0.8 * (1 - np.exp(-params["rate"] * 1.2 * episodes))
        cr_noise = 0.05 * np.exp(-0.002 * episodes) + 0.02
        completion_rate = np.clip(
            base_cr + np.random.normal(0, 1, num_episodes) * cr_noise, 0, 1
        )

        # 碰撞率（随训练递减）/ Collision rate (decreasing with training)
        base_collision = 0.3 * np.exp(-params["rate"] * 0.8 * episodes) + 0.02
        collision_noise = 0.02 * np.exp(-0.001 * episodes) + 0.005
        collision_rate = np.clip(
            base_collision + np.abs(np.random.normal(0, 1, num_episodes)) * collision_noise,
            0, 1,
        )

        # 步数（随训练递减）/ Steps (decreasing with training)
        base_steps = 300 + 200 * np.exp(-params["rate"] * 0.5 * episodes)
        steps_noise = 30 * np.exp(-0.001 * episodes) + 10
        steps = np.clip(
            base_steps + np.random.normal(0, 1, num_episodes) * steps_noise,
            50, 500,
        )

        data[algo] = {
            "episodes": episodes,
            "reward": reward,
            "completion_rate": completion_rate,
            "collision_rate": collision_rate,
            "steps": steps,
            "convergence_episode": params["conv_ep"],
        }

    return data


def load_csv_data(data_dir: Path) -> Dict[str, Dict[str, np.ndarray]]:
    """从CSV文件加载真实训练数据。

    / Load real training data from CSV files.

    CSV格式预期 / Expected CSV format:
    level,algo,episode,episode_reward,completion_rate,collision_count,steps,...

    Returns:
        {algo: {metric: np.ndarray}} 嵌套字典 / Nested dict
    """
    import csv

    # 找到所有CSV文件 / Find all CSV files
    csv_files = sorted(data_dir.glob("training_log_*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"未找到训练日志CSV文件（training_log_*.csv）于: {data_dir}"
            f" / No training log CSV found in: {data_dir}"
        )

    # 合并所有CSV / Merge all CSVs
    rows = []
    for csv_file in csv_files:
        with open(str(csv_file), "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

    # 按算法分组 / Group by algorithm
    from collections import defaultdict
    algo_data = defaultdict(lambda: {
        "episodes": [], "reward": [], "completion_rate": [],
        "collision_rate": [], "steps": [],
    })

    for row in rows:
        algo = row["algo"]
        algo_data[algo]["episodes"].append(int(row["episode"]))
        algo_data[algo]["reward"].append(float(row["episode_reward"]))
        algo_data[algo]["completion_rate"].append(float(row["completion_rate"]))
        algo_data[algo]["collision_rate"].append(float(row["collision_count"]))
        algo_data[algo]["steps"].append(float(row["steps"]))

    # 转换为numpy数组 / Convert to numpy arrays
    data = {}
    for algo, metrics in algo_data.items():
        data[algo] = {k: np.array(v) for k, v in metrics.items()}
        # 碰撞率归一化 / Normalize collision rate
        if len(data[algo]["collision_rate"]) > 0:
            max_steps = np.max(data[algo]["steps"]) if np.max(data[algo]["steps"]) > 0 else 1
            data[algo]["collision_rate"] = data[algo]["collision_rate"] / max_steps

        # 计算收敛episode / Compute convergence episode
        rewards = data[algo]["reward"]
        if len(rewards) > 100:
            final_avg = np.mean(rewards[-100:])
            threshold = final_avg * 0.9
            converged = np.where(np.convolve(rewards, np.ones(100)/100, mode="valid") >= threshold)[0]
            data[algo]["convergence_episode"] = int(converged[0]) + 100 if len(converged) > 0 else len(rewards)
        else:
            data[algo]["convergence_episode"] = len(rewards)

    return data


def smooth_curve(values: np.ndarray, window: int = 50) -> np.ndarray:
    """滑动窗口平滑曲线。

    / Moving average smoothing.

    Args:
        values: 原始数据 / Raw data
        window: 窗口大小 / Window size

    Returns:
        平滑后的数据 / Smoothed data
    """
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def compute_confidence_interval(values: np.ndarray, window: int = 50,
                                confidence: float = 0.95
                                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算滑动窗口均值和置信区间（纯numpy实现，无scipy依赖）。

    / Compute moving average and confidence interval (pure numpy, no scipy).

    Args:
        values: 原始数据 / Raw data
        window: 窗口大小 / Window size
        confidence: 置信水平 / Confidence level (e.g., 0.95 for 95%)

    Returns:
        (smoothed_mean, ci_lower, ci_upper) 三元组 / Tuple of 3 arrays
    """
    n = len(values)
    if n < window:
        return values, values, values

    # 常用z值近似（无需scipy）/ Common z-score approximations (no scipy needed)
    z_map = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}
    z = z_map.get(round(confidence, 2), 1.960)

    means = np.array([
        np.mean(values[max(0, i - window + 1):i + 1])
        for i in range(n - 1)
    ])
    stds = np.array([
        np.std(values[max(0, i - window + 1):i + 1])
        for i in range(n - 1)
    ])
    counts = np.array([
        min(i + 1, window) for i in range(n - 1)
    ])

    ci_half = z * stds / np.sqrt(np.maximum(counts, 1))
    return means, means - ci_half, means + ci_half


# ============================================================
# 图表1：训练收敛曲线 / Figure 1: Training Convergence Curves
# ============================================================

def plot_convergence(data: Dict[str, Dict[str, np.ndarray]],
                     output_dir: Path,
                     save_pdf: bool = True):
    """生成训练收敛曲线图（带置信区间阴影）。

    / Generate training convergence curves with confidence interval shading.

    Args:
        data: 训练数据 / Training data
        output_dir: 输出目录 / Output directory
        save_pdf: 是否同时保存PDF / Also save PDF
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 左图：奖励收敛 / Left: Reward convergence
    ax = axes[0]
    for algo in data:
        episodes = data[algo]["episodes"]
        rewards = data[algo]["reward"]
        color = PALETTE.get(algo, "#333333")

        # 尝试计算置信区间 / Try computing confidence interval
        try:
            means, ci_lo, ci_hi = compute_confidence_interval(rewards, window=50)
            x = np.arange(1, len(means) + 1)
            ax.plot(x, means, color=color, label=ALGO_DISPLAY.get(algo, algo))
            ax.fill_between(x, ci_lo, ci_hi, color=color, alpha=0.15)
        except Exception:
            # scipy不可用时降级为简单平滑 / Fallback to simple smoothing
            smoothed = smooth_curve(rewards, window=50)
            x = np.arange(1, len(smoothed) + 1)
            ax.plot(x, smoothed, color=color, label=ALGO_DISPLAY.get(algo, algo))

    ax.set_xlabel("Episode")
    ax.set_ylabel("Average Reward")
    ax.set_title("(a) Reward Convergence")
    ax.legend(loc="lower right")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    # 右图：完成率收敛 / Right: Completion rate convergence
    ax = axes[1]
    for algo in data:
        episodes = data[algo]["episodes"]
        cr = data[algo]["completion_rate"]
        color = PALETTE.get(algo, "#333333")

        try:
            means, ci_lo, ci_hi = compute_confidence_interval(cr, window=50)
            x = np.arange(1, len(means) + 1)
            ax.plot(x, means, color=color, label=ALGO_DISPLAY.get(algo, algo))
            ax.fill_between(x, ci_lo, ci_hi, color=color, alpha=0.15)
        except Exception:
            smoothed = smooth_curve(cr, window=50)
            x = np.arange(1, len(smoothed) + 1)
            ax.plot(x, smoothed, color=color, label=ALGO_DISPLAY.get(algo, algo))

    ax.set_xlabel("Episode")
    ax.set_ylabel("Completion Rate")
    ax.set_title("(b) Completion Rate Convergence")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower right")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    plt.tight_layout()
    _save_figure(fig, output_dir, "fig1_convergence_curves", save_pdf)
    plt.close(fig)
    print("  [1/4] 训练收敛曲线已生成 / Convergence curves generated")


# ============================================================
# 图表2：算法对比柱状图 / Figure 2: Algorithm Comparison Bar Chart
# ============================================================

def plot_algorithm_comparison(data: Dict[str, Dict[str, np.ndarray]],
                              output_dir: Path,
                              save_pdf: bool = True):
    """生成6种算法x4指标的对比柱状图。

    / Generate 6-algorithm x 4-metric comparison bar chart.

    指标 / Metrics:
    1. 完成率（最后100回合平均）/ Completion rate (last 100 ep avg)
    2. 平均奖励（最后100回合平均）/ Average reward (last 100 ep avg)
    3. 碰撞率（最后100回合平均）/ Collision rate (last 100 ep avg)
    4. 收敛速度（达到90%最终性能的episode数）/ Convergence speed
    """
    algos = list(data.keys())
    n_algos = len(algos)

    # 计算各指标 / Compute metrics
    completion_rates = []
    avg_rewards = []
    collision_rates = []
    convergence_speeds = []

    for algo in algos:
        d = data[algo]
        last_n = min(100, len(d["reward"]))
        completion_rates.append(np.mean(d["completion_rate"][-last_n:]))
        avg_rewards.append(np.mean(d["reward"][-last_n:]))
        collision_rates.append(np.mean(d["collision_rate"][-last_n:]))
        convergence_speeds.append(d.get("convergence_episode", len(d["reward"])))

    # 归一化收敛速度（越小越好 → 反转为分数）
    # / Normalize convergence speed (smaller is better → invert as score)
    max_conv = max(convergence_speeds) if max(convergence_speeds) > 0 else 1
    convergence_scores = [1 - cs / max_conv for cs in convergence_speeds]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    colors = [PALETTE.get(a, "#333333") for a in algos]
    labels = [ALGO_DISPLAY.get(a, a) for a in algos]
    x = np.arange(n_algos)
    width = 0.6

    # 2a: 完成率 / Completion rate
    ax = axes[0, 0]
    bars = ax.bar(x, completion_rates, width, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Completion Rate")
    ax.set_title("(a) Completion Rate (Last 100 Episodes)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    _add_bar_labels(ax, bars, completion_rates, fmt="{:.2f}")

    # 2b: 平均奖励 / Average reward
    ax = axes[0, 1]
    bars = ax.bar(x, avg_rewards, width, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Average Reward")
    ax.set_title("(b) Average Reward (Last 100 Episodes)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    _add_bar_labels(ax, bars, avg_rewards, fmt="{:.0f}")

    # 2c: 碰撞率 / Collision rate
    ax = axes[1, 0]
    bars = ax.bar(x, collision_rates, width, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Collision Rate")
    ax.set_title("(c) Collision Rate (Last 100 Episodes)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.invert_yaxis()  # 碰撞率越低越好，倒置Y轴 / Lower is better
    _add_bar_labels(ax, bars, collision_rates, fmt="{:.3f}")

    # 2d: 收敛速度 / Convergence speed
    ax = axes[1, 1]
    bars = ax.bar(x, convergence_speeds, width, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Episodes to 90% Performance")
    ax.set_title("(d) Convergence Speed")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    _add_bar_labels(ax, bars, convergence_speeds, fmt="{:.0f}")

    plt.tight_layout()
    _save_figure(fig, output_dir, "fig2_algorithm_comparison", save_pdf)
    plt.close(fig)
    print("  [2/4] 算法对比柱状图已生成 / Algorithm comparison bar chart generated")


# ============================================================
# 图表3：多指标雷达图 / Figure 3: Multi-Metric Radar Chart
# ============================================================

def plot_radar(data: Dict[str, Dict[str, np.ndarray]],
               output_dir: Path,
               save_pdf: bool = True):
    """生成5维雷达图，对比各算法的综合表现。

    / Generate 5-dimension radar chart comparing algorithm performance.

    维度 / Dimensions:
    1. 样本效率（Sample Efficiency）— 同等episode下的奖励
    2. 最终性能（Final Performance）— 最后100回合奖励
    3. 稳定性（Stability）— 最后100回合奖励的方差（取反）
    4. 泛化性（Generalization）— 完成率
    5. 计算效率（Computational Efficiency）— 每episode耗时（取反）
    """
    algos = list(data.keys())
    n_algos = len(algos)
    n_dims = 5
    dim_labels = [
        "Sample\nEfficiency",
        "Final\nPerformance",
        "Stability",
        "Generalization",
        "Computational\nEfficiency",
    ]

    # 计算原始指标 / Compute raw metrics
    raw_metrics = {}
    for algo in algos:
        d = data[algo]
        last_n = min(100, len(d["reward"]))

        # 样本效率：前500回合的平均奖励 / Sample efficiency: avg reward in first 500 ep
        first_n = min(500, len(d["reward"]))
        sample_eff = np.mean(d["reward"][:first_n])

        # 最终性能 / Final performance
        final_perf = np.mean(d["reward"][-last_n:])

        # 稳定性（方差越小越好）/ Stability (smaller variance is better)
        stability = np.std(d["reward"][-last_n:])

        # 泛化性（完成率）/ Generalization (completion rate)
        generalization = np.mean(d["completion_rate"][-last_n:])

        # 计算效率 / Computational efficiency
        convergence_ep = d.get("convergence_episode", len(d["reward"]))

        raw_metrics[algo] = {
            "sample_eff": sample_eff,
            "final_perf": final_perf,
            "stability": stability,
            "generalization": generalization,
            "convergence_ep": convergence_ep,
        }

    # 归一化到[0, 1] / Normalize to [0, 1]
    metrics_matrix = np.zeros((n_algos, n_dims))
    for i, algo in enumerate(algos):
        rm = raw_metrics[algo]
        metrics_matrix[i, 0] = rm["sample_eff"]
        metrics_matrix[i, 1] = rm["final_perf"]
        metrics_matrix[i, 2] = -rm["stability"]  # 反转 / Invert
        metrics_matrix[i, 3] = rm["generalization"]
        metrics_matrix[i, 4] = -rm["convergence_ep"]  # 反转 / Invert

    # Min-max归一化 / Min-max normalization
    for j in range(n_dims):
        col = metrics_matrix[:, j]
        col_min, col_max = col.min(), col.max()
        if col_max - col_min > 1e-8:
            metrics_matrix[:, j] = (col - col_min) / (col_max - col_min)
        else:
            metrics_matrix[:, j] = 0.5

    # 绘制雷达图 / Draw radar chart
    angles = np.linspace(0, 2 * np.pi, n_dims, endpoint=False).tolist()
    angles += angles[:1]  # 闭合 / Close

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for i, algo in enumerate(algos):
        values = metrics_matrix[i].tolist()
        values += values[:1]  # 闭合 / Close
        color = PALETTE.get(algo, "#333333")
        label = ALGO_DISPLAY.get(algo, algo)

        ax.plot(angles, values, color=color, linewidth=2, label=label)
        ax.fill(angles, values, color=color, alpha=0.08)

    # 标签 / Labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dim_labels, size=10)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], size=8, color="grey")
    ax.set_title("Multi-Metric Algorithm Comparison", pad=20, fontsize=13)

    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.3, 1.1),
        fontsize=10,
    )

    plt.tight_layout()
    _save_figure(fig, output_dir, "fig3_radar_chart", save_pdf)
    plt.close(fig)
    print("  [3/4] 多指标雷达图已生成 / Radar chart generated")


# ============================================================
# 图表4：四层级难度递增图 / Figure 4: Level Difficulty Progression
# ============================================================

def plot_level_progression(output_dir: Path, save_pdf: bool = True):
    """生成L1→L4性能递减示意图。

    / Generate L1→L4 performance degradation illustration.

    展示同一算法（PPO/Q-learning）在不同层级下的表现对比。
    由于L1-L4的指标体系不同，这里使用归一化的综合得分。
    """
    levels = ["L1", "L2", "L3", "L4"]
    x = np.arange(len(levels))

    # 模拟各层级最佳算法的综合得分（归一化到[0, 1]）
    # / Simulated normalized scores for best algorithm at each level
    np.random.seed(42)
    base_scores = np.array([0.95, 0.78, 0.60, 0.42])
    noise = np.random.normal(0, 0.02, len(levels))
    scores = np.clip(base_scores + noise, 0, 1)

    # 各维度分解 / Dimension breakdown
    dimensions = {
        "Route\nOptimality":   [0.95, 0.80, 0.55, 0.35],
        "Order\nCompletion":   [0.92, 0.75, 0.62, 0.45],
        "Safety\n(no crash)":  [0.98, 0.85, 0.70, 0.50],
        "Time\nEfficiency":    [0.90, 0.72, 0.58, 0.40],
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 左图：综合得分趋势 / Left: Overall score trend
    ax = axes[0]
    colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]
    bars = ax.bar(x, scores, 0.6, color=colors, edgecolor="white", linewidth=0.5)

    # 趋势线 / Trend line
    z = np.polyfit(x, scores, 2)
    p = np.poly1d(z)
    x_smooth = np.linspace(0, 3, 100)
    ax.plot(x_smooth, p(x_smooth), "--", color="grey", alpha=0.5, linewidth=1.5,
            label="Difficulty trend")

    ax.set_xticks(x)
    ax.set_xticklabels([LEVEL_DISPLAY[l] for l in levels], fontsize=10)
    ax.set_ylabel("Normalized Performance Score")
    ax.set_title("(a) Performance Degradation with Level Complexity")
    ax.set_ylim(0, 1.1)
    ax.legend()
    _add_bar_labels(ax, bars, scores, fmt="{:.2f}")

    # 右图：多维度分组柱状图 / Right: Multi-dimension grouped bar chart
    ax = axes[1]
    n_dims = len(dimensions)
    dim_names = list(dimensions.keys())
    bar_width = 0.18
    offsets = np.arange(n_dims) - (n_dims - 1) / 2

    for i, level in enumerate(levels):
        dim_values = [dimensions[d][i] for d in dim_names]
        pos = np.arange(n_dims) + offsets[i] * bar_width
        ax.barh(
            pos, dim_values, bar_width,
            color=colors[i], edgecolor="white", linewidth=0.5,
            label=level,
        )

    ax.set_yticks(np.arange(n_dims))
    ax.set_yticklabels(dim_names, fontsize=10)
    ax.set_xlabel("Normalized Score")
    ax.set_title("(b) Multi-Dimension Comparison Across Levels")
    ax.set_xlim(0, 1.1)
    ax.legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    _save_figure(fig, output_dir, "fig4_level_progression", save_pdf)
    plt.close(fig)
    print("  [4/4] 四层级难度递增图已生成 / Level progression chart generated")


# ============================================================
# 图表工具函数 / Figure utility functions
# ============================================================

def _save_figure(fig, output_dir: Path, name: str, save_pdf: bool = True):
    """保存图表为PNG和PDF。

    / Save figure as PNG and PDF.

    Args:
        fig: matplotlib Figure对象 / Figure object
        output_dir: 输出目录 / Output directory
        name: 文件名（无后缀）/ Filename (without extension)
        save_pdf: 是否同时保存PDF / Also save PDF
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    png_path = output_dir / f"{name}.png"
    fig.savefig(str(png_path), dpi=300, bbox_inches="tight", pad_inches=0.1)
    print(f"    PNG: {png_path}")

    if save_pdf:
        pdf_path = output_dir / f"{name}.pdf"
        fig.savefig(str(pdf_path), format="pdf", bbox_inches="tight", pad_inches=0.1)
        print(f"    PDF: {pdf_path}")


def _add_bar_labels(ax, bars, values, fmt: str = "{:.2f}"):
    """在柱状图顶部添加数值标签。

    / Add value labels on top of bar chart bars.
    """
    for bar, val in zip(bars, values):
        height = bar.get_height()
        if height >= 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01 * ax.get_ylim()[1],
                fmt.format(val),
                ha="center", va="bottom", fontsize=8, color="0.3",
            )
        else:
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height - 0.02 * abs(ax.get_ylim()[1]),
                fmt.format(val),
                ha="center", va="top", fontsize=8, color="0.3",
            )


# ============================================================
# 主入口 / Main entry point
# ============================================================

def parse_args():
    """解析命令行参数。/ Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="科研绘图脚本 / Scientific Publication Plotting Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例 / Examples:
  # 使用模拟数据生成所有图表
  python scientific_plots.py --demo

  # 从真实CSV数据生成图表
  python scientific_plots.py --data-dir ../02_数据保存/experiment_results/

  # 仅生成指定图表
  python scientific_plots.py --demo --figures convergence,radar
        """,
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="使用模拟数据（无需真实训练结果）/ Use simulated data (no real training needed)",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="训练数据CSV目录 / Training data CSV directory",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="图表输出目录（默认: 04_脚本图片输出/科研图表/）/ "
             "Plot output directory (default: 04_脚本图片输出/科研图表/)",
    )
    parser.add_argument(
        "--figures", type=str, default="all",
        help="要生成的图表（逗号分隔）: convergence,bar,radar,levels / "
             "Figures to generate (comma-separated)",
    )
    parser.add_argument(
        "--episodes", type=int, default=2000,
        help="模拟数据的episode数（仅--demo模式）/ "
             "Episodes for simulated data (--demo only, default: 2000)",
    )
    parser.add_argument(
        "--no-pdf", action="store_true",
        help="不保存PDF格式（仅PNG）/ Don't save PDF format (PNG only)",
    )
    return parser.parse_args()


def main():
    """主入口。/ Main entry point."""
    args = parse_args()

    # 确定输出目录 / Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = PROJECT_ROOT / "04_脚本图片输出" / "科研图表"

    output_dir.mkdir(parents=True, exist_ok=True)
    save_pdf = not args.no_pdf

    # 配置绘图风格 / Configure plot style
    setup_plot_style()

    # 加载数据 / Load data
    if args.demo:
        print("使用模拟数据 / Using simulated data")
        data = generate_demo_data(num_episodes=args.episodes)
    elif args.data_dir:
        print(f"从CSV加载数据 / Loading CSV data from: {args.data_dir}")
        data = load_csv_data(Path(args.data_dir))
    else:
        # 默认尝试加载，失败则使用模拟数据
        # / Try loading, fallback to demo
        default_csv = PROJECT_ROOT / "02_数据保存" / "experiment_results"
        if default_csv.exists() and list(default_csv.glob("training_log_*.csv")):
            print(f"自动检测CSV数据 / Auto-detected CSV data: {default_csv}")
            data = load_csv_data(default_csv)
        else:
            print("未找到训练数据，使用模拟数据 / No data found, using simulated data")
            data = generate_demo_data(num_episodes=args.episodes)

    # 确定要生成的图表 / Determine which figures to generate
    if args.figures == "all":
        figures = ["convergence", "bar", "radar", "levels"]
    else:
        figures = [f.strip() for f in args.figures.split(",")]

    # 验证图表名 / Validate figure names
    valid_figures = {"convergence", "bar", "radar", "levels"}
    for f in figures:
        if f not in valid_figures:
            print(f"警告：未知图表名 '{f}'，跳过。"
                  f"有效选项: {valid_figures} / "
                  f"Warning: unknown figure '{f}', skipped.")
    figures = [f for f in figures if f in valid_figures]

    print(f"\n输出目录 / Output directory: {output_dir}")
    print(f"图表列表 / Figures: {figures}")
    print(f"格式 / Format: PNG" + (" + PDF" if save_pdf else ""))
    print()

    # 生成图表 / Generate figures
    if "convergence" in figures:
        plot_convergence(data, output_dir, save_pdf)
    if "bar" in figures:
        plot_algorithm_comparison(data, output_dir, save_pdf)
    if "radar" in figures:
        plot_radar(data, output_dir, save_pdf)
    if "levels" in figures:
        plot_level_progression(output_dir, save_pdf)

    print(f"\n所有图表已保存到 / All figures saved to: {output_dir}")
    print("完成! / Done!")


if __name__ == "__main__":
    main()
