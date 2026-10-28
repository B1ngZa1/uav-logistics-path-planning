"""
职责 / Responsibility: 科研绘图指南 -- 学习曲线、算法对比、环境示意图、训练可视化、导出规范
参考 / Reference: sci_style_config.py, plot_convergence_detail.py, plot_algorithm_bars.py,
                   plot_radar_comparison.py, plot_training_curves.py
领域适配 / Domain Adaptation: Nature/Science出版标准 → 无人机配送RL实验图表
"""

## 9. 科研绘图指南 / Scientific Visualization Guide

---

### 9.1 学习曲线绘制 / Learning Curve Plotting

#### 9.1.1 EMA平滑 + 置信带方法 / EMA Smoothing + Confidence Band

训练曲线通常噪声很大，需要平滑处理。推荐两种方法：
Training curves are typically noisy and need smoothing. Two methods are recommended:

| 方法 / Method | 公式 / Formula | 优点 / Pros | 缺点 / Cons |
|---|---|---|---|
| 移动平均 (MA) | `smooth[i] = mean(raw[i-w:i+w])` | 简单、直观 | 滞后性 |
| 指数移动平均 (EMA) | `ema[i] = α*raw[i] + (1-α)*ema[i-1]` | 响应快 | 需选α |

##### 核心函数：置信带计算 / Core Function: Confidence Band

```python
import numpy as np

def confidence_band(values, window=50, z=1.96):
    """
    计算滑动窗口均值和95%置信带。
    Compute sliding window mean and 95% confidence band.

    Parameters
    ----------
    values : np.ndarray
        原始训练指标序列 / Raw training metric sequence
    window : int
        滑动窗口大小 / Sliding window size
    z : float
        z分数（1.96对应95%CI）/ z-score (1.96 for 95% CI)

    Returns
    -------
    means, ci_lower, ci_upper : np.ndarray
        均值、下界、上界 / Mean, lower, upper
    """
    n = len(values)
    if n < window:
        return values, values, values
    means, stds = [], []
    for i in range(n - window + 1):
        seg = values[i:i + window]
        means.append(np.mean(seg))
        stds.append(np.std(seg))
    means = np.array(means)
    stds = np.array(stds)
    ci_half = z * stds / np.sqrt(window)
    return means, means - ci_half, means + ci_half
```

#### 9.1.2 matplotlib代码模板 / matplotlib Code Template

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# 六算法配色方案（色盲友好）/ Six-algorithm color palette (colorblind-friendly)
PALETTE = {
    "q_learning":  "#1f77b4",  # 蓝 / Blue
    "sarsa":       "#ff7f0e",  # 橙 / Orange
    "monte_carlo": "#2ca02c",  # 绿 / Green
    "dqn":         "#d62728",  # 红 / Red
    "a2c":         "#9467bd",  # 紫 / Purple
    "ppo":         "#8c564b",  # 棕 / Brown
}

ALGO_DISPLAY = {
    "q_learning":  "Q-Learning",
    "sarsa":       "SARSA",
    "monte_carlo": "Monte Carlo",
    "dqn":         "DQN",
    "a2c":         "A2C",
    "ppo":         "PPO",
}

def setup_sci_style():
    """配置SCI论文绘图风格 / Configure SCI paper plot style"""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["SimHei", "Microsoft YaHei", "DejaVu Sans", "Arial"],
        "font.size": 11,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 1.8,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.unicode_minus": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    })
```

```python
def plot_reward_convergence(data, output_dir):
    """
    绘制6种算法奖励收敛曲线（含置信区间和收敛标注）。
    Plot reward convergence for 6 algorithms with CI shading and convergence markers.

    Parameters
    ----------
    data : dict
        {algo_name: {"reward": np.ndarray, "episodes": np.ndarray}}
    output_dir : Path
        输出目录 / Output directory
    """
    setup_sci_style()
    fig, ax = plt.subplots(figsize=(12, 7))

    for algo_name, algo_data in data.items():
        rewards = algo_data["reward"]
        color = PALETTE[algo_name]
        label = ALGO_DISPLAY[algo_name]

        means, ci_lo, ci_hi = confidence_band(rewards, window=50)
        x = np.arange(1, len(means) + 1)
        ax.plot(x, means, color=color, label=label, linewidth=1.8)
        ax.fill_between(x, ci_lo, ci_hi, color=color, alpha=0.12)

    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Average Reward")
    ax.set_title("Reward Convergence Curves")
    ax.legend(loc="lower right", ncol=2, framealpha=0.9)
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    )

    plt.tight_layout()
    plt.savefig(output_dir / "fig_reward_convergence.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "fig_reward_convergence.pdf", bbox_inches="tight")
    plt.close(fig)
```

#### 9.1.3 SCI/Nature标准规范 / SCI/Nature Standards

| 规范项 / Specification | 标准 / Standard | 说明 / Notes |
|---|---|---|
| 分辨率 | 300 dpi | 顶级期刊标准 |
| 格式 | PNG（预览）+ PDF/SVG（投稿） | 矢量格式用于印刷 |
| 字体 | 中文宋体/SimHei + 英文 Arial/TNR | 统一中英文字体 |
| 线宽 | 1.5-2.5pt | 清晰可辨 |
| 图表尺寸 | 单栏8cm，双栏17cm | 适配期刊模板 |
| 坐标轴标签 | 含单位，如 "Episode" 或 "Average Reward" | 不缩写 |
| 图例 | 包含所有数据系列，带alpha=0.9背景 | 不遮挡数据 |
| 配色 | 色盲友好（使用tab10或自定义6色） | 避免纯红绿对比 |

---

### 9.2 算法对比图表 / Algorithm Comparison Charts

#### 9.2.1 柱状图 / Bar Chart

```python
def plot_four_metric_bars(data, output_dir):
    """
    生成4子图柱状图（完成率/奖励/碰撞率/收敛速度）。
    Generate 4-panel bar chart (completion/reward/collision/convergence).

    Parameters
    ----------
    data : dict
        {algo: {"metric": (mean, std)}}
    output_dir : Path
    """
    setup_sci_style()
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    algos = list(data.keys())
    colors = [PALETTE[a] for a in algos]
    labels = [ALGO_DISPLAY[a] for a in algos]
    x = np.arange(len(algos))

    metrics = [
        ("completion_rate", "Completion Rate", axes[0, 0], "(a)", True, "{:.2f}"),
        ("avg_reward",      "Average Reward",  axes[0, 1], "(b)", True, "{:.0f}"),
        ("collision_rate",  "Collision Rate",  axes[1, 0], "(c)", False, "{:.3f}"),
        ("convergence_ep",  "Episodes",        axes[1, 1], "(d)", False, "{:.0f}"),
    ]

    for metric_key, ylabel, ax, tag, higher_better, fmt in metrics:
        means = [data[a][metric_key][0] for a in algos]
        stds  = [data[a][metric_key][1] for a in algos]
        bars = ax.bar(x, means, 0.6, color=colors, edgecolor="white",
                      linewidth=0.5, yerr=stds, capsize=4,
                      error_kw={"linewidth": 1.2})

        # 标记最优 / Mark best
        best_idx = np.argmax(means) if higher_better else np.argmin(means)
        bars[best_idx].set_edgecolor("gold")
        bars[best_idx].set_linewidth(2.5)

        ax.set_ylabel(ylabel)
        ax.set_title(f"{tag} {ylabel}")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Six-Algorithm Four-Metric Comparison",
                 fontsize=16, fontweight="bold", y=1.05)
    plt.tight_layout()
    plt.savefig(output_dir / "fig_four_metric_bars.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "fig_four_metric_bars.pdf", bbox_inches="tight")
    plt.close(fig)
```

#### 9.2.2 箱线图 / Box Plot

```python
def plot_algorithm_boxplot(data, metric, output_path):
    """
    箱线图展示多种子评估分布 / Boxplot showing multi-seed evaluation distribution

    Parameters
    ----------
    data : dict
        {algo_name: np.ndarray of per-seed scores}
    metric : str
        指标名称 / Metric name
    output_path : Path
    """
    setup_sci_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    algo_names = list(data.keys())
    scores = [data[a] for a in algo_names]
    bp = ax.boxplot(scores, labels=[ALGO_DISPLAY[a] for a in algo_names],
                    patch_artist=True, widths=0.5)

    for patch, algo in zip(bp["boxes"], algo_names):
        patch.set_facecolor(PALETTE[algo])
        patch.set_alpha(0.7)

    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(f"Algorithm {metric} Distribution (Multi-Seed)")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
```

#### 9.2.3 雷达图 / Radar Chart

```python
def plot_radar_comparison(radar_data, output_path):
    """
    5维雷达图对比6种算法 / 5-dim radar chart comparing 6 algorithms

    Parameters
    ----------
    radar_data : dict
        {algo: [dim1, dim2, dim3, dim4, dim5]}
        五个维度: 样本效率、最终性能、稳定性、泛化性、计算效率
    output_path : Path
    """
    setup_sci_style()
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

    dim_labels = [
        "Sample\nEfficiency",
        "Final\nPerformance",
        "Stability",
        "Generalization",
        "Computational\nEfficiency",
    ]
    dims = len(dim_labels)
    angles = np.linspace(0, 2 * np.pi, dims, endpoint=False).tolist()
    angles += angles[:1]  # 闭合 / Close

    for algo, values in radar_data.items():
        vals = values + values[:1]
        color = PALETTE.get(algo, "#333333")
        ax.plot(angles, vals, color=color, linewidth=2.2, label=ALGO_DISPLAY.get(algo, algo))
        ax.fill(angles, vals, color=color, alpha=0.06)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dim_labels, size=9.5, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], size=8, color="grey")
    ax.set_title("Six-Algorithm Five-Dimension Evaluation", pad=25,
                 fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.spines["polar"].set_visible(False)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=10, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(str(output_path).replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
```

#### 9.2.4 配色方案（色盲友好）/ Color Palette (Colorblind-Friendly)

本项目统一使用以下6色配色方案（源自matplotlib tab10，色盲安全）：
This project uses the following 6-color palette (from matplotlib tab10, colorblind-safe):

| 算法 / Algorithm | 颜色 / Color | HEX代码 | RGB |
|---|---|---|---|
| Q-learning | 蓝色 / Blue | `#1f77b4` | (31, 119, 180) |
| SARSA | 橙色 / Orange | `#ff7f0e` | (255, 127, 14) |
| Monte Carlo | 绿色 / Green | `#2ca02c` | (44, 160, 44) |
| DQN | 红色 / Red | `#d62728` | (214, 39, 40) |
| A2C | 紫色 / Purple | `#9467bd` | (148, 103, 189) |
| PPO | 棕色 / Brown | `#8c564b` | (140, 86, 75) |

---

### 9.3 环境示意图使用 / Environment Visualization

#### 9.3.1 outputs/图片索引表 / outputs/ Image Index

| 图片 / Image | 路径 / Path | 用途 / Usage |
|---|---|---|
| 四区总览 | `outputs/环境示意图/wuhan_environment_overview.png` | 项目全景介绍 |
| 3D环境 | `outputs/环境示意图/wuhan_environment_3d.png` | 三维场景展示 |
| 建筑高度热力图 | `outputs/环境示意图/wuhan_building_height_heatmap.png` | 高度分布分析 |
| 障碍物图例 | `outputs/环境示意图/wuhan_obstacle_legend.png` | 风险类型图例 |
| 四区分区图 | `outputs/环境示意图/wuhan_zone_top_view.png` | 分区布局 |
| Zone A 2D | `outputs/环境示意图_2D审核/zone_A_guanggu_core_2d.png` | 光谷广场2D |
| Zone B 2D | `outputs/环境示意图_2D审核/zone_B_lakeside_campus_2d.png` | 武汉学院2D |
| Zone C 2D | `outputs/环境示意图_2D审核/zone_C_river_bridgehead_2d.png` | 长江大桥2D |
| Zone D 2D | `outputs/环境示意图_2D审核/zone_D_canal_highrise_2d.png` | 楚河汉界2D |
| Zone A 3D | `outputs/环境示意图_3D预览/zone_A_scene_lod0_preview.png` | 光谷3D预览 |
| Zone B 3D | `outputs/环境示意图_3D预览/zone_B_scene_lod0_preview.png` | 学院3D预览 |
| Zone C 3D | `outputs/环境示意图_3D预览/zone_C_scene_lod0_preview.png` | 桥头3D预览 |
| Zone D 3D | `outputs/环境示意图_3D预览/zone_D_scene_lod0_preview.png` | 河道3D预览 |

#### 9.3.2 已有绘图脚本索引 / Existing Plotting Scripts Index

位于 `03_图表脚本/Python脚本/` 下的绘图脚本：
Located in `03_图表脚本/Python脚本/`:

| 脚本 / Script | 功能 / Function | 输出 / Output |
|---|---|---|
| `sci_style_config.py` | SCI风格配置（颜色、字体、图例） | 全局样式 |
| `plot_convergence_detail.py` | 6种算法训练收敛曲线（含置信区间） | fig1a-1d |
| `plot_algorithm_bars.py` | 4指标分组柱状图 + 归一化对比 | fig2a-2c |
| `plot_radar_comparison.py` | 5维雷达图（标准/分组/PPO高亮） | fig3a-3c |
| `plot_training_curves.py` | 综合训练指标（奖励/完成率/碰撞/熵/损失） | 7张图 |
| `plot_algorithm_compare.py` | 算法对比柱状图 + 雷达图 | 4张图 |
| `plot_scene_heatmap.py` | 场景热力图 | 分区热力图 |
| `plot_environment.py` | 环境示意图 | 2D/3D环境 |
| `plot_wuhan_environment.py` | 武汉环境绘制 | 武汉场景 |
| `plot_wuhan_reference_scenes_2d.py` | 四区2D参考场景 | 2D审核图 |
| `convert_wuhan_reference_scenes_2d_to_3d.py` | 2D→3D场景转换 | OBJ/MTL |
| `plot_path_3d.py` | 3D路径可视化 | 飞行轨迹 |
| `plot_reward_curve.py` | 奖励曲线 | 奖励图 |
| `plot_workflow.py` | 工作流图 | 流程图 |

#### 9.3.3 图片引用方式 / Image Reference Method

在开发指南中引用图片时使用相对路径：
When referencing images in the guide, use relative paths:

```markdown
![四区环境总览](outputs/环境示意图/wuhan_environment_overview.png)
![Zone B 2D](outputs/环境示意图_2D审核/zone_B_lakeside_campus_2d.png)
```

---

### 9.4 训练过程可视化 / Training Process Visualization

#### 9.4.1 Q表增长曲线 / Q-table Growth Curve

```python
def plot_qtable_growth(q_table_sizes, output_path):
    """
    绘制Q表状态数随训练增长的曲线。
    Plot Q-table state count growth during training.

    Parameters
    ----------
    q_table_sizes : list
        每回合记录的Q表大小 / Q-table size per episode
    output_path : Path
    """
    setup_sci_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    episodes = np.arange(1, len(q_table_sizes) + 1)
    means, ci_lo, ci_hi = confidence_band(np.array(q_table_sizes), window=50)
    x = np.arange(1, len(means) + 1)

    ax.plot(x, means, color="#1f77b4", linewidth=2, label="Q-table States")
    ax.fill_between(x, ci_lo, ci_hi, color="#1f77b4", alpha=0.15)

    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Number of Visited States")
    ax.set_title("Q-table Growth During Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
```

#### 9.4.2 奖励分量堆叠图 / Reward Component Stacked Chart

```python
def plot_reward_components_stacked(components_history, output_path):
    """
    绘制奖励分量堆叠面积图。
    Plot reward components as stacked area chart.

    Parameters
    ----------
    components_history : dict
        {component_name: [values_per_episode]}
    output_path : Path
    """
    setup_sci_style()
    fig, ax = plt.subplots(figsize=(12, 6))

    names = list(components_history.keys())
    values = np.array([components_history[n] for n in names])

    # 平滑 / Smooth
    window = 20
    smoothed = np.array([
        np.convolve(v, np.ones(window)/window, mode="valid") for v in values
    ])

    # 正负分量分离绘制 / Separate positive and negative components
    pos_mask = np.all(smoothed >= 0, axis=1)
    neg_mask = np.all(smoothed < 0, axis=1)

    pos_names = [n for n, m in zip(names, pos_mask) if m]
    neg_names = [n for n, m in zip(names, neg_mask) if m]

    if any(pos_mask):
        ax.stackplot(np.arange(smoothed.shape[1]),
                     smoothed[pos_mask],
                     labels=pos_names, alpha=0.7)
    if any(neg_mask):
        ax.stackplot(np.arange(smoothed.shape[1]),
                     smoothed[neg_mask],
                     labels=neg_names, alpha=0.7)

    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Reward Component Value")
    ax.set_title("Reward Component Decomposition")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
```

#### 9.4.3 多指标综合面板 / Multi-Metric Combined Panel

```python
def plot_combined_panel(data, output_path):
    """
    生成2x2综合面板：奖励+完成率+碰撞率+收敛速度。
    Generate 2x2 panel: reward + completion + collision + convergence speed.
    """
    setup_sci_style()
    fig, axes = plt.subplots(2, 2, figsize=(18, 13))

    # 三个曲线子图 + 一个柱状图 / Three curve subplots + one bar chart
    # ... (参考 plot_convergence_detail.py 的 plot_combined_panel 函数)

    plt.suptitle("RL Training Convergence Analysis",
                 fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(str(output_path).replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
```

---

### 9.5 图表导出规范 / Chart Export Standards

#### 9.5.1 期刊PDF/SVG格式 / Journal PDF/SVG Format

```python
def save_for_journal(fig, output_path, dpi=300):
    """
    保存为期刊投稿格式 / Save in journal submission format

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    output_path : str or Path
        基础路径（自动追加.pdf和.svg）/ Base path
    dpi : int
        PNG分辨率 / PNG resolution
    """
    # PNG（用于预览和Word文档）/ PNG for preview and Word docs
    fig.savefig(str(output_path) + ".png", dpi=dpi, bbox_inches="tight",
                pad_inches=0.1)
    # PDF（矢量，用于LaTeX投稿）/ PDF vector for LaTeX submission
    fig.savefig(str(output_path) + ".pdf", bbox_inches="tight", pad_inches=0.1)
    # SVG（矢量，用于编辑修改）/ SVG vector for editing
    fig.savefig(str(output_path) + ".svg", bbox_inches="tight", pad_inches=0.1)
```

#### 9.5.2 答辩PPT格式 / Defense PPT Format

答辩用图表要求更高对比度和更大字体：
Defense slides require higher contrast and larger fonts:

```python
def save_for_ppt(fig, output_path, dpi=150):
    """
    保存为答辩PPT格式 / Save in defense PPT format
    - 降低dpi以减小文件大小
    - 使用更大字体
    - 更高对比度
    """
    # 调整字体大小 / Adjust font sizes
    for ax in fig.get_axes():
        ax.title.set_fontsize(18)
        ax.xaxis.label.set_fontsize(14)
        ax.yaxis.label.set_fontsize(14)
        for item in ax.get_xticklabels() + ax.get_yticklabels():
            item.set_fontsize(12)

    fig.savefig(str(output_path) + "_ppt.png", dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
```

#### 9.5.3 300dpi标准 / 300dpi Standard

所有正式输出图表统一遵守以下导出标准：
All formal output charts follow these export standards:

| 输出场景 / Output | DPI | 格式 / Format | 尺寸 / Size | 背景 / Background |
|---|---|---|---|---|
| 期刊投稿 / Journal | 300 | PDF + PNG | 单栏8cm 或 双栏17cm宽 | 白色 |
| 答辩幻灯片 / Slides | 150 | PNG | 宽度16cm | 白色 |
| 海报 / Poster | 300 | PDF | A0/A1 | 白色 |
| 网页/屏幕 / Screen | 96-150 | PNG | 宽度1200px | 白色 |
| 论文草稿 / Draft | 150 | PNG | 宽度12cm | 白色 |

```python
# 全局设置（放在脚本开头）/ Global settings (at script top)
plt.rcParams.update({
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "savefig.facecolor": "white",
    "savefig.edgecolor": "none",
})
```

---

> **下一篇 / Next**: [10. 编码规范 / Coding Standards](./10_coding_standards.md)
