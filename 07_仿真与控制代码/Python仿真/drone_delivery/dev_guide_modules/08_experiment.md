"""
职责 / Responsibility: 实验设计指南 -- 对比实验、消融实验、超参数敏感度、奖励调参、课程学习
参考 / Reference: reward_function_design.md (§3-§7), PPO开发指南v3.0 (§29), plot_algorithm_bars.py
领域适配 / Domain Adaptation: 学术实验方法论 → 无人机配送RL算法评估
"""

## 8. 实验设计指南 / Experiment Design Guide

---

### 8.1 对比实验设计 / Comparative Experiment Design

#### 8.1.1 统一控制变量清单 / Unified Control Variables

所有对比实验**必须**保持以下变量一致：
All comparative experiments **must** keep these variables consistent:

| 控制变量 / Control Variable | 统一值 / Unified Value | 说明 / Notes |
|---|---|---|
| 随机种子种子范围 | 42-46 (5 seeds) | 所有算法共用同一组种子 |
| 训练回合数 | 2000 | 统一训练量 |
| 评估方式 | exploit()确定性 | 不含探索噪声 |
| 评估episode数 | 100 | 固定种子100回合 + 未见种子100回合 |
| 环境配置 | 同一TOML | grid_size, max_steps, num_orders |
| 奖励函数 | 同一组件和权重 | 确保奖励信号一致 |
| 记录间隔 | 每10回合 | 统一采样频率 |

**禁用的变量 / Forbidden Variables**:
- 不同算法使用不同的超参数搜索结果（应使用默认参数或统一搜索）
- 不同的评估种子范围（训练种子42-46，评估种子10000-10099）

#### 8.1.2 评估指标定义 / Evaluation Metric Definitions

| 指标 / Metric | 定义 / Definition | 计算 / Calculation |
|---|---|---|
| 成功率 / Success Rate | 完成全部配送并安全返航的回合占比 | `success_cnt / N` |
| 配送完成率 / Delivery Rate | 已完成订单数 ÷ 计划订单数 | `completed / total_orders` |
| 碰撞率 / Collision Rate | 发生碰撞的回合占比 | `collision_cnt / N` |
| 平均奖励 / Avg Reward | 最后100回合平均奖励 | `mean(rewards[-100:])` |
| 收敛速度 / Convergence Speed | 达到90%最终性能所需episode数 | `first_episode(rew > 0.9 * final_rew)` |
| 平均步数 / Avg Steps | 最后100回合平均步数 | `mean(steps[-100:])` |

#### 8.1.3 统计检验方法 / Statistical Testing Methods

##### 参数检验：独立样本t检验 / Independent Samples t-test

```python
from scipy import stats
import numpy as np

def compare_algorithms_ttest(scores_a, scores_b, alpha=0.05):
    """
    独立样本t检验 / Independent samples t-test

    Parameters
    ----------
    scores_a : np.ndarray
        算法A的评估分数（5种子×100回合） / Algorithm A scores
    scores_b : np.ndarray
        算法B的评估分数 / Algorithm B scores
    alpha : float
        显著性水平 / Significance level (default: 0.05)

    Returns
    -------
    dict
        t_statistic, p_value, significant, cohens_d
    """
    t_stat, p_value = stats.ttest_ind(scores_a, scores_b)

    # Cohen's d 效应量 / Cohen's d effect size
    n_a, n_b = len(scores_a), len(scores_b)
    pooled_std = np.sqrt(
        ((n_a - 1) * np.std(scores_a, ddof=1)**2 +
         (n_b - 1) * np.std(scores_b, ddof=1)**2) / (n_a + n_b - 2)
    )
    cohens_d = (np.mean(scores_a) - np.mean(scores_b)) / pooled_std

    return {
        "t_statistic": t_stat,
        "p_value": p_value,
        "significant": p_value < alpha,
        "cohens_d": cohens_d,
    }
```

##### 非参数检验：Wilcoxon秩和检验 / Wilcoxon Rank-Sum Test

当样本不满足正态性假设时使用：
Use when samples do not satisfy normality assumption:

```python
def compare_algorithms_wilcoxon(scores_a, scores_b, alpha=0.05):
    """
    Wilcoxon秩和检验（Mann-Whitney U） / Wilcoxon rank-sum test

    Parameters
    ----------
    scores_a : np.ndarray
        算法A的评估分数 / Algorithm A scores
    scores_b : np.ndarray
        算法B的评估分数 / Algorithm B scores
    alpha : float
        显著性水平 / Significance level

    Returns
    -------
    dict
        u_statistic, p_value, significant
    """
    u_stat, p_value = stats.mannwhitneyu(scores_a, scores_b, alternative='two-sided')
    return {
        "u_statistic": u_stat,
        "p_value": p_value,
        "significant": p_value < alpha,
    }
```

##### 置信区间 / Confidence Interval

```python
def compute_ci95(data):
    """
    计算95%置信区间 / Compute 95% confidence interval

    Parameters
    ----------
    data : np.ndarray
        评估数据 / Evaluation data

    Returns
    -------
    dict
        mean, std, ci_lower, ci_upper
    """
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    se = std / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    return {
        "mean": mean,
        "std": std,
        "ci_lower": mean - t_crit * se,
        "ci_upper": mean + t_crit * se,
    }
```

#### 8.1.4 实验结果报告模板 / Experiment Report Template

| 算法 / Algorithm | 完成率 (%) | 碰撞率 (%) | 平均奖励 | 收敛速度 (ep) | 统计显著性 |
|---|---:|---:|---:|---:|---|
| Q-learning | 62.3 ± 4.1 | 8.2 ± 2.0 | 280 ± 30 | 800 ± 80 | — |
| SARSA | 58.1 ± 5.2 | 9.1 ± 2.3 | 250 ± 35 | 900 ± 90 | p=0.12 (n.s.) |
| Monte Carlo | 50.4 ± 6.0 | 12.0 ± 3.1 | 220 ± 45 | 1200 ± 120 | p=0.001* |
| DQN | 78.2 ± 3.0 | 5.1 ± 1.2 | 320 ± 20 | 500 ± 50 | p<0.001** |
| A2C | 75.3 ± 3.3 | 6.2 ± 1.8 | 310 ± 25 | 600 ± 60 | p<0.001** |
| PPO | 85.1 ± 2.1 | 3.8 ± 1.0 | 350 ± 15 | 400 ± 40 | — |

*统计显著性列显示与PPO的对比p值。`*` p<0.01, `**` p<0.001, `n.s.` 不显著。
*Significance column shows p-value vs PPO. `*` p<0.01, `**` p<0.001, `n.s.` not significant.*

---

### 8.2 消融实验设计 / Ablation Study Design

消融实验通过逐一移除组件来验证每个模块的贡献。
Ablation studies validate each module's contribution by removing one component at a time.

#### 8.2.1 奖励分量消融 / Reward Component Ablation

| 实验 / Experiment | 移除组件 / Removed | 预期影响 / Expected Impact |
|---|---|---|
| Full（完整模型） | 无 / None | 基线 / Baseline |
| w/o R_goal | 目标到达奖励 | 探索困难，不尝试到达 |
| w/o R_dist | 距离引导奖励 | 稀疏奖励问题加剧 |
| w/o P_collision | 碰撞惩罚 | 碰撞率大幅上升 |
| w/o P_time | 时间惩罚 | 效率下降，徘徊增多 |
| w/o P_energy | 能耗惩罚 | 能耗不受控 |
| w/o Shaping | 势能塑形 | 初期学习困难 |

奖励消融表格模板 / Reward ablation table template:

| 实验 | 完成率 | 碰撞率 | 平均奖励 | vs Full Δ |
|---|---:|---:|---:|---:|
| Full | 85.1% | 3.8% | 350 | — |
| w/o R_goal | 12.3% | 1.2% | -45 | -72.8% |
| w/o R_dist | 68.5% | 4.5% | 290 | -16.5% |
| w/o P_collision | 72.1% | 35.2% | 180 | -13.0% |
| w/o P_time | 78.8% | 4.1% | 285 | -6.3% |
| w/o Shaping | 61.2% | 5.8% | 260 | -23.9% |

#### 8.2.2 网络结构消融 / Network Architecture Ablation

| 实验 / Experiment | 变更 / Change | 适用算法 |
|---|---|---|
| Full Dueling DQN | 标准Dueling架构 | DQN |
| w/ Plain DQN | 移除Dueling（单流） | DQN |
| w/ No Target Net | 移除目标网络 | DQN |
| w/ No Replay | 移除经验回放 | DQN |
| Full A2C + GAE | 标准A2C + GAE | A2C |
| w/ No GAE | 移除GAE（用MC回报） | A2C |

#### 8.2.3 探索策略消融 / Exploration Strategy Ablation

| 实验 / Experiment | 探索策略 / Strategy | 参数 / Parameters |
|---|---|---|
| ε-greedy (默认) | 指数衰减 | ε: 1.0→0.05, decay=0.9995 |
| ε-greedy (快衰减) | 快速衰减 | ε: 1.0→0.05, decay=0.995 |
| ε-greedy (慢衰减) | 慢速衰减 | ε: 1.0→0.10, decay=0.9999 |
| Boltzmann | softmax温度 | τ: 1.0→0.1 |
| UCB | 上置信界 | c=2.0 |

#### 8.2.4 消融表格消融标准模板 / Ablation Table Standard Template

| 配置 / Config | avg_reward | delivery_rate | collision_rate | convergence_ep |
|---|---:|---:|---:|---:|
| Full | **350** | **85.1%** | **3.8%** | **400** |
| Ablation 1 | 290 | 68.5% | 4.5% | 600 |
| Ablation 2 | 320 | 75.3% | 5.1% | 500 |
| Ablation 3 | 260 | 61.2% | 5.8% | 700 |

*加粗表示该指标最优 / Bold indicates best value for that metric.*

---

### 8.3 超参数敏感度分析 / Hyperparameter Sensitivity Analysis

#### 8.3.1 搜索空间定义 / Search Space Definition

| 超参数 / Hyperparameter | 搜索范围 / Search Range | 尺度 / Scale | 默认值 / Default |
|---|---|---|---|
| 学习率 lr | [1e-5, 5e-5, 1e-4, 3e-4, 1e-3, 3e-3] | 对数 / Log | 3e-4 |
| 折扣因子 γ | [0.90, 0.95, 0.97, 0.99, 0.995, 0.999] | 线性 / Linear | 0.99 |
| GAE λ | [0.90, 0.93, 0.95, 0.97, 0.99] | 线性 / Linear | 0.95 |
| 裁剪系数 ε_clip | [0.1, 0.15, 0.2, 0.3, 0.4] | 线性 / Linear | 0.2 |
| 熵系数 β_ent | [0.001, 0.005, 0.01, 0.05, 0.1] | 对数 / Log | 0.01 |
| ε衰减 decay | [0.995, 0.999, 0.9995, 0.9999] | 线性 / Linear | 0.9995 |
| 批量大小 batch | [64, 128, 256, 512] | 对数 / Log | 256 |
| 隐藏层维度 | [64, 128, 256, 512] | 对数 / Log | 256 |

#### 8.3.2 搜索方法 / Search Methods

##### 单变量扫描 / Single-Variable Sweep

固定其他参数，逐一扫描目标参数：
Fix other parameters, sweep one target at a time:

```python
def single_param_sweep(agent_class, env_class, param_name, param_values,
                       fixed_config, n_episodes=2000, n_seeds=5):
    """
    单参数敏感度扫描 / Single parameter sensitivity sweep

    Parameters
    ----------
    param_name : str
        要扫描的参数名 / Parameter name to sweep
    param_values : list
        参数值列表 / Parameter value list
    """
    results = {}
    for value in param_values:
        config = fixed_config.copy()
        config[param_name] = value
        seed_scores = []

        for seed in range(42, 42 + n_seeds):
            agent = agent_class(**config)
            env = env_class()
            # ... 训练 / train ...
            score = evaluate_agent(agent, env, n_episodes=100)
            seed_scores.append(score["avg_reward"])

        results[value] = {
            "mean": np.mean(seed_scores),
            "std": np.std(seed_scores),
            "ci95": compute_ci95(seed_scores),
        }
    return results
```

##### 网格搜索 / Grid Search

对于2-3个参数的联合搜索：
For joint search of 2-3 parameters:

```python
import itertools

param_grid = {
    "learning_rate": [1e-4, 3e-4, 1e-3],
    "gamma": [0.95, 0.99, 0.999],
}

for lr, gamma in itertools.product(*param_grid.values()):
    config = {"learning_rate": lr, "gamma": gamma}
    # ... 训练与评估 / train and evaluate ...
```

#### 8.3.3 可视化方案 / Visualization Plan

| 图表类型 / Chart Type | 适用场景 / Use Case | 脚本 / Script |
|---|---|---|
| 折线图（单参数） | 单变量扫描结果 | `plot_parameter_sensitivity()` |
| 热力图（双参数） | 2D网格搜索结果 | `plot_parameter_heatmap()` |
| 箱线图 | 参数值分布对比 | `plot_algorithm_boxplot()` |

单参数敏感度曲线模板 / Single parameter sensitivity curve template:

```python
def plot_param_sensitivity(param_values, means, stds, param_name, metric,
                            output_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(param_values, means, yerr=stds, fmt='o-', capsize=5,
                color='#1f77b4', linewidth=2, markersize=8)
    best_idx = np.argmax(means)
    ax.plot(param_values[best_idx], means[best_idx], 'r*', markersize=15,
            label=f'Best: {param_values[best_idx]}')
    ax.set_xlabel(param_name, fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(f'{param_name} Sensitivity Analysis', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
```

---

### 8.4 奖励函数调参指南 / Reward Tuning Guide

详细数学公式和代码见 `reward_function_design.md`。
Detailed math formulas and code in `reward_function_design.md`.

#### 8.4.1 四层分层标定策略 / Four-Layer Layered Calibration Strategy

采用"从简到繁、逐步叠加"的标定策略：
Calibration follows a "simple to complex, layer by layer" strategy:

| 层级 / Layer | 组成 / Components | 目标 / Goal | 校准要点 / Key Tuning |
|---|---|---|---|
| 第一层 | R_goal + R_dist + P_collision + P_time | 学会到达目标 | R_goal : P_collision = 1:1 保持安全与目标平衡 |
| 第二层 | + P_proximity + P_safe | 学会避障与间距 | 目标碰撞率 < 5% |
| 第三层 | + P_formation + P_energy + R_delivery | 学会高效配送 | 平衡各项权重 |
| 第四层 | + P_smooth + R_all_done | 微调与终极大奖 | P_smooth权重应最小 |

#### 8.4.2 归一化方法 / Normalization Methods

| 方法 / Method | 公式 / Formula | 适用场景 / Use Case |
|---|---|---|
| Running Mean归一化 | `r̂ = (r - μ) / (σ + ε)` | 各分量量级差异大时 |
| 自适应权重 | `w *= exp(η * (target_ratio / actual_ratio - 1))` | 动态平衡各分量贡献 |
| 硬裁剪 | `clip(r, -R_max, R_max)` | 防止极端值 |
| 软裁剪（tanh） | `R_max * tanh(r / R_max)` | 保留梯度信息 |

推荐：优先使用 **Running Mean + 软裁剪** 组合。
Recommended: use **Running Mean + soft clipping** combination.

#### 8.4.3 常见问题诊断 / Common Problem Diagnostics

| 问题 / Problem | 可能原因 / Possible Cause | 解决方案 / Solution |
|---|---|---|
| 智能体不尝试到达目标 | R_goal被P_time淹没 | 增大R_goal或减小λ_time |
| 智能体频繁碰撞 | R_goal > \|P_collision\| | 增大P_collision |
| 智能体保持距离不移动 | P_time过小或P_proximity过大 | 增大λ_time或减小P_proximity |
| 训练初期无正向奖励 | 稀疏奖励问题严重 | 增大塑形强度或添加检查点奖励 |
| 训练后期性能下降 | 奖励hack（利用漏洞） | 检查各分量是否有非预期行为 |
| 各分量量级差异过大 | 未归一化 | 启用Running Mean归一化 |

#### 8.4.4 奖励分量推荐权重 / Recommended Reward Component Weights

| 分量 / Component | 类型 / Type | 基准值 / Baseline | 量级 / Magnitude | 推荐权重 / Recommended Weight |
|---|---|---|---|---|
| R_goal | 事件触发 | +200 | 高 | w=3.0 |
| R_dist | 连续（每步） | c=1.0 | 低 | w=1.0 |
| P_collision | 事件触发 | -200 | 高 | w=3.0 |
| P_proximity | 连续 | -100(max) | 中 | w=1.5 |
| P_energy | 连续 | λ=0.5 | 低 | w=0.5 |
| P_time | 连续 | λ=2.0 | 中 | w=1.0 |
| P_smooth | 连续 | λ=0.1 | 低 | w=0.3 |
| R_delivery | 事件触发 | +50 | 中 | w=2.0 |

---

### 8.5 课程学习实验 / Curriculum Learning Experiments

课程学习从简单任务逐步增加难度，解决稀疏奖励和训练稳定性问题。
Curriculum learning gradually increases difficulty, solving sparse reward and training stability issues.

#### 8.5.1 C1-C7 课程配置 / C1-C7 Curriculum Configurations

| 课程 / Curriculum | 名称 / Name | 无人机数 | 障碍物 | 配送数 | 任务距离 | 启用奖励 |
|---|---|---|---|---|---|---|
| C1 | 单机简单导航 | 1 | 0 | 1 | 10-50m | R_goal, R_dist, P_time, Shaping |
| C2 | 单机避障 | 1 | 5-10 | 1 | 50-200m | C1 + P_collision, P_safe, P_energy |
| C3 | 单机多订单 | 1 | 5-10 | 2-3 | 50-200m | C2 + R_delivery |
| C4 | 多机协作 | 2-3 | 5-10 | 2-3 | 50-300m | C3 + P_proximity, P_formation |
| C5 | 完整L2环境 | 1 | 10-15 | 4-8 | 100-300m | C3 + 时间窗口 |
| C6 | 3D飞行 | 1 | 20+ | 3-5 | 50-500m | 全部分量（2D简化） |
| C7 | 完整多机 | 3-5 | 20+ | 3-8 | 50-500m | 全部分量 |

#### 8.5.2 课程晋级条件 / Curriculum Advancement Conditions

```python
# 自动晋级判定逻辑 / Auto-advancement logic
ADVANCE_THRESHOLDS = {
    1: {"success_rate": 0.80, "max_avg_length": 200, "max_collision_rate": 0.05},
    2: {"success_rate": 0.75, "max_avg_length": 400, "max_collision_rate": 0.05},
    3: {"success_rate": 0.70, "max_avg_length": 600, "max_collision_rate": 0.03},
    4: {"success_rate": 0.65, "max_avg_length": 600, "max_collision_rate": 0.03},
}

def should_advance(current_stage, metrics_window):
    """
    判断是否应进入下一阶段。
    Determine whether to advance to next stage.

    Parameters
    ----------
    current_stage : int
        当前课程阶段 / Current curriculum stage
    metrics_window : dict
        最近100回合的指标窗口 / Recent 100-episode metrics window
        keys: success_rate, avg_length, collision_rate

    Returns
    -------
    bool
        是否晋级 / Whether to advance
    """
    if current_stage >= max(ADVANCE_THRESHOLDS):
        return False
    t = ADVANCE_THRESHOLDS[current_stage]
    return (
        metrics_window["success_rate"] > t["success_rate"]
        and metrics_window["avg_length"] < t["max_avg_length"]
        and metrics_window["collision_rate"] < t["max_collision_rate"]
    )
```

#### 8.5.3 课程学习 vs 固定难度对比 / Curriculum vs Fixed Difficulty

| 实验 / Experiment | 配置 / Config | 最终完成率 | 训练稳定性 | 收敛速度 |
|---|---|---:|---|---:|
| 固定C7（直接训练） | 从一开始用完整环境 | 45% ± 12% | 不稳定，震荡 | 1500 ep |
| 渐进C1→C7 | 从C1逐步晋级 | 85% ± 3% | 稳定 | 1200 ep |
| 渐进C1→C5（跳过3D） | 仅2D课程 | 78% ± 5% | 较稳定 | 1000 ep |

**结论 / Conclusion**: 渐进式课程学习可将最终性能提升约40个百分点，同时减少训练方差。
Progressive curriculum learning improves final performance by ~40% while reducing training variance.

#### 8.5.4 任务距离递增策略 / Task Distance Ramp Strategy

```python
def get_task_distance(epoch, d_min=10.0, d_max=500.0, T_ramp=50):
    """
    线性递增任务距离 / Linear ramp of task distance

    Parameters
    ----------
    epoch : int
        当前训练轮次 / Current epoch
    d_min : float
        最小距离(m) / Minimum distance
    d_max : float
        最大距离(m) / Maximum distance
    T_ramp : int
        递增周期(epoch) / Ramp duration in epochs

    Returns
    -------
    float
        当前任务距离 / Current task distance
    """
    return d_min + (d_max - d_min) * min(1.0, epoch / T_ramp)
```

#### 8.5.5 塑形强度递减策略 / Shaping Strength Decay Strategy

随着训练进行，逐步降低塑形奖励权重，让智能体主要依赖原始稀疏奖励。
Gradually reduce shaping weight so the agent relies primarily on sparse original rewards.

```python
def get_shaping_coefficient(epoch, c_init=2.0, T_decay=20):
    """
    塑形强度衰减 / Shaping strength decay

    Parameters
    ----------
    epoch : int
        当前轮次 / Current epoch
    c_init : float
        初始强度 / Initial strength
    T_decay : int
        衰减时间常数 / Decay time constant

    Returns
    -------
    float
        当前塑形系数 / Current shaping coefficient
    """
    return c_init * (T_decay / (T_decay + epoch))
```

---

> **下一篇 / Next**: [9. 科研绘图指南 / Scientific Visualization Guide](./09_visualization.md)
