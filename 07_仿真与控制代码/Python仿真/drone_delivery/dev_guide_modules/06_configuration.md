"""
模块06 — 配置文件格式详解 / Module 06 — Configuration File Format Guide

职责 / Responsibility:
    TOML配置格式规范、Python配置类设计、算法路由配置、各层级推荐超参数表和环境配置速查。
    TOML format spec, Python Config class design, algorithm routing config,
    per-level recommended hyperparameters, and environment config quick reference.

参考 / Reference:
    drone_delivery/agent_q_learning/conf/train_env_conf.toml (L1 TOML示例)
    drone_delivery/agent_dqn/conf/train_env_conf.toml (L2 TOML示例)
    drone_delivery/agent_a2c/conf/train_env_conf.toml (L2/L3 TOML示例)
    drone_delivery/agent_q_learning/conf/conf.py (Config类示例)
    drone_delivery/conf/algo_conf_drone_delivery.toml (算法路由)
    drone_delivery/conf/app_conf_drone_delivery.toml (应用配置)
    drone_delivery/conf/configure_app.toml (全局训练配置)
    config.py (全局配置字典)

领域适配 / Domain Adaptation:
    峡谷漫步v2 → 无人机配送:
    - start/end坐标 → depot/merchants/delivery_points
    - max_step → max_steps (500/800/1000)
    - treasure_id → delivery_count / num_orders
    - 64x64地图 → 40x40网格
"""

---

## 6. 配置文件格式详解 / Configuration File Format Guide

> **版本 / Version**: v2.0  
> **日期 / Date**: 2026-07-11  
> **前置模块 / Prerequisites**: 模块01（环境安装与配置）

---

### 6.1 TOML配置格式规范 / TOML Configuration Format

#### TOML基本语法 / TOML Basic Syntax

```toml
# 注释用 # 号开头 / Comments start with #
# 中文注释在上，英文在下 / Chinese comment above, English below

[section]
key = value              # 键值对 / Key-value pair
string_key = "text"      # 字符串 / String
array_key = [1, 2, 3]    # 数组 / Array
bool_key = true          # 布尔值 / Boolean
```

#### 双语注释规范 / Bilingual Comment Convention

每个TOML配置项必须包含双语注释，格式如下：

```toml
# 中文说明 — 详细描述
# English description — detailed explanation
# 类型 / Type: <类型>, 范围 / Range: <范围>
key = value
```

#### train_env_conf.toml 完整结构（L1示例）/ Complete Structure (L1 Example)

文件位置 / Location: `agent_q_learning/conf/train_env_conf.toml`

```toml
[env_conf]
# 配送中心坐标 [x, y]
# Depot coordinates [x, y]
# 无人机起始位置和全部完成后返回的位置。
# The drone's starting position and return position after all deliveries.
depot = [30, 10]

# 商家坐标列表 (取货点)
# Merchant coordinate list (pickup points)
# 每个子数组为 [x, y]，长度与 delivery_points 相同。
# Each sub-array is [x, y]; length matches delivery_points.
merchants = [
    [16, 17],
    [22, 15],
    [14, 12],
    [25, 18],
    [18, 24],
]

# 配送点坐标列表 (送货点)
# Delivery point coordinate list (drop-off points)
# 每个子数组为 [x, y]，与 merchants 按索引一一对应。
# Each sub-array is [x, y]; one-to-one correspondence with merchants by index.
delivery_points = [
    [20, 25],
    [28, 22],
    [10, 20],
    [32, 12],
    [16, 17],
]

# 订单数量 / Number of delivery orders
# 整型，取值范围 1~10。
# Integer, range: 1~10.
delivery_count = 5

# 最大步数 / Maximum steps per episode
# 整型，取值范围 1~2000。
# Integer, range: 1~2000.
max_steps = 500

# 网格尺寸 / Grid size (grid_size x grid_size)
# 整型，固定值。
# Integer, fixed value.
grid_size = 40
```

#### 各层级TOML示例 / Per-Level TOML Examples

**L1: Q-learning / SARSA / Monte Carlo — 固定商家**

```toml
[env_conf]
depot = [30, 10]
merchants = [[16, 17], [22, 15], [14, 12], [25, 18], [18, 24]]
delivery_points = [[20, 25], [28, 22], [10, 20], [32, 12], [16, 17]]
delivery_count = 5
max_steps = 500
grid_size = 40
```

**L2: DQN / A2C — 随机商家**

```toml
[env_conf]
depot = [30, 10]
max_steps = 800          # DQN需要较长回合积累样本
min_orders = 4           # 每回合最小订单数
max_orders = 8           # 每回合最大订单数
max_carry = 2            # 最大同时携带订单数
grid_size = 40
```

**L3/L4: PPO+CTDE — 3D环境（由WUHAN_ENV_CONFIG控制）**

```toml
# L3/L4的环境配置由 config.py 中的 WUHAN_ENV_CONFIG 字典控制
# L3/L4 env config is controlled by WUHAN_ENV_CONFIG dict in config.py
# TOML仅配置算法超参数 / TOML only configures algorithm hyperparameters

[algo_conf]
learning_rate = 3e-4
gamma = 0.99
gae_lambda = 0.95
clip_epsilon = 0.2
```

---

### 6.2 Python配置类设计 / Python Config Class Design

#### Config类模板（纯类变量）/ Config Class Template (Pure Class Variables)

每个Agent的`conf/conf.py`必须使用纯类变量的Config类，不使用实例变量：

```python
"""
Q-learning配置模块 / Q-learning Configuration Module

职责 / Responsibility:
    定义Q-learning算法和环境的所有超参数与常量。
参考 / Reference:
    kaiwu_gorge_walk_v2/agent_q_learning/conf/conf.py
领域适配 / Domain Adaptation:
    hero_pos → drone_pos, treasure → delivery_order
    max_step=1000 → max_steps=500
    treasure_score=100 → delivery_reward=+100
    end_score=150 → safe_return_bonus=+500
"""


class Config:
    """
    Q-learning配置类 / Q-learning Configuration Class

    State space / 状态空间:
        position(40x40=1600) * order_status(2^5=32) * carrying(2) = 102,400
    """

    # ============================================================
    # 核心超参数 / Core Hyper-parameters
    # ============================================================
    GAMMA = 0.99                      # 折扣因子 / Discount factor
    LEARNING_RATE = 0.1               # 学习率 / Learning rate

    # ============================================================
    # Epsilon-greedy 探索策略 / Exploration Strategy
    # ============================================================
    EPSILON_START = 1.0               # 初始探索率 / Initial exploration rate
    EPSILON_MIN = 0.05                # 最小探索率 / Minimum exploration rate
    EPSILON_DECAY = 0.9995            # 每回合衰减因子 / Per-episode decay factor

    # ============================================================
    # 状态空间与动作空间 / State and Action Space
    # ============================================================
    GRID_SIZE = 40                    # 网格尺寸 / Grid size
    NUM_ORDERS = 5                    # 订单数量 / Number of orders
    STATE_SIZE = 40 * 40 * (2 ** 5) * 2  # = 102,400
    ACTION_SIZE = 4                   # 4方向: 上/下/左/右 / 4-directional

    # ============================================================
    # 环境配置 / Environment Configuration
    # ============================================================
    MAX_STEPS = 500                   # 每回合最大步数 / Max steps per episode
    DEPOT = [30, 10]                  # 配送中心坐标 / Depot coordinates

    # ============================================================
    # 训练配置 / Training Configuration
    # ============================================================
    NUM_EPISODES = 10000              # 最大训练回合数 / Maximum training episodes
    EVAL_INTERVAL = 100               # 评估间隔 / Evaluation interval
    EVAL_EPISODES = 10                # 每次评估回合数 / Episodes per evaluation

    # ============================================================
    # 奖励设计参数 / Reward Design Parameters
    # ============================================================
    DELIVERY_REWARD = 100.0           # 完成一单配送奖励 / Single delivery reward
    SAFE_RETURN_BONUS = 500.0         # 全部完成+返航奖励 / All done + return bonus
    STEP_PENALTY = -1.0               # 每步时间惩罚 / Per-step time penalty
    COLLISION_PENALTY = -200.0        # 碰撞惩罚 / Collision penalty

    # ============================================================
    # 模型保存 / Model Saving
    # ============================================================
    MODEL_SAVE_INTERVAL = 300         # 保存最小间隔(秒) / Min save interval (s)
    MAX_SAVE_COUNT = 100              # 最大保存次数 / Max save count

    # ============================================================
    # 观测特征维度 / Observation Feature Dimension
    # ============================================================
    OBSERVATION_SHAPE = 55            # 观测特征向量维度 / Obs feature vector dim
```

#### config.py 全局配置字典详解 / Global Config Dict Detailed

`config.py` 中的全局配置分为8个字典，按用途组织：

**ENV_CONFIG — 3D环境基础参数**

```python
ENV_CONFIG = {
    "space_size": (100, 100, 50),   # 空域大小 (x, y, z)
    "num_uavs": 3,                  # 无人机数量
    "num_targets": 5,               # 配送目标数量
    "num_obstacles": 10,            # 障碍物数量
    "dt": 0.1,                      # 时间步长 (s)
    "max_steps": 1000,              # 最大步数
    "v_max": 15.0,                  # 最大速度 (m/s)
    "a_max": 5.0,                   # 最大加速度 (m/s^2)
    "z_min": 10.0,                  # 最低飞行高度 (m)
    "z_max": 50.0,                  # 最高飞行高度 (m)
    "d_safe_uav": 10.0,             # UAV间安全距离 (m)
    "d_safe_obstacle": 5.0,         # 障碍物安全距离 (m)
}
```

**PPO_CONFIG — PPO算法超参数**

```python
PPO_CONFIG = {
    "learning_rate": 3e-4,      # 学习率
    "min_learning_rate": 1e-5,  # 余弦退火下限
    "gamma": 0.99,              # 折扣因子
    "gae_lambda": 0.95,         # GAE参数
    "clip_epsilon": 0.2,        # 裁剪参数
    "entropy_coef": 0.01,       # 熵系数
    "value_coef": 0.5,          # 价值系数
    "max_grad_norm": 0.5,       # 梯度裁剪
    "batch_size": 256,          # 批量大小
    "hidden_dim": 256,          # 隐藏层维度
    "update_epochs": 4,         # 每epoch更新次数
    "buffer_size": 10000,       # 经验池大小
}
```

#### 超参数优先级 / Hyperparameter Priority

```text
算法conf.py Config类 > 全局config.py字典 > 硬编码默认值
```

配置读取顺序：

```python
# 1. 尝试从算法Config类读取
value = getattr(Config, 'LEARNING_RATE', None)

# 2. 若无，从全局字典读取
if value is None:
    value = PPO_CONFIG.get('learning_rate', 3e-4)

# 3. 若无，使用硬编码默认值
if value is None:
    value = 3e-4
```

---

### 6.3 算法路由配置 / Algorithm Routing Configuration

#### algo_conf_drone_delivery.toml 完整内容 / Complete Algorithm Routing

文件位置 / Location: `drone_delivery/conf/algo_conf_drone_delivery.toml`

```toml
# 无人机外卖配送 — 算法路由配置
# Drone Food Delivery — Algorithm Routing Configuration

[q_learning]
# Q-learning智能体 / Q-learning Agent
actor_agent = "agent_q_learning.agent.Agent"
learner_agent = "agent_q_learning.agent.Agent"
aisrv_agent = "agent_q_learning.agent.Agent"
train_workflow = "agent_q_learning.workflow.train_workflow.workflow"
eval_workflow = "agent_q_learning.workflow.train_workflow.workflow"
exam_workflow = "agent_q_learning.workflow.train_workflow.workflow"

[sarsa]
# SARSA智能体 / SARSA Agent
actor_agent = "agent_sarsa.agent.Agent"
learner_agent = "agent_sarsa.agent.Agent"
aisrv_agent = "agent_sarsa.agent.Agent"
train_workflow = "agent_sarsa.workflow.train_workflow.workflow"
eval_workflow = "agent_sarsa.workflow.train_workflow.workflow"
exam_workflow = "agent_sarsa.workflow.train_workflow.workflow"

[monte_carlo]
# Monte Carlo智能体 / Monte Carlo Agent
actor_agent = "agent_monte_carlo.agent.Agent"
learner_agent = "agent_monte_carlo.agent.Agent"
aisrv_agent = "agent_monte_carlo.agent.Agent"
train_workflow = "agent_monte_carlo.workflow.train_workflow.workflow"
eval_workflow = "agent_monte_carlo.workflow.train_workflow.workflow"
exam_workflow = "agent_monte_carlo.workflow.train_workflow.workflow"

[dqn]
# DQN智能体 / DQN Agent
actor_agent = "agent_dqn.agent.Agent"
learner_agent = "agent_dqn.agent.Agent"
aisrv_agent = "agent_dqn.agent.Agent"
train_workflow = "agent_dqn.workflow.train_workflow.workflow"
eval_workflow = "agent_dqn.workflow.train_workflow.workflow"
exam_workflow = "agent_dqn.workflow.train_workflow.workflow"

[a2c]
# A2C智能体 / A2C Agent
actor_agent = "agent_a2c.agent.Agent"
learner_agent = "agent_a2c.agent.Agent"
aisrv_agent = "agent_a2c.agent.Agent"
train_workflow = "agent_a2c.workflow.train_workflow.workflow"
eval_workflow = "agent_a2c.workflow.train_workflow.workflow"
exam_workflow = "agent_a2c.workflow.train_workflow.workflow"

[diy]
# DIY智能体（空白模板） / DIY Agent (blank template)
actor_agent = "agent_diy.agent.Agent"
learner_agent = "agent_diy.agent.Agent"
aisrv_agent = "agent_diy.agent.Agent"
train_workflow = "agent_diy.workflow.train_workflow.workflow"
eval_workflow = "agent_diy.workflow.train_workflow.workflow"
exam_workflow = "agent_diy.workflow.train_workflow.workflow"
```

#### 新增算法注册步骤（5步）/ Register New Algorithm (5 Steps)

以新增`agent_ppo`为例：

**步骤1：创建Agent目录 / Create Agent Directory**

```powershell
mkdir drone_delivery\agent_ppo
mkdir drone_delivery\agent_ppo\agent.py
mkdir drone_delivery\agent_ppo\algorithm
mkdir drone_delivery\agent_ppo\conf
mkdir drone_delivery\agent_ppo\feature
mkdir drone_delivery\agent_ppo\model
mkdir drone_delivery\agent_ppo\workflow
```

**步骤2：编写conf/conf.py / Write conf/conf.py**

```python
class Config:
    """PPO配置类 / PPO Configuration Class"""
    LEARNING_RATE = 3e-4
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIP_EPSILON = 0.2
    # ... 其他参数
```

**步骤3：编写conf/train_env_conf.toml / Write conf/train_env_conf.toml**

```toml
[env_conf]
depot = [30, 10]
max_steps = 500
grid_size = 40
# ... 根据层级配置
```

**步骤4：注册到algo_conf / Register in algo_conf**

在`drone_delivery/conf/algo_conf_drone_delivery.toml`末尾添加：

```toml
[ppo]
actor_agent = "agent_ppo.agent.Agent"
learner_agent = "agent_ppo.agent.Agent"
aisrv_agent = "agent_ppo.agent.Agent"
train_workflow = "agent_ppo.workflow.train_workflow.workflow"
eval_workflow = "agent_ppo.workflow.train_workflow.workflow"
exam_workflow = "agent_ppo.workflow.train_workflow.workflow"
```

**步骤5：添加冒烟测试 / Add Smoke Test**

在`drone_delivery/train_test.py`中注册冒烟测试函数。

#### app_conf_drone_delivery.toml 说明 / Application Config Description

文件位置 / Location: `drone_delivery/conf/app_conf_drone_delivery.toml`

```toml
[app]
# RL辅助类 / RL helper class
rl_helper = "common_python.utils.rl_helper.RLHelper"

# 策略构建器 / Policy builder
policy_builder = "common_python.utils.policy_builder.PolicyBuilder"

# 当前使用的算法 / Current active algorithm
algorithm_name = "q_learning"

# 默认层级 / Default level
default_level = "level1"
```

---

### 6.4 各层级推荐超参数表 / Recommended Hyperparameters

#### L1: Q-learning / SARSA / Monte Carlo

| 参数 | Q-learning | SARSA | Monte Carlo |
|---|---|---|---|
| LEARNING_RATE | 0.1 | 0.1 | 0.01 |
| GAMMA | 0.99 | 0.99 | 0.99 |
| EPSILON_START | 1.0 | 1.0 | 1.0 |
| EPSILON_MIN | 0.05 | 0.05 | 0.05 |
| EPSILON_DECAY | 0.9995 | 0.9995 | 0.9995 |
| NUM_EPISODES | 10000 | 10000 | 10000 |
| MAX_STEPS | 500 | 500 | 500 |
| GRID_SIZE | 40 | 40 | 40 |

#### L2: DQN / A2C

| 参数 | DQN | A2C |
|---|---|---|
| LEARNING_RATE | 1e-4 | 3e-4 |
| GAMMA | 0.99 | 0.99 |
| BATCH_SIZE | 64 | 128 |
| MAX_STEPS | 800 | 800 |
| MIN_ORDERS | 4 | 4 |
| MAX_ORDERS | 8 | 8 |
| MAX_CARRY | 2 | 2 |
| HIDDEN_DIM | 128 | 256 |

#### L3/L4: PPO+CTDE（参考config.py）

| 参数 | 值 | 来源 |
|---|---|---|
| LEARNING_RATE | 3e-4 | config.py PPO_CONFIG |
| MIN_LEARNING_RATE | 1e-5 | config.py PPO_CONFIG |
| GAMMA | 0.99 | config.py PPO_CONFIG |
| GAE_LAMBDA | 0.95 | config.py PPO_CONFIG |
| CLIP_EPSILON | 0.2 | config.py PPO_CONFIG |
| ENTROPY_COEF | 0.01 | config.py PPO_CONFIG |
| VALUE_COEF | 0.5 | config.py PPO_CONFIG |
| MAX_GRAD_NORM | 0.5 | config.py PPO_CONFIG |
| BATCH_SIZE | 256 | config.py PPO_CONFIG |
| HIDDEN_DIM | 256 | config.py PPO_CONFIG |
| UPDATE_EPOCHS | 4 | config.py PPO_CONFIG |
| BUFFER_SIZE | 10000 | config.py PPO_CONFIG |

#### 调参建议 / Tuning Tips

- **学习率**：从3e-4开始，若训练不稳定则降至1e-4
- **gamma**：配送任务通常用0.99，若需要更关注短期奖励则降至0.95
- **epsilon衰减**：确保训练后期epsilon接近0.05，过快衰减会减少探索
- **batch_size**：DQN用64，A2C/PPO用128-256
- **hidden_dim**：L1/L2用128，L3/L4用256

---

### 6.5 环境配置速查表 / Environment Config Quick Reference

#### 四层级环境参数统一对比表 / 4-Level Environment Parameter Comparison

| 参数 | L1 | L2 | L3 | L4 |
|---|---|---|---|---|
| 地图 | 40x40网格 | 40x40网格 | 3D城市空域 | 3D城市空域 |
| 动作空间 | Discrete(4) | Discrete(8) | Box(3) | Box(4) |
| 订单 | 5个固定 | 4-8个随机 | 固定 | 随机 |
| 时间窗口 | 无 | 有 | 有 | 有 |
| 多单携带 | 1单 | 2单 | 2单 | 2单+ |
| max_steps | 500 | 800 | 由WUHAN_ENV_CONFIG控制 | 同L3 |
| 障碍物 | 建筑+水域+电力线 | 同L1 | +禁飞区+风力 | +动态禁飞 |
| 无人机数量 | 1 | 1 | 1 | 1-3 |

#### WUHAN_ENV_CONFIG 关键参数说明 / Key Parameter Description

| 参数 | 值 | 说明 / Description |
|---|---|---|
| `zone_id` | "A" | A/B/C/D 四个分区 |
| `num_drones` | 1 | 单机训练（多机用3） |
| `physics_dt_s` | 0.1 | 物理积分周期 |
| `control_dt_s` | 0.5 | 策略动作周期 |
| `max_episode_time_s` | 900.0 | 单分区时间上限(s) |
| `min_altitude_m` | 10.0 | 最低飞行高度 |
| `max_altitude_m` | 200.0 | 最高飞行高度 |
| `max_speed_xy_mps` | 15.0 | 水平最大速度(m/s) |
| `max_speed_z_mps` | 5.0 | 垂直最大速度(m/s) |
| `safety_clearance_m` | 8.0 | 障碍物安全净空(m) |
| `battery_capacity_wh` | 250.0 | 电池容量(Wh) |
| `reserve_soc` | 0.15 | 返航保留电量比例 |
| `return_time_margin_s` | 30.0 | 返航时间安全裕量(s) |
| `return_route_factor` | 1.25 | 直线到可飞路径保守系数 |

#### 奖励参数8分量速查 / Reward 8-Component Quick Reference

| 分量 | 配置键 | 值 | 触发方式 | 说明 |
|---|---|---|---|---|
| 目标到达 | delivery_reward_single | +100 | 事件触发 | 完成一单配送 |
| 安全返航 | delivery_reward_all | +500 | 事件触发 | 全部完成+返航 |
| 距离惩罚 | distance_weight | -0.1/m | 连续型 | 每步距离引导 |
| 时间惩罚 | time_penalty | -1.0/step | 连续型 | 每步时间惩罚 |
| 碰撞惩罚 | collision_weight | -10.0/次 | 事件触发 | 碰撞障碍物 |
| 能耗惩罚 | energy_weight | -0.05/J | 连续型 | 电池消耗 |
| 协同奖励 | cooperation_weight | +1.0/次 | 事件触发 | 多机协作 |
| 安全距离 | safety_weight | -5.0/次 | 连续型 | 低于安全净空 |
| 边界惩罚 | boundary_weight | -20.0/次 | 事件触发 | 超出空域边界 |

*来源 / Source: `config.py` REWARD_CONFIG*

---

*模块06结束 / End of Module 06*  
*下一模块 / Next: 模块07 — 奖励系统设计 / Module 07 — Reward System Design*
