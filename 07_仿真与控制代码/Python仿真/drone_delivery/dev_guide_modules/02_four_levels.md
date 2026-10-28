"""
职责 / Responsibility: 四层级问题框架完整规格说明，覆盖L1-L4的状态空间、动作空间、约束条件与数学模型
参考 / Reference: dev_guide.md (v1.0) §1.2-§2.2、PPO开发指南v3.0 §场景一览、config.py WUHAN_ENV_CONFIG、reward_function_design.md
领域适配 / Domain Adaptation: 峡谷漫步v2单层固定地图 → 无人机配送四层渐进式约束扩展
"""

---

## 2. 四层级问题框架 / Four-Level Problem Framework

本项目采用**渐进式四层级**设计，从最简单的2D固定商家网格逐步扩展到完整的3D城市多机协同配送系统。每个层级在前一层级基础上新增一组约束条件，帮助学习者循序渐进地掌握强化学习在无人机配送场景中的应用。

This project uses a **progressive 4-level** design, scaling from the simplest 2D fixed-merchant grid to a full 3D multi-drone cooperative city delivery system. Each level adds a new set of constraints on top of the previous one, helping learners gradually master RL applications in drone delivery scenarios.

**层级总览 / Level Overview**:

| 层级 / Level | 名称 / Name | 动作空间 / Action Space | 关键约束 / Key Constraints | 推荐算法 / Algorithms |
|:---|:---|:---:|:---|:---|
| L1 | 2D固定商家 / 2D Fixed | Discrete(4) | 地图边界+障碍物 / boundary+obstacles | Q-learning, SARSA, MC |
| L2 | 2D随机商家 / 2D Random | Discrete(8) | +时间窗口+多单携带 / +time+multi-carry | DQN, A2C |
| L3 | 3D固定商家 / 3D Fixed | Box(3) 连续 / continuous | +禁飞区+能耗+碰撞 / +NFZ+energy+collision | A2C, PPO |
| L4 | 3D随机+全约束 / 3D Full | Box(4) 连续 / continuous | +多机+风+通信+动态禁飞 / +multi+wind+comm | PPO+CTDE |

---

### 2.1 L1: 2D固定商家 / Level 1: 2D Fixed Merchants

#### 2.1.1 问题定义与数学模型 / Problem Definition & Mathematical Model

最基础的配送模型，对应外卖骑手在已知区域内配送固定订单。智能体在40x40网格地图上，以4方向离散动作，从配送中心出发，依次完成5个固定订单的取货-送货流程，最终返回配送中心。

The most basic delivery model — a rider delivering known orders in a fixed area. The agent operates on a 40x40 grid with 4-directional discrete actions, departing from a depot, sequentially completing pickup-delivery for 5 fixed orders, and returning to the depot.

**状态空间 S / State Space**:

$$
\mathcal{S} = \mathcal{S}_{pos} \times \mathcal{S}_{order} \times \mathcal{S}_{carry}
$$

| 分量 / Component | 空间 / Space | 大小 / Size | 说明 / Description |
|:---|:---|:---:|:---|
| $\mathcal{S}_{pos}$ | $\{0, \ldots, 39\}^2$ | 1,600 | 无人机网格坐标 / drone grid position |
| $\mathcal{S}_{order}$ | $\{0, 1\}^5$ | 32 | 订单完成状态掩码 / order completion mask |
| $\mathcal{S}_{carry}$ | $\{0, 1\}$ | 2 | 是否携带货物 / carrying payload |

状态空间总大小 / Total state space size: $1600 \times 32 \times 2 = 102{,}400$

**动作空间 A / Action Space**:

$$
\mathcal{A} = \text{Discrete}(4) = \{0:\text{UP}, \; 1:\text{DOWN}, \; 2:\text{LEFT}, \; 3:\text{RIGHT}\}
$$

| 动作 / Action | 值 / Value | 位移向量 / Delta |
|:---|:---:|:---:|
| 上 / UP | 0 | $(0, +1)$ |
| 下 / DOWN | 1 | $(0, -1)$ |
| 左 / LEFT | 2 | $(-1, 0)$ |
| 右 / RIGHT | 3 | $(+1, 0)$ |

**约束条件 C / Constraints**:

| 约束 / Constraint | 类型 / Type | 说明 / Description |
|:---|:---:|:---|
| 地图边界 / Map boundary | 硬约束 / hard | $x, y \in [0, 39]$ |
| 障碍物 / Obstacles | 硬约束 / hard | 建筑、水域、电力线区域不可通行 |
| 取货顺序 / Pickup order | 逻辑 / logical | 必须先到商家取货，再到配送点送货 |
| 步数限制 / Step limit | 软约束 / soft | $T_{max} = 500$ 步后截断 / truncated after 500 steps |

**目标函数 J / Objective Function**:

$$
J = \sum_{t=0}^{T} r_t = \sum_{k=1}^{5} \mathbb{1}[\text{delivery}_k] \times 100 + \mathbb{1}[\text{safe\_return}] \times 500 - T
$$

| 奖励事件 / Event | 分值 / Points | 触发条件 / Trigger |
|:---|:---:|:---|
| 完成一单配送 / Complete delivery | +100 | 从商家取货后送达配送点 |
| 全部完成+安全返航 / All done + return | +500 | 5单全部完成且到达配送中心 |
| 每步时间惩罚 / Per-step penalty | -1 | 每执行一步 |
| 碰撞障碍物 / Obstacle collision | -200 | 移动到障碍物格子（终止） |

#### 2.1.2 Simple2DDeliveryEnv 接口 / Simple2DDeliveryEnv Interface

**文件位置 / File Location**: `environment/simple_delivery_env.py`

```python
class Simple2DDeliveryEnv(gym.Env):
    """
    问题层级一：2D固定商家配送环境
    Level 1: 2D fixed merchant delivery environment

    40x40 grid, 5 fixed orders, Discrete(4) actions.
    Based on Wuhan College Zone B building layout.
    """
```

**reset 方法 / reset Method**:

```python
def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None
          ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    # Returns: (obs_dict, info_dict)
    # 返回: (观测字典, 信息字典)
```

| 返回值 / Return | 类型 / Type | 说明 / Description |
|:---|:---|:---|
| `observation` | `dict` | `{position: (2,), order_status: (5,), carrying: int}` |
| `info` | `dict` | `{position, completed, carrying, steps, depot_reached}` |

**step 方法 / step Method**:

```python
def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
    # Returns: (obs, reward, terminated, truncated, info)
    # 返回: (观测, 奖励, 自然终止, 时间截断, 信息)
```

| 参数 / Parameter | 类型 / Type | 说明 / Description |
|:---|:---:|:---|
| `action` | `int` | 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT |

| 返回值 / Return | 类型 / Type | 说明 / Description |
|:---|:---:|:---|
| `observation` | `dict` | 执行后的观测 / observation after action |
| `reward` | `float` | 本步奖励 / step reward |
| `terminated` | `bool` | 自然终止（完成/碰撞）/ natural termination |
| `truncated` | `bool` | 时间截断（超max_steps）/ time truncation |
| `info` | `dict` | 审计信息 / audit info |

#### 2.1.3 状态编码公式 / State Encoding Formula

用于表格型RL（Q-learning、SARSA、Monte Carlo）的唯一状态键：

Unique state key for tabular RL (Q-learning, SARSA, Monte Carlo):

$$
\text{state\_key} = x \cdot G + y
$$

$$
\text{state\_key}_{encoded} = (\text{state\_key} \ll 6) \mid (\text{order\_bits} \ll 1) \mid \text{carrying}
$$

其中 $G = 40$ 为网格宽度，`order_bits` 为5位订单状态的二进制编码。编码后的 `state_key` 取值范围为 $[0, 102399]$。

Where $G = 40$ is the grid width, `order_bits` is the 5-bit binary encoding of order statuses. The encoded `state_key` ranges in $[0, 102399]$.

```python
# state_key = position_x * grid_size + position_y  # 0-1599 (11 bits)
state_key = (state_key << 6) | (order_status_bits << 1) | carrying
```

#### 2.1.4 地图布局说明 / Map Layout Description

基于武汉学院（Zone B）建筑布局简化而来，缩放到40x40网格：

Simplified from Wuhan College (Zone B) building layout, scaled to 40x40 grid:

**配送中心 / Depot**: $(30, 10)$

**障碍物区域 / Obstacle Areas**:

| 地图元素 / Element | 网格区域 / Grid Region | 说明 / Description |
|:---|:---|:---|
| 教学楼群 / Teaching buildings | $[14\!:\!19, \; 14\!:\!21]$ | 武汉学院教学区 / campus teaching area |
| 宿舍楼群 / Dormitory buildings | $[26\!:\!31, \; 20\!:\!25]$ | 学生宿舍区 / student dormitory area |
| 湖区水域 / Lake area | $[8\!:\!13, \; 4\!:\!9]$ | 不可通行水域 / impassable water |
| 电力线路 / Power lines | $[22, \; 12\!:\!17]$ | 横向电力走廊 / horizontal power corridor |
| 小建筑A / Small building A | $[6\!:\!8, \; 22\!:\!24]$ | 附属设施 / ancillary facility |
| 小建筑B / Small building B | $[19\!:\!21, \; 10\!:\!12]$ | 附属设施 / ancillary facility |

**5个固定订单 / 5 Fixed Orders**:

| 订单ID / Order | 商家坐标 / Merchant | 配送点 / Delivery Point |
|:---:|:---:|:---:|
| 0 | $(16, 17)$ | $(20, 25)$ |
| 1 | $(22, 15)$ | $(28, 22)$ |
| 2 | $(14, 12)$ | $(10, 20)$ |
| 3 | $(25, 18)$ | $(32, 12)$ |
| 4 | $(18, 24)$ | $(16, 17)$ |

#### 2.1.5 环境示意图 / Environment Visualization

![武汉学院2D地图](../../../outputs/环境示意图_2D审核/zone_B_lakeside_campus_2d.png)

---

### 2.2 L2: 2D随机商家 / Level 2: 2D Random Merchants

#### 2.2.1 新增约束说明 / New Constraints

层级二在层级一基础上新增四组约束，模拟高峰时段随机涌来的外卖订单：

Level 2 adds four constraint groups on top of Level 1, simulating randomly arriving orders during peak hours:

| 新增约束 / New Constraint | 说明 / Description |
|:---|:---|
| 随机订单 / Random orders | 每回合随机生成4-8个订单，商家和配送点位置随机 |
| 时间窗口 / Time windows | 每个订单有截止时间：$\text{deadline} = \text{dist} \times 1.5 + \text{rand}(50, 200)$ |
| 多单携带 / Multi-order carry | 最多同时携带2个订单 |
| 8方向移动 / 8-directional | 增加4个对角线方向 |

#### 2.2.2 Random2DDeliveryEnv 接口 / Random2DDeliveryEnv Interface

**文件位置 / File Location**: `environment/random_delivery_env.py`

```python
class Random2DDeliveryEnv(gym.Env):
    """
    问题层级二：2D随机商家配送环境
    Level 2: 2D random merchant delivery environment

    40x40 grid, 4-8 random orders per episode, Discrete(8) actions.
    Time windows, multi-order carrying (max 2), vectorized 50-dim observation.
    """
```

**8方向动作 / 8-Directional Actions**:

$$
\mathcal{A} = \text{Discrete}(8)
$$

| 值 / Value | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 方向 / Dir | 上 | 下 | 左 | 右 | 左上 | 右上 | 左下 | 右下 |
| 位移 / Delta | $(0,\!+\!1)$ | $(0,\!-\!1)$ | $(-\!1,\!0)$ | $(+\!1,\!0)$ | $(-\!1,\!+\!1)$ | $(+\!1,\!+\!1)$ | $(-\!1,\!-\!1)$ | $(+\!1,\!-\!1)$ |

**50维向量化观测 / 50-Dim Vectorized Observation**:

固定长度特征向量，适配DQN/A2C神经网络输入 / Fixed-length feature vector for DQN/A2C neural network input:

| 维度范围 / Range | 维数 / Dim | 内容 / Content |
|:---|:---:|:---|
| $[0\!:\!2]$ | 2 | 无人机位置（归一化到$[0,1]$）/ drone position (normalized) |
| $[2\!:\!32]$ | 30 | 订单信息：每单6维（商家x,y、配送x,y、状态、剩余时间）/ order info |
| $[32\!:\!34]$ | 2 | 携带状态 + 步数比例 / carrying + step ratio |
| $[34\!:\!50]$ | 16 | 零填充至固定50维 / zero-padded to fixed 50 |

**额外info字段 / Additional info fields**:

| 键 / Key | 类型 / Type | 说明 / Description |
|:---|:---:|:---|
| `total_orders` | `int` | 本回合订单总数 / total orders this episode |
| `on_time` | `int` | 按时完成的订单数 / on-time completions |
| `failed` | `int` | 超时失败的订单数 / deadline-missed orders |
| `carrying_count` | `int` | 当前携带的订单数 / orders currently carrying |

**目标函数扩展 / Extended Objective**:

$$
J = \sum_{t=0}^{T} r_t = \sum_{k} \left( \mathbb{1}[\text{delivery}_k] \cdot 100 + \mathbb{1}[\text{on\_time}_k] \cdot 50 - \mathbb{1}[\text{missed}_k] \cdot 30 \right) + \mathbb{1}[\text{return}] \cdot 500 - T
$$

#### 2.2.3 L1与L2差异对比表 / L1 vs L2 Detailed Comparison

| 特性 / Feature | L1 | L2 |
|:---|:---:|:---:|
| 动作空间 / Action space | Discrete(4) | Discrete(8) |
| 订单数量 / Orders | 固定5个 / fixed 5 | 随机4-8个 / random 4-8 |
| 时间窗口 / Time windows | 无 / None | 有 / Yes |
| 多单携带 / Multi-carry | 1单 / 1 order | 最多2单 / max 2 |
| 观测空间 / Observation | Dict(3 keys) | Dict(2 keys) + 50-dim vector |
| 最大步数 / Max steps | 500 | 800 |
| 特征维度 / Feature dim | 1 (state_key) | 50 (vector) |
| 推荐算法 / Algorithms | Q-learning, SARSA, MC | DQN, A2C |

---

### 2.3 L3: 3D固定商家 / Level 3: 3D Fixed Merchants

#### 2.3.1 2D到3D维度跃升 / 2D to 3D Dimensional Leap

层级三是从网格世界到真实物理空间的关键跃升，引入以下根本变化：

Level 3 is the critical leap from grid world to real physical space, introducing fundamental changes:

1. **连续动作控制 / Continuous control**: 从离散方向选择变为连续加速度/偏航率指令
2. **物理约束 / Physical constraints**: 建筑物碰撞检测、禁飞区、能耗模型
3. **真实场景 / Real scenes**: 基于OSM数据的四区3D城市环境
4. **3D状态空间 / 3D state**: 166维连续观测向量

#### 2.3.2 四区场景数据 / 4-Zone Scene Data

| 分区 / Zone | 参考区域 / Reference | 建筑数 / Buildings | 平面包络 / Extent | 桥梁/电力线 / Bridges/Poles | 风险区 / Risks | 动态实体 / Dynamic |
|:---|:---|:---:|:---|:---:|:---:|:---:|
| A | 光谷广场 / Guanggu Plaza | 837 | 2400x1800 m | 20 座桥 | 4 | 3 |
| B | 武汉学院 / Wuhan College | 43 | 2400x1800 m | 3 条电力线 | 4 | 3 |
| C | 长江大桥附近 / Yangtze Bridge | 538 | 2800x2000 m | 12 座桥 | 5 | 3 |
| D | 楚河汉界 / Chuhe Hanjie | 592 | 2400x1800 m | 17 座桥 | 5 | 3 |

场景入口文件 / Scene entry: `outputs/环境示意图_3D预览/scene_3d_index.json`

运行时读取各分区的 `zone_*_scene_lod0.json` 与 `zone_*_collision.json`，碰撞语义以 collision JSON 为准。

![四区环境总览](../../../outputs/环境示意图/wuhan_environment_overview.png)

#### 2.3.3 3D状态空间166维分解 / 166-Dim State Space Decomposition

L3观测空间为166维连续向量 / L3 observation is a 166-dim continuous vector:

$$
\mathbf{o} \in \mathbb{R}^{166} = [\mathbf{o}_{pos}; \; \mathbf{o}_{vel}; \; \mathbf{o}_{att}; \; \mathbf{o}_{order}; \; \mathbf{o}_{sensor}; \; \mathbf{o}_{energy}; \; \mathbf{o}_{meta}]
$$

| 分量 / Component | 维数 / Dim | 内容 / Content |
|:---|:---:|:---|
| $\mathbf{o}_{pos}$ | 3 | 3D位置（归一化）/ 3D position (normalized) |
| $\mathbf{o}_{vel}$ | 3 | 3D速度 / 3D velocity |
| $\mathbf{o}_{att}$ | 4 | 姿态四元数 / attitude quaternion |
| $\mathbf{o}_{order}$ | 60 | 12个全局任务点 x 5维 / 12 global task points x 5-dim |
| $\mathbf{o}_{sensor}$ | 36 | LiDAR 36方向测距 / 36-direction LiDAR range |
| $\mathbf{o}_{energy}$ | 4 | 电量SOC、已用时间、载重、航程余量 / battery, time, payload, margin |
| $\mathbf{o}_{meta}$ | 56 | 障碍物编码、禁飞区掩码、风场分量 / obstacles, NFZ mask, wind |

#### 2.3.4 Box(3)连续动作空间 / Box(3) Continuous Action Space

$$
\mathcal{A} = \text{Box}(3) \subset \mathbb{R}^3
$$

| 维度 / Dim | 含义 / Meaning | 范围 / Range | 单位 / Unit |
|:---:|:---|:---:|:---:|
| $a_0$ | XY平面加速度方向 / XY acceleration direction | $[-1, 1]$ | rad |
| $a_1$ | 加速度幅值 / acceleration magnitude | $[0, 1]$ | m/s$^2$ |
| $a_2$ | 垂直速度指令 / vertical velocity command | $[-1, 1]$ | m/s |

环境侧物理映射 / Physical mapping:
- $a_{xy} \in [0, \; a_{max,xy}]$，$a_{max,xy} = 5.0$ m/s$^2$
- $v_z \in [-v_{max,z}, \; v_{max,z}]$，$v_{max,z} = 5.0$ m/s

#### 2.3.5 禁飞区与碰撞检测 / No-Fly Zones and Collision Detection

**硬禁飞区 / Hard No-Fly Zones**:

$$
\mathbf{p} \in \mathcal{V}_{nfz} \implies \text{terminated} = \text{True}, \quad r = -300
$$

其中 $\mathcal{V}_{nfz}$ 为禁飞区体积多边形集合。Where $\mathcal{V}_{nfz}$ is the set of no-fly zone volume polygons.

**建筑碰撞检测 / Building Collision Detection**:

使用连续扫掠体检测（非中心距减半径）/ Using swept-volume detection (not center-distance-minus-radius):

$$
\text{collision}_{build} \iff \exists \, t \in [0, \Delta t]: \; \mathbf{p}(t) \cap \mathcal{B}_{prism} \neq \emptyset
$$

其中 $\mathcal{B}_{prism}$ 为建筑轮廓棱柱体。碰撞惩罚 $-250$。

**净空检查 / Clearance Check**:

$$
\text{clearance}(\mathbf{p}) = \min_{\mathbf{o} \in \mathcal{O}} \|\mathbf{p} - \mathbf{o}_{nearest}\|_2 - r_{drone}
$$

若 $\text{clearance} < d_{safety} = 8.0$ m，施加连续惩罚 $-4$/次。

#### 2.3.6 能耗模型公式 / Energy Model Formula

$$
P_{total} = P_{hover} + P_{climb} + P_{forward} + P_{payload}
$$

各分量 / Components:

$$
P_{hover} = \frac{(mg)^{3/2}}{\sqrt{2\rho A}}, \quad P_{climb} = mg \cdot v_z, \quad P_{forward} = \frac{1}{2}\rho C_D A_{ref} v_{xy}^3
$$

电池约束 / Battery constraint:

$$
E_{remain} = E_{init} - \int_0^t P_{total}(\tau) \, d\tau \geq E_{reserve} = 0.15 \cdot E_{capacity}
$$

其中 $E_{capacity} = 250.0$ Wh（来自 `config.py WUHAN_ENV_CONFIG`）。

#### 2.3.7 L3奖励系统 / L3 Reward System

| 事件 / Event | 奖励 / Reward | 类型 / Type |
|:---|:---:|:---:|
| 完成单次配送 / Single delivery | +120 | 事件触发 / event |
| 全部配送完成 / All deliveries done | +300 | 事件触发 / event |
| 安全返航降落 / Safe return landing | +500 | 事件触发 / event |
| 建筑/地形碰撞 / Building/terrain collision | -250 | 事件触发 / event |
| 进入硬禁飞区 / Enter hard NFZ | -300 | 事件触发 / event |
| 电量耗尽 / Battery depleted | 失败 / Failure | 终止 / terminal |
| 净空不足 / Insufficient clearance | -4/次 | 连续 / continuous |
| 能耗惩罚 / Energy penalty | -0.8/Wh | 连续 / continuous |
| 软风险暴露 / Soft risk exposure | -1.5/次 | 连续 / continuous |
| 控制不平滑 / Control jerk | -0.03/次 | 连续 / continuous |
| 时间惩罚 / Time penalty | -0.08/s | 连续 / continuous |

**业务Score与训练Reward分离 / Business Score vs Training Reward Separation**:
- 训练使用密集Reward（含进度引导、能耗惩罚等）/ Dense rewards for training
- 验收使用业务Score（目标数严格优先字典序）/ Target-count-first lexicographic for evaluation

---

### 2.4 L4: 3D随机 + 全约束 / Level 4: 3D Random + Full Constraints

#### 2.4.1 多机协同CTDE框架 / Multi-Drone CTDE Framework

L4在L3基础上引入多无人机协同，采用**集中式训练分布式执行**（CTDE）架构：

L4 extends L3 with multi-drone cooperation using **Centralized Training with Decentralized Execution (CTDE)**:

```text
Training phase / 训练阶段:
  Critic sees global state (all drones' obs + communication)
  Actor only sees local obs (single drone)

Execution phase / 执行阶段:
  Each drone independently selects actions from its own Actor
```

#### 2.4.2 Box(4)连续动作空间 / Box(4) Continuous Action Space

$$
\mathcal{A} = \text{Box}(4) \subset \mathbb{R}^4
$$

| 维度 / Dim | 含义 / Meaning | 范围 / Range |
|:---:|:---|:---:|
| $a_0$ | XY平面加速度方向 / XY accel direction | $[-1, 1]$ |
| $a_1$ | 加速度幅值 / accel magnitude | $[0, 1]$ |
| $a_2$ | 垂直速度指令 / vertical velocity | $[-1, 1]$ |
| $a_3$ | 目标优先级评分 / target priority score | $[-1, 1]$ |

每架无人机的环境侧动作空间为16维 / Each drone's env-side action space is 16-dim:
- 3维加速度 + 1维偏航率 + 12维目标评分
- 分层规划器（ScoreFirstDeliveryPlanner）处理12维目标评分
- PPO执行4维连续飞行控制

#### 2.4.3 风场约束 / Wind Disturbance

$$
\mathbf{v}_{wind}(\mathbf{p}, t) = \mathbf{v}_{base} + \mathbf{v}_{terrain}(\mathbf{p}) + \mathbf{v}_{gust}(t)
$$

| 分量 / Component | 说明 / Description |
|:---|:---|
| $\mathbf{v}_{base}$ | 基本风向风速 / base wind direction and speed |
| $\mathbf{v}_{terrain}(\mathbf{p})$ | 地形扰动（街谷加速、湖面侧风）/ terrain disturbance |
| $\mathbf{v}_{gust}(t)$ | 随机阵风（Ornstein-Uhlenbeck过程）/ random gust (O-U process) |

风场对动力学的影响 / Wind effect on dynamics: $\dot{\mathbf{p}} = \mathbf{v}_{air} + \mathbf{v}_{wind}$

#### 2.4.4 通信距离限制 / Communication Range Limits

$$
\text{comm}_{ij} \iff \|\mathbf{p}_i - \mathbf{p}_j\|_2 \leq d_{comm}
$$

超出通信距离时 / When out of range:
- 无法共享任务信息（订单状态不可见）
- 无法协调冲突（可能选择相同订单）
- CTDE训练时Critic仍然可见全局状态

#### 2.4.5 动态禁飞区 / Dynamic No-Fly Zones

与L3的静态禁飞区不同，L4引入时间依赖的动态禁飞区：

Unlike L3's static NFZs, L4 introduces time-dependent dynamic no-fly zones:

$$
\mathcal{V}_{nfz}(t) = \mathcal{V}_{nfz,static} \cup \mathcal{V}_{nfz,construction}(t) \cup \mathcal{V}_{nfz,event}(t)
$$

#### 2.4.6 完整奖励系统（8分量）/ Complete Reward System (8 Components)

基于 `reward_function_design.md` 的完整奖励设计 / Complete reward design from `reward_function_design.md`:

$$
R_t = R_{goal} + P_{collision} + P_{proximity} + P_{formation} + P_{energy} + P_{time} + P_{smooth} + R_{delivery}
$$

| 分量 / Component | 公式 / Formula | 类型 / Type |
|:---|:---|:---:|
| 目标到达 $R_{goal}$ | $+200$ 当 $\|\mathbf{p}_t - \mathbf{p}_{goal}\|_2 \leq 5$m | 事件 / event |
| 碰撞惩罚 $P_{collision}$ | $-200$ 每次碰撞 | 事件 / event |
| 间距惩罚 $P_{proximity}$ | $-100 \sum \max(0, 1 - d_{ij}/d_{safe})$ | 连续 / continuous |
| 队形惩罚 $P_{formation}$ | $-\lambda_{form} \cdot \frac{1}{N}\sum\|\mathbf{p}_i - \mathbf{p}_{des,i}\|_2$ | 连续 / continuous |
| 能耗惩罚 $P_{energy}$ | $-\lambda_{energy} \cdot \frac{1}{N}\sum E_t^{(i)}$ | 连续 / continuous |
| 时间惩罚 $P_{time}$ | $-2$/步 | 连续 / continuous |
| 平滑惩罚 $P_{smooth}$ | $-\lambda_{smooth} \cdot \|\mathbf{a}_t - \mathbf{a}_{t-1}\|_2^2$ | 连续 / continuous |
| 配送奖励 $R_{delivery}$ | $+200$ 每完成一单 | 事件 / event |

---

### 2.5 层级对比与升级路径 / Level Comparison & Upgrade Path

#### 2.5.1 四层级统一对比表 / Four-Level Unified Comparison

| 维度 / Dimension | L1 | L2 | L3 | L4 |
|:---|:---:|:---:|:---:|:---:|
| 空间维度 / Spatial dim | 2D | 2D | 3D | 3D |
| 状态空间 / State space | Dict(3 keys) | Dict + 50-vec | 166-dim cont | 166-dim/drone |
| 动作空间 / Action space | Discrete(4) | Discrete(8) | Box(3) | Box(4) |
| 特征维度 / Feature dim | 1 (state_key) | 50 | 166 | 166 x N |
| 订单模式 / Order mode | 固定5个 | 随机4-8个 | 固定任务点 | 随机分配 |
| 时间窗口 / Time windows | 无 | 有 | 无 | 有 |
| 多单携带 / Multi-carry | 1单 | 最多2单 | 单次规划 | 多次重规划 |
| 禁飞区 / No-fly zones | 无 | 无 | 静态 | 动态 |
| 碰撞检测 / Collision | 网格碰撞 | 网格碰撞 | 连续扫掠体 | 连续扫掠体 |
| 能耗模型 / Energy | 无 | 无 | 简化Wh | 完整Wh+载重 |
| 风场 / Wind | 无 | 无 | 无 | 有 |
| 多机 / Multi-drone | 单机 | 单机 | 单机 | CTDE多机 |
| 通信 / Communication | 无 | 无 | 无 | 有 |
| 最大步数 / Max steps | 500 | 800 | 900s | 900s |
| 推荐算法 / Algorithms | Q-learning, SARSA, MC | DQN, A2C | A2C, PPO | PPO+CTDE |

#### 2.5.2 12个全局任务点坐标 / 12 Global Task Points

| 全局ID | 分区 / Zone | 名称 / Name | 类型 / Type | ENU坐标(m) / ENU Coord |
|:---:|:---:|:---|:---:|:---|
| 1 | A | 配送中心 / Depot | 基地 / Base | $[408, 324, 0.8]$ |
| 2 | A | 商业裙楼屋顶 / Commercial roof | 配送 / Delivery | $[1149, 1073, 15.8]$ |
| 3 | A | 商业综合体平台 / Complex platform | 配送 / Delivery | $[2013, 491, 9.2]$ |
| 4 | B | 校园后勤点 / Campus logistics | 基地 / Base | $[1790, 589, 14.8]$ |
| 5 | B | 公共建筑屋顶 / Public roof | 配送 / Delivery | $[945, 1011, 30.8]$ |
| 6 | B | 湖岸维护设施 / Lakeside facility | 配送 / Delivery | $[1160, 990, 6.8]$ |
| 7 | C | 桥头服务设施 / Bridgehead facility | 基地 / Base | $[2139, 1189, 11.2]$ |
| 8 | C | 老城配送平台 / Old city platform | 配送 / Delivery | $[2363, 438, 11.2]$ |
| 9 | C | 滨水公共设施 / Waterfront facility | 配送 / Delivery | $[1929, 463, 12.4]$ |
| 10 | D | 商业裙楼屋顶 / Commercial roof | 基地 / Base | $[714, 1047, 6.8]$ |
| 11 | D | 办公塔楼平台 / Office tower | 配送 / Delivery | $[1249, 642, 138.7]$ |
| 12 | D | 社区公共建筑 / Community building | 配送 / Delivery | $[2028, 689, 47.7]$ |

*来源 / Source: `outputs/环境示意图_2D审核/scene_2d_manifest.json`*

#### 2.5.3 课程学习C1-C7映射 / Curriculum Learning C1-C7 Mapping

| 阶段 / Stage | 层级 / Level | 分区 / Zone | 训练重点 / Focus | 转换条件 / Transition |
|:---:|:---:|:---:|:---|:---|
| C1 | L1 | B (43栋) | 基础导航+单配送 / basic nav + 1 delivery | 完成率 > 80% |
| C2 | L1 | B | 5单全部配送 / all 5 deliveries | 完成率 > 70% |
| C3 | L2 | B | 随机订单+时间窗口 / random + time | 按时率 > 60% |
| C4 | L3 | B | 3D物理约束 / 3D physical constraints | 安全返航 > 50% |
| C5 | L3 | A (837栋) | 高密度建筑 / high-density buildings | 安全返航 > 40% |
| C6 | L3 | C/D | 复杂地形 / complex terrain | 安全返航 > 35% |
| C7 | L4 | A+B | 多机协同 / multi-drone cooperation | 协同效率 > 阈值 |

---

### 2.6 环境示意图索引 / Environment Visualization Index

#### 总览图 / Overview Images

| 图片 / Image | 路径 / Path | 用途 / Purpose |
|:---|:---|:---|
| 四区总览 / Zone overview | `outputs/环境示意图/wuhan_environment_overview.png` | 项目全景介绍 |
| 3D环境 / 3D environment | `outputs/环境示意图/wuhan_environment_3d.png` | 三维场景展示 |
| 建筑高度热力图 / Height heatmap | `outputs/环境示意图/wuhan_building_height_heatmap.png` | 高度分布 |
| 障碍物图例 / Obstacle legend | `outputs/环境示意图/wuhan_obstacle_legend.png` | 风险类型图例 |
| 四区分区图 / Zone top view | `outputs/环境示意图/wuhan_zone_top_view.png` | 分区布局 |

#### 2D审核图 / 2D Review Maps

| 图片 / Image | 路径 / Path | 分区 / Zone |
|:---|:---|:---:|
| Zone A 2D | `outputs/环境示意图_2D审核/zone_A_guanggu_core_2d.png` | A |
| Zone B 2D | `outputs/环境示意图_2D审核/zone_B_lakeside_campus_2d.png` | B |
| Zone C 2D | `outputs/环境示意图_2D审核/zone_C_river_bridgehead_2d.png` | C |
| Zone D 2D | `outputs/环境示意图_2D审核/zone_D_canal_highrise_2d.png` | D |

#### 3D预览图 / 3D Preview Images

| 图片 / Image | 路径 / Path | 分区 / Zone |
|:---|:---|:---:|
| Zone A 3D | `outputs/环境示意图_3D预览/zone_A_scene_lod0_preview.png` | A |
| Zone B 3D | `outputs/环境示意图_3D预览/zone_B_scene_lod0_preview.png` | B |
| Zone C 3D | `outputs/环境示意图_3D预览/zone_C_scene_lod0_preview.png` | C |
| Zone D 3D | `outputs/环境示意图_3D预览/zone_D_scene_lod0_preview.png` | D |

#### 数据索引文件 / Data Index Files

| 文件 / File | 路径 / Path | 用途 / Purpose |
|:---|:---|:---|
| 3D场景索引 / 3D scene index | `outputs/环境示意图_3D预览/scene_3d_index.json` | 四区3D场景入口 |
| 2D审核清单 / 2D review manifest | `outputs/环境示意图_2D审核/scene_2d_manifest.json` | 四区2D审核数据 |

---

> **下一节 / Next**: [3. 算法体系设计 / Algorithm System Design](03_algorithms.md)
