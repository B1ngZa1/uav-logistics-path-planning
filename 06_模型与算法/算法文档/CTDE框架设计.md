# CTDE框架设计文档（完整版）

> 项目：基于深度强化学习的多无人机协同物流配送路径规划
> 模块：CTDE（Centralized Training with Decentralized Execution）框架设计
> 版本：v2.0
> 创建日期：2026-07-01
> 更新日期：2026-07-01

---

## 一、框架概述

### 1.1 什么是CTDE

CTDE（Centralized Training with Decentralized Execution，集中式训练分散式执行）是多智能体强化学习（MARL）中的主流训练范式。其核心思想是：

- **训练阶段**：利用全局信息（所有智能体的状态、动作、奖励）训练策略和价值网络
- **执行阶段**：每个智能体仅基于局部观测独立决策，无需全局信息

这种设计完美契合实际无人机配送场景：训练可以在拥有完整信息的仿真环境中进行，而部署时每架无人机只能获取自身的传感器数据。

### 1.2 CTDE与本项目的契合

| 方面 | 训练阶段（集中式） | 执行阶段（分散式） |
|------|-------------------|-------------------|
| 状态信息 | 全局状态（所有无人机位置、速度、目标等） | 局部观测（自身位置、速度、最近障碍物等） |
| 决策主体 | 集中式Critic统一评估 | 独立Actor各自决策 |
| 奖励信号 | 全局奖励（考虑所有无人机） | 无（执行时不计算奖励） |
| 通信需求 | 无限制（仿真环境） | 低带宽、事件触发 |

### 1.3 框架总体架构

```
                    +-----------------------------------------------------+
                    |              集中式训练阶段                          |
                    |                                                     |
                    |   全局状态 s_global --> 集中式Critic --> V(s)       |
                    |         |                                         |
                    |         v                                         |
                    |   +------------------------------------------+   |
                    |   |          共享Actor网络                     |   |
                    |   |   s_local_i --> a_i (i=1,...,N)          |   |
                    |   +------------------------------------------+   |
                    |         |                                         |
                    |         v                                         |
                    |   环境交互 --> 全局奖励 R --> 更新网络             |
                    +-----------------------------------------------------+

                    +-----------------------------------------------------+
                    |              分散式执行阶段                          |
                    |                                                     |
                    |   UAV_1: s_local_1 --> Actor_1 --> a_1             |
                    |   UAV_2: s_local_2 --> Actor_2 --> a_2             |
                    |   UAV_3: s_local_3 --> Actor_3 --> a_3             |
                    |   ...                                               |
                    |   UAV_N: s_local_N --> Actor_N --> a_N             |
                    |                                                     |
                    |   （各无人机独立决策，可选低带宽通信）               |
                    +-----------------------------------------------------+
```

---

## 二、集中式训练设计

### 2.1 共享Actor网络

在CTDE框架下，所有无人机共享同一个Actor网络参数 $\theta$。这样做的优势：

1. **参数效率**：N架无人机只需维护一套Actor参数，减少内存占用
2. **经验共享**：所有无人机的经验都可以用于更新同一个策略
3. **泛化能力**：共享参数迫使策略学习通用的飞行控制技能

**共享策略的数学表达**：

$$
\pi_\theta(a_i | o_i), \quad i = 1, 2, \ldots, N
$$

其中 $o_i$ 为无人机 $i$ 的局部观测。所有无人机使用相同的 $\theta$，但输入不同的局部观测，输出不同的动作。

**参数共享方式**：

```python
class SharedActor(nn.Module):
    """共享Actor网络（所有无人机使用同一套参数）"""
    def __init__(self, obs_dim, action_dim, hidden_dims=[256, 128, 64]):
        super().__init__()
        self.feature_net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU()
        )
        self.mean_head = nn.Linear(64, action_dim)
        self.log_std_head = nn.Linear(64, action_dim)

    def forward(self, obs):
        features = self.feature_net(obs)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        return mean, torch.exp(log_std)
```

### 2.2 全局状态聚合方法

#### 2.2.1 全局状态定义

集中式Critic在训练时接收全局状态信息，能够评估联合动作的价值。这是CTDE的核心优势：训练时利用全局信息，执行时不需要。

**全局状态向量**：

$$
s_{global} = \begin{bmatrix} o_1 \\ o_2 \\ \vdots \\ o_N \\ \mathbf{g}_{shared} \end{bmatrix} \in \mathbb{R}^{N \cdot D_{obs} + D_{shared}}
$$

其中：
- $o_i \in \mathbb{R}^{D_{obs}}$：无人机 $i$ 的局部观测（11维）
- $\mathbf{g}_{shared}$：共享的全局信息（目标位置、障碍物信息等）
- $N$：无人机数量
- $D_{obs} = 11$：局部观测维度

#### 2.2.2 全局状态聚合维度

| 聚合层次 | 包含信息 | 维度 | 说明 |
|---------|---------|------|------|
| 所有无人机状态 | 各无人机位置、速度、电量 | $N \times 6$ | 拼接所有无人机的6维基本状态 |
| 目标距离信息 | 各无人机到各自目标的距离 | $N \times 1$ | 拼接各无人机的目标距离 |
| 障碍物距离信息 | 各无人机到最近障碍物的距离 | $N \times 1$ | 拼接各无人机的障碍物距离 |
| 邻居距离信息 | 各无人机到最近其他无人机的距离 | $N \times 1$ | 拼接各无人机的邻居距离 |
| 电量信息 | 各无人机剩余电量 | $N \times 1$ | 拼接各无人机的电量 |
| 配送进度信息 | 各无人机的配送进度 | $N \times 1$ | 拼接各无人机的配送状态 |
| 共享全局信息 | 目标坐标、障碍物位置等 | $D_{shared}$ | 固定长度的全局环境信息 |

**总全局状态维度**：

$$
D_{global} = N \times D_{obs} + D_{shared} = N \times 11 + D_{shared}
$$

对于 $N=3$ 架无人机，$D_{shared}=10$，则 $D_{global} = 3 \times 11 + 10 = 43$。

#### 2.2.3 全局状态编码方法

**方法一：直接拼接（推荐）**

将所有无人机的局部观测和共享信息直接拼接为一个向量：

$$
s_{global} = [o_1; o_2; \ldots; o_N; g_{shared}]
$$

优点：简单直接，信息无损。
缺点：维度随无人机数量线性增长。

**方法二：注意力机制聚合**

使用注意力机制将所有无人机的观测聚合为固定长度的全局表示：

$$
\mathbf{h}_{global} = \text{Attention}(Q = \mathbf{q}_{global}, K = [o_1; \ldots; o_N], V = [o_1; \ldots; o_N])
$$

其中 $\mathbf{q}_{global}$ 为可学习的全局查询向量。

优点：维度固定，可处理变数量无人机。
缺点：实现复杂，可能丢失部分信息。

**方法三：均值池化聚合**

对所有无人机的状态取均值作为全局状态的一部分：

$$
\bar{o} = \frac{1}{N} \sum_{i=1}^{N} o_i
$$

$$
s_{global} = [\bar{o}; \text{std}(o_1, \ldots, o_N); \max(o_1, \ldots, o_N); g_{shared}]
$$

优点：维度固定，计算简单。
缺点：丢失个体差异信息。

**推荐方案**：方法一（直接拼接），因为本项目无人机数量固定（3-6架），且需要保留每个无人机的个体信息以便Critic准确评估联合动作价值。

#### 2.2.4 全局状态编码实现

```python
def encode_global_state(all_observations, shared_info):
    """
    将所有无人机的局部观测和共享信息编码为全局状态向量

    参数:
        all_observations: np.ndarray, shape (N, D_obs)  各无人机局部观测
        shared_info: np.ndarray, shape (D_shared,)      共享全局信息

    返回:
        global_state: np.ndarray, shape (N*D_obs + D_shared,)
    """
    N = all_observations.shape[0]
    D_obs = all_observations.shape[1]

    # 拼接所有无人机观测
    flat_obs = all_observations.flatten()  # shape: (N*D_obs,)

    # 拼接共享信息
    global_state = np.concatenate([flat_obs, shared_info])

    return global_state
```

### 2.3 集中式Critic网络

#### 2.3.1 网络结构

```
全局状态输入: s_global in R^(N*11 + D_shared)
    |
    v
全连接层1: Linear(D_global, 256) + Tanh
    |
    v
全连接层2: Linear(256, 128) + Tanh
    |
    v
全连接层3: Linear(128, 64) + Tanh
    |
    v
输出层: Linear(64, 1) --> V(s_global)
```

**数学表达**：

$$
V_\phi(s_{global}) = W_v \cdot \text{Tanh}(W_3 \cdot \text{Tanh}(W_2 \cdot \text{Tanh}(W_1 s_{global} + b_1) + b_2) + b_3) + b_v
$$

#### 2.3.2 与独立Critic的对比

| 特性 | 集中式Critic | 独立Critic |
|------|-------------|-----------|
| 输入 | 全局状态 | 局部观测 |
| 评估范围 | 联合动作价值 | 单智能体动作价值 |
| 信息量 | 完整 | 局部 |
| 训练阶段 | 使用 | 使用 |
| 执行阶段 | 不使用 | 不使用 |
| 信用分配 | 能够解决 | 难以解决 |

### 2.4 全局奖励函数设计

#### 2.4.1 全局奖励结构

训练时使用全局奖励信号，考虑所有无人机的整体表现：

$$
R_{global} = \sum_{i=1}^{N} R_i + R_{cooperation}
$$

其中：
- $R_i$：无人机 $i$ 的个体奖励（详见奖励函数设计文档）
- $R_{cooperation}$：协同奖励项

#### 2.4.2 协同约束奖励分量

$$
R_{cooperation} = R_{task\_global} + R_{collision\_global} + R_{formation} + R_{load\_balance}
$$

| 分量 | 公式 | 说明 |
|------|------|------|
| 全局任务完成 | $R_{task} = \sum_{k=1}^{K} 50 \cdot \mathbb{1}[\text{delivery}_k \text{ done}]$ | 完成配送任务的总奖励 |
| 全局避碰协同 | $R_{collision} = -200 \cdot \mathbb{1}[\text{any collision}]$ | 任意碰撞的全局惩罚 |
| 队形保持 | $R_{form} = -1.0 \cdot \frac{1}{N}\sum_i \|p_i - p_{des}^i\|$ | 编队偏差惩罚 |
| 负载均衡 | $R_{balance} = -\lambda_{bal} \cdot \text{std}(\text{delivery\_count}_1, \ldots, \text{delivery\_count}_N)$ | 各无人机配送量均衡 |

#### 2.4.3 全局奖励的优势

- **避免信用分配问题**：每架无人机都能从全局成功中获益
- **鼓励协同行为**：无人机学会互相配合而非各自为政
- **简化算法设计**：无需设计复杂的局部奖励分解机制

#### 2.4.4 全局奖励计算实现

```python
def compute_global_reward(positions, prev_positions, actions, prev_actions,
                          velocities, goal_positions, formation_offsets,
                          obstacles, cargo_loaded, cargo_delivered,
                          battery_ratios, step_count, max_steps, boundary):
    """
    计算全局奖励（包含协同约束）

    参数:
        positions: (N, 3) 各无人机位置
        其他参数同RewardCalculator

    返回:
        global_reward: 全局标量奖励
        reward_components: 各分量明细
    """
    N = positions.shape[0]
    rewards = {}

    # 1. 个体奖励之和
    individual_rewards = 0.0
    for i in range(N):
        # 各无人机的个体奖励计算（复用RewardCalculator）
        r_i = compute_individual_reward(i, positions, prev_positions, ...)
        individual_rewards += r_i

    # 2. 全局协同奖励
    # 2.1 全局碰撞惩罚（任意碰撞即惩罚）
    collision_detected = check_any_collision(positions, obstacles, boundary)
    R_collision = -200.0 if collision_detected else 0.0

    # 2.2 队形保持惩罚
    center = positions.mean(axis=0)
    desired_pos = center + formation_offsets[:N]
    formation_errors = np.linalg.norm(positions - desired_pos, axis=1)
    R_formation = -1.0 * formation_errors.mean()

    # 2.3 负载均衡奖励
    delivery_counts = np.array([count_deliveries(i) for i in range(N)])
    R_balance = -0.5 * np.std(delivery_counts) if N > 1 else 0.0

    # 2.4 全局任务完成奖励
    R_task = 50.0 * cargo_delivered.sum()

    rewards['individual'] = individual_rewards
    rewards['collision'] = R_collision
    rewards['formation'] = R_formation
    rewards['balance'] = R_balance
    rewards['task'] = R_task

    global_reward = sum(rewards.values())
    return global_reward, rewards
```

### 2.5 集中式训练算法流程

```
算法：CTDE-PPO集中式训练
输入：环境 env，共享Actor参数 theta，集中式Critic参数 phi
输出：训练后的共享策略 theta*

1. 初始化：
    - 创建共享Actor网络，参数 theta
    - 创建集中式Critic网络，参数 phi
    - 创建经验缓冲区 buffer
    - 创建Adam优化器

2. for epoch = 1, 2, ..., N_epochs do

3.     // ========== 阶段1：分散式经验收集 ==========
4.     s_global = env.reset()
5.     for t = 0, 1, ..., T-1 do
6.         // 各无人机独立决策（使用共享Actor）
7.         for i = 1, 2, ..., N do
8.             o_i = get_local_obs(s_global, i)
9.             mu_i, sigma_i = Actor_theta(o_i)
10.            a_i ~ N(mu_i, sigma_i^2)
11.            a_i_mapped = tanh(a_i) * a_max
12.        end for
13.
14.        // 环境执行联合动作
15.        s_global', R_global, done = env.step([a_1, ..., a_N])
16.
17.        // 集中式Critic评估全局状态价值
18.        V_t = Critic_phi(s_global)
19.
20.        // 存储经验（每个无人机都存储相同的经验）
21.        for i = 1, ..., N do
22.            buffer.store(o_i, a_i, R_global, V_t, log_pi_i, done)
23.        end for
24.
25.        s_global = s_global'
26.        if done then s_global = env.reset()
27.    end for
28.
29.    // ========== 阶段2：GAE计算 ==========
30.    V_T = Critic_phi(s_global_T)
31.    for t = T-1, T-2, ..., 0 do
32.        delta_t = R_t + gamma * V_{t+1} * (1-done_t) - V_t
33.        A_t = delta_t + gamma * lambda * (1-done_t) * A_{t+1}
34.        R_t = A_t + V_t
35.    end for
36.    归一化优势：A = (A - mean(A)) / (std(A) + 1e-8)
37.
38.    // ========== 阶段3：策略更新（K轮） ==========
39.    for k = 1, 2, ..., K do
40.        随机采样 mini-batches (batch_size=256)
41.        for each batch B do
42.            // --- Actor更新（使用局部观测） ---
43.            new_log_pi_i = Actor_theta(o_i).log_prob(a_i)
44.            ratio = exp(new_log_pi - old_log_pi)
45.            surr1 = ratio * A
46.            surr2 = clamp(ratio, 1-eps, 1+eps) * A
47.            actor_loss = -min(surr1, surr2).mean()
48.            actor_loss -= c2 * entropy.mean()
49.
50.            // --- Critic更新（使用全局状态） ---
51.            V_pred = Critic_phi(s_global)
52.            critic_loss = c1 * MSE(V_pred, R)
53.
54.            // --- 梯度更新 ---
55.            total_loss = actor_loss + critic_loss
56.            optimizer.zero_grad()
57.            total_loss.backward()
58.            clip_grad_norm_(parameters, max_norm=0.5)
59.            optimizer.step()
60.        end for
61.    end for
62.
63.    // ========== 阶段4：学习率更新 ==========
64.    scheduler.step()
65.
66.    // ========== 阶段5：日志记录 ==========
67.    记录：全局回报、各无人机回报、碰撞率、任务完成率
68.
69. end for
70.
71. 返回训练后的共享策略 theta*
```

---

## 三、分散式执行设计

### 3.1 独立Actor网络

在执行阶段，每架无人机加载相同的Actor网络参数，但基于各自的局部观测独立做出决策。

### 3.2 局部观测定义

#### 3.2.1 局部观测向量

$$
o_i = \begin{bmatrix} x_i \\ y_i \\ z_i \\ v_{x,i} \\ v_{y,i} \\ v_{z,i} \\ d_{goal,i} \\ d_{obs,i} \\ d_{uav,i} \\ E_{remain,i} \\ p_{delivery,i} \end{bmatrix} \in \mathbb{R}^{11}
$$

#### 3.2.2 各维度说明

| 维度 | 符号 | 含义 | 取值范围 | 传感器来源 |
|------|------|------|---------|-----------|
| 1-3 | (x, y, z) | 三维位置 | [0,100] x [0,100] x [10,120] | GPS/IMU |
| 4-6 | (vx, vy, vz) | 三维速度 | [-15, 15] m/s | IMU |
| 7 | d_goal | 到目标距离 | [0, 180] m | GPS计算 |
| 8 | d_obs | 到最近障碍物距离 | [0, 100] m | 激光雷达/视觉 |
| 9 | d_uav | 到最近其他无人机距离 | [0, 100] m | 通信/雷达 |
| 10 | E_remain | 剩余电量比例 | [0, 1] | 电池管理系统 |
| 11 | p_delivery | 配送进度 | {0, 1, 2, 3} | 任务系统 |

#### 3.2.3 局部观测与全局状态的关系

局部观测是全局状态的子集，满足以下约束：

$$
o_i \subset s_{global}, \quad \forall i \in \{1, \ldots, N\}
$$

**信息映射关系**：

```
全局状态 s_global:
[ UAV_1状态 | UAV_2状态 | ... | UAV_N状态 | 共享信息 ]
     |           |                  |
     v           v                  v
局部观测 o_1  局部观测 o_2  ...  局部观测 o_N
(自身状态)     (自身状态)        (自身状态)
```

**局部观测的信息层次**：

| 信息层次 | 内容 | 是否包含 | 说明 |
|---------|------|---------|------|
| 自身状态 | 位置、速度、电量 | 是 | 核心决策信息 |
| 目标信息 | 到目标距离、配送进度 | 是 | 任务导向信息 |
| 障碍物信息 | 到最近障碍物距离 | 是 | 安全避障信息 |
| 其他无人机 | 到最近无人机距离 | 是（简化） | 协同避碰信息 |
| 其他无人机完整状态 | 所有无人机位置速度 | 否 | 执行阶段不可用 |
| 全局目标位置 | 所有目标坐标 | 否 | 执行阶段不可用 |

#### 3.2.4 局部观测范围设计

**传感器感知范围**：

| 传感器 | 感知范围 | 更新频率 | 数据类型 |
|--------|---------|---------|---------|
| GPS | 全球定位 | 10 Hz | 三维位置 |
| IMU | 自身状态 | 100 Hz | 三维速度、姿态 |
| 激光雷达 | 50-100 m | 10 Hz | 障碍物距离 |
| 通信模块 | 100-500 m | 1-10 Hz | 邻居位置 |
| 电池管理系统 | 自身 | 1 Hz | 电量百分比 |

**局部观测的物理含义**：

- **自身状态（维度1-6）**：无人机通过自身传感器直接获取
- **目标距离（维度7）**：通过GPS位置与已知目标坐标计算得出
- **障碍物距离（维度8）**：通过激光雷达或视觉传感器检测最近障碍物
- **邻居距离（维度9）**：通过通信模块或雷达检测最近的其他无人机
- **电量（维度10）**：通过电池管理系统读取
- **配送进度（维度11）**：通过任务管理系统获取当前任务状态

### 3.3 分散式执行算法流程

```
算法：分散式执行
输入：训练好的共享Actor网络参数 theta
输出：各无人机的动作序列

1. 初始化：各无人机获取初始局部观测 o_i
2. while 任务未完成 do
3.     for each 无人机 i = 1, 2, ..., N do
4.         // 独立决策（无需通信）
5.         mu_i, sigma_i = Actor_theta(o_i)
6.         a_i ~ N(mu_i, sigma_i^2)
7.         a_i = tanh(a_i) * a_max  // 动作映射
8.     end for
9.
10.    // 并行执行
11.    for each 无人机 i do
12.        o_i' = env.step_i(a_i)  // 更新局部观测
13.    end for
14.
15.    // 通信（可选，事件触发）
16.    if 触发通信条件 then
17.        无人机之间交换信息
18.    end if
19. end while
```

### 3.4 无需实时通信的设计优势

- **降低系统复杂度**：无需维护实时通信链路
- **提高鲁棒性**：单机通信故障不影响其他无人机决策
- **减少延迟**：无需等待通信数据即可决策
- **降低成本**：减少通信硬件和带宽需求

---

## 四、通信机制设计（可选）

### 4.1 通信机制概述

虽然分散式执行阶段无人机可以完全独立决策，但引入可选的低带宽通信机制可以进一步提升系统性能。本节设计三种通信模式：

1. **低带宽通信**：共享目标位置、速度方向
2. **事件触发通信**：碰撞风险预警时触发
3. **分层通信**：区域内无人机局部通信

### 4.2 低带宽通信协议

#### 4.2.1 通信消息格式

```python
@dataclass
class CommMessage:
    """无人机通信消息"""
    uav_id: int                        # 无人机ID (1字节)
    position: np.ndarray               # 位置 (3 x 4字节 = 12字节)
    velocity: np.ndarray               # 速度方向 (3 x 4字节 = 12字节)
    target_position: np.ndarray        # 目标位置 (3 x 4字节 = 12字节)
    battery_level: float               # 电量 (4字节)
    task_status: int                   # 任务状态 (1字节)
    priority: int                      # 优先级 (1字节)
    timestamp: float                   # 时间戳 (8字节)

    def to_bytes(self) -> bytes:
        """序列化为字节（约51字节）"""
        # ID(1B) + 位置(12B) + 速度(12B) + 目标(12B) + 电量(4B)
        # + 状态(1B) + 优先级(1B) + 时间戳(8B) = 51字节
        pass

    @staticmethod
    def from_bytes(data: bytes) -> 'CommMessage':
        """从字节反序列化"""
        pass
```

#### 4.2.2 带宽需求分析

| 参数 | 值 | 说明 |
|------|-----|------|
| 单条消息大小 | ~51 字节 | 包含位置、速度、目标等 |
| 通信频率 | 1 Hz（每秒1次） | 低频通信 |
| 无人机数量 | 3 架 | 默认配置 |
| 总带宽需求 | 51 x 3 = 153 字节/秒 = 1224 bps | 极低带宽 |
| 最大通信距离 | 500 m | 视距通信 |

#### 4.2.3 低带宽通信实现

```python
class LowBandwidthComm:
    """低带宽通信模块"""

    def __init__(self, uav_id, comm_range=500.0, comm_freq=1.0):
        self.uav_id = uav_id
        self.comm_range = comm_range  # 通信范围（米）
        self.comm_freq = comm_freq    # 通信频率（Hz）
        self.last_comm_time = 0.0
        self.neighbors = {}           # 邻居信息缓存

    def should_communicate(self, current_time):
        """判断是否应该发送消息"""
        return (current_time - self.last_comm_time) >= (1.0 / self.comm_freq)

    def create_message(self, position, velocity, target, battery, task_status):
        """创建通信消息"""
        return CommMessage(
            uav_id=self.uav_id,
            position=position,
            velocity=velocity,
            target_position=target,
            battery_level=battery,
            task_status=task_status,
            priority=0,
            timestamp=time.time()
        )

    def send_message(self, message, all_uavs):
        """广播消息给通信范围内的无人机"""
        for uav in all_uavs:
            if uav.id != self.uav_id:
                dist = np.linalg.norm(message.position - uav.position)
                if dist <= self.comm_range:
                    uav.receive_message(message)

    def receive_message(self, message):
        """接收邻居消息"""
        self.neighbors[message.uav_id] = {
            'position': message.position,
            'velocity': message.velocity,
            'target': message.target_position,
            'battery': message.battery_level,
            'task_status': message.task_status,
            'timestamp': message.timestamp
        }
```

### 4.3 事件触发通信

#### 4.3.1 触发条件定义

并非每个时间步都需要通信。设计事件触发机制，仅在必要时进行信息交换：

| 触发事件 | 条件 | 优先级 | 通信内容 | 响应时间要求 |
|---------|------|--------|---------|-------------|
| 碰撞风险 | $d_{uav} < d_{safe\_warn}$ (3m) | 紧急 | 位置、速度、意图 | < 100ms |
| 近距离预警 | $d_{uav} < d_{safe}$ (10m) | 高 | 位置、速度 | < 500ms |
| 任务完成 | 到达目标点 | 中 | 任务状态 | < 1s |
| 电量预警 | $E_{remain} < 0.2$ | 高 | 电量、位置 | < 1s |
| 新订单 | 系统分配新任务 | 中 | 任务信息 | < 2s |
| 定期同步 | 每60秒 | 低 | 完整状态 | 无要求 |

#### 4.3.2 事件触发通信算法

```python
class EventTriggeredComm:
    """事件触发通信模块"""

    def __init__(self, uav_id, d_safe_warn=3.0, d_safe=10.0,
                 battery_warn=0.2, sync_interval=60.0):
        self.uav_id = uav_id
        self.d_safe_warn = d_safe_warn
        self.d_safe = d_safe
        self.battery_warn = battery_warn
        self.sync_interval = sync_interval
        self.last_sync_time = 0.0
        self.event_queue = []

    def check_events(self, local_obs, neighbors, current_time):
        """
        检查是否触发通信事件

        参数:
            local_obs: 局部观测 (11维)
            neighbors: 邻居信息字典
            current_time: 当前时间

        返回:
            events: 触发的事件列表
        """
        events = []

        # 1. 碰撞风险检测
        d_uav = local_obs[8]  # 到最近无人机距离
        if d_uav < self.d_safe_warn:
            events.append({
                'type': 'COLLISION_RISK',
                'priority': 'CRITICAL',
                'data': {'distance': d_uav}
            })
        elif d_uav < self.d_safe:
            events.append({
                'type': 'PROXIMITY_WARNING',
                'priority': 'HIGH',
                'data': {'distance': d_uav}
            })

        # 2. 电量预警
        battery = local_obs[9]
        if battery < self.battery_warn:
            events.append({
                'type': 'BATTERY_WARNING',
                'priority': 'HIGH',
                'data': {'battery': battery}
            })

        # 3. 定期同步
        if (current_time - self.last_sync_time) >= self.sync_interval:
            events.append({
                'type': 'PERIODIC_SYNC',
                'priority': 'LOW',
                'data': {}
            })
            self.last_sync_time = current_time

        return events

    def create_event_message(self, event, local_obs, position, velocity):
        """根据事件类型创建通信消息"""
        if event['priority'] == 'CRITICAL':
            # 紧急消息：只包含位置和速度
            return CommMessage(
                uav_id=self.uav_id,
                position=position,
                velocity=velocity,
                target_position=np.zeros(3),
                battery_level=local_obs[9],
                task_status=int(local_obs[10]),
                priority=2,
                timestamp=time.time()
            )
        else:
            # 普通消息：包含完整信息
            return CommMessage(
                uav_id=self.uav_id,
                position=position,
                velocity=velocity,
                target_position=np.zeros(3),  # 可从任务系统获取
                battery_level=local_obs[9],
                task_status=int(local_obs[10]),
                priority=1 if event['priority'] == 'HIGH' else 0,
                timestamp=time.time()
            )
```

#### 4.3.3 事件触发通信流程

```
算法：事件触发通信
输入：当前局部观测 o_t，邻居信息 neighbors，当前时间 t
输出：是否触发通信，通信内容

1. 检查碰撞风险：
   if d_uav < 3m then
       触发紧急通信，发送位置和速度
       优先级 = CRITICAL
       响应时间 < 100ms

2. 检查近距离预警：
   if d_uav < 10m then
       触发高优先级通信
       优先级 = HIGH

3. 检查电量：
   if E_remain < 0.2 then
       触发高优先级通信，发送电量和位置
       优先级 = HIGH

4. 检查任务事件：
   if 任务完成 or 新订单 then
       触发中优先级通信
       优先级 = MEDIUM

5. 定期同步：
   if time_since_last_comm > 60s then
       触发低优先级通信，发送完整状态
       优先级 = LOW

6. 返回通信决策和内容
```

### 4.4 分层通信架构

#### 4.4.1 分层通信设计

在大规模无人机集群中，全网通信会导致带宽爆炸。设计分层通信架构：

```
                    +-------------------+
                    |    区域协调器      |
                    |  (Area Coordinator)|
                    +-------------------+
                           |
            +--------------+--------------+
            |              |              |
      +---------+    +---------+    +---------+
      | 区域A   |    | 区域B   |    | 区域C   |
      | UAV 1-3 |    | UAV 4-6 |    | UAV 7-9 |
      +---------+    +---------+    +---------+
```

#### 4.4.2 分层通信协议

**第一层：区域内通信（Local Communication）**

- 范围：同一区域内的无人机（3-5架）
- 频率：10 Hz
- 内容：位置、速度、意图
- 方式：广播

**第二层：区域间通信（Inter-area Communication）**

- 范围：相邻区域之间
- 频率：1 Hz
- 内容：区域摘要信息（无人机数量、任务状态）
- 方式：通过区域协调器转发

**第三层：全局协调（Global Coordination）**

- 范围：所有区域
- 频率：0.1 Hz
- 内容：全局任务分配、紧急事件
- 方式：中心化服务器

#### 4.4.3 分层通信实现

```python
class HierarchicalComm:
    """分层通信模块"""

    def __init__(self, uav_id, area_id, area_size=3):
        self.uav_id = uav_id
        self.area_id = area_id
        self.area_size = area_size
        self.local_neighbors = {}      # 区域内邻居
        self.area_coordinator = None   # 区域协调器
        self.global_server = None      # 全局服务器

    def local_broadcast(self, message, area_uavs):
        """区域内广播"""
        for uav in area_uavs:
            if uav.id != self.uav_id:
                uav.receive_local_message(message)

    def report_to_coordinator(self, summary):
        """向区域协调器报告"""
        if self.area_coordinator:
            self.area_coordinator.receive_report(self.uav_id, summary)

    def receive_area_summary(self, summary):
        """接收区域摘要"""
        # 更新对其他区域的了解
        pass

    def get_local_enhanced_obs(self, local_obs):
        """
        使用通信信息增强局部观测

        参数:
            local_obs: 原始局部观测 (11维)

        返回:
            enhanced_obs: 增强后的观测 (11 + 通信信息维度)
        """
        # 从邻居获取信息
        nearest_neighbor_info = self._get_nearest_neighbor_info()

        # 增强观测
        enhanced_obs = np.concatenate([
            local_obs,
            nearest_neighbor_info['relative_position'],  # 3维
            nearest_neighbor_info['relative_velocity'],  # 3维
            [nearest_neighbor_info['distance']],          # 1维
        ])

        return enhanced_obs

    def _get_nearest_neighbor_info(self):
        """获取最近邻居的信息"""
        if not self.local_neighbors:
            return {
                'relative_position': np.zeros(3),
                'relative_velocity': np.zeros(3),
                'distance': 100.0  # 默认大距离
            }

        # 找到最近邻居
        min_dist = float('inf')
        nearest = None
        for uid, info in self.local_neighbors.items():
            dist = np.linalg.norm(info['position'] - self.position)
            if dist < min_dist:
                min_dist = dist
                nearest = info

        if nearest:
            return {
                'relative_position': nearest['position'] - self.position,
                'relative_velocity': nearest['velocity'] - self.velocity,
                'distance': min_dist
            }
        else:
            return {
                'relative_position': np.zeros(3),
                'relative_velocity': np.zeros(3),
                'distance': 100.0
            }
```

### 4.5 通信对训练的影响

#### 4.5.1 通信增强的状态表示

在训练阶段，通信信息可以作为Critic的额外输入：

$$
s_{global}^{comm} = s_{global} \oplus \mathbf{m}_{comm}
$$

其中 $\mathbf{m}_{comm}$ 为通信获得的邻居信息向量。

**通信增强的全局状态**：

$$
s_{global}^{comm} = \begin{bmatrix} o_1 \\ o_2 \\ \vdots \\ o_N \\ \mathbf{g}_{shared} \\ \mathbf{m}_{1 \to all} \\ \mathbf{m}_{2 \to all} \\ \vdots \\ \mathbf{m}_{N \to all} \end{bmatrix}
$$

#### 4.5.2 通信感知的训练策略

```python
class CommAwareCritic(nn.Module):
    """通信增强的集中式Critic"""

    def __init__(self, global_state_dim, comm_info_dim, num_agents,
                 hidden_dims=[256, 128, 64]):
        super().__init__()

        # 基础全局状态编码
        self.state_encoder = nn.Sequential(
            nn.Linear(global_state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

        # 通信信息编码
        self.comm_encoder = nn.Sequential(
            nn.Linear(comm_info_dim * num_agents, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(64 + 32, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, global_state, comm_info=None):
        """
        前向传播

        参数:
            global_state: 全局状态 [batch_size, global_state_dim]
            comm_info: 通信信息 [batch_size, comm_info_dim * num_agents] (可选)
        """
        state_features = self.state_encoder(global_state)

        if comm_info is not None:
            comm_features = self.comm_encoder(comm_info)
            combined = torch.cat([state_features, comm_features], dim=-1)
        else:
            combined = torch.cat([state_features, torch.zeros_like(state_features[:, :32])], dim=-1)

        return self.fusion(combined).squeeze(-1)
```

---

## 五、完整Python代码框架

### 5.1 Actor网络实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np


class ActorNetwork(nn.Module):
    """
    PPO Actor网络（共享参数）

    所有无人机使用同一套网络参数，输入各自的局部观测，
    输出动作分布参数（均值和标准差）。
    """

    def __init__(self, obs_dim: int = 11, action_dim: int = 3,
                 hidden_dims: list = [256, 128, 64]):
        super().__init__()

        # 构建特征提取层
        layers = []
        prev_dim = obs_dim
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            # 前两层使用BatchNorm
            if i < len(hidden_dims) - 1:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        self.feature_net = nn.Sequential(*layers)

        # 输出层：均值和对数标准差
        self.mean_head = nn.Linear(prev_dim, action_dim)
        self.log_std_head = nn.Linear(prev_dim, action_dim)

        # 初始化
        self._init_weights()

    def _init_weights(self):
        """正交初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                nn.init.constant_(module.bias, 0)
        # 对数标准差初始化为-0.5
        nn.init.constant_(self.log_std_head.weight, 0)
        nn.init.constant_(self.log_std_head.bias, -0.5)

    def forward(self, obs: torch.Tensor):
        """
        前向传播

        参数:
            obs: 局部观测 [batch_size, obs_dim]

        返回:
            mean: 动作均值 [batch_size, action_dim]
            std: 动作标准差 [batch_size, action_dim]
        """
        features = self.feature_net(obs)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, -20, 2)  # 数值稳定性
        std = torch.exp(log_std)
        return mean, std

    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        """获取动作（用于执行阶段）"""
        mean, std = self.forward(obs)
        if deterministic:
            return torch.tanh(mean)
        else:
            dist = Normal(mean, std)
            raw_action = dist.rsample()  # 重参数化采样
            return torch.tanh(raw_action)

    def get_log_prob(self, obs: torch.Tensor, action: torch.Tensor):
        """计算对数概率（用于训练阶段）"""
        mean, std = self.forward(obs)
        dist = Normal(mean, std)
        # 反tanh变换
        raw_action = torch.atanh(torch.clamp(action, -0.999, 0.999))
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        # Jacobian修正
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)
        return log_prob

    def get_entropy(self, obs: torch.Tensor):
        """计算策略熵"""
        mean, std = self.forward(obs)
        dist = Normal(mean, std)
        return dist.entropy().sum(dim=-1)
```

### 5.2 Critic网络实现

```python
class CriticNetwork(nn.Module):
    """
    集中式Critic网络

    训练时接收全局状态，输出状态价值 V(s_global)。
    执行阶段不使用。
    """

    def __init__(self, state_dim: int, hidden_dims: list = [256, 128, 64]):
        super().__init__()

        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.Tanh())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        """正交初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.constant_(module.bias, 0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数:
            state: 全局状态 [batch_size, state_dim]

        返回:
            value: 状态价值 [batch_size]
        """
        return self.network(state).squeeze(-1)
```

### 5.3 CTDE-PPO算法实现

```python
class CTDE_PPO:
    """
    CTDE框架下的PPO算法实现

    训练阶段：共享Actor + 集中式Critic + 全局奖励
    执行阶段：共享Actor + 局部观测 + 独立决策
    """

    def __init__(self, obs_dim: int, action_dim: int, num_agents: int,
                 global_state_dim: int, config: dict):
        """
        参数:
            obs_dim: 局部观测维度 (11)
            action_dim: 动作维度 (3)
            num_agents: 无人机数量 (3)
            global_state_dim: 全局状态维度
            config: 超参数配置
        """
        self.num_agents = num_agents
        self.config = config

        # 共享Actor网络
        self.actor = ActorNetwork(obs_dim, action_dim)

        # 集中式Critic网络
        self.critic = CriticNetwork(global_state_dim)

        # 优化器
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=config['learning_rate'],
            eps=1e-5
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=config['learning_rate'],
            eps=1e-5
        )

        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.actor_optimizer,
            T_max=config['num_epochs'],
            eta_min=config['min_learning_rate']
        )

        # 超参数
        self.gamma = config['gamma']
        self.gae_lambda = config['gae_lambda']
        self.clip_epsilon = config['clip_epsilon']
        self.value_coef = config['value_coef']
        self.entropy_coef = config['entropy_coef']
        self.max_grad_norm = config['max_grad_norm']

    def select_actions(self, observations: np.ndarray):
        """
        各无人机独立选择动作（分散式执行）

        参数:
            observations: 各无人机的局部观测 [num_agents, obs_dim]

        返回:
            actions: 各无人机的动作 [num_agents, action_dim]
            log_probs: 各动作的对数概率 [num_agents]
        """
        obs_tensor = torch.FloatTensor(observations)
        with torch.no_grad():
            actions = self.actor.get_action(obs_tensor)
            log_probs = self.actor.get_log_prob(obs_tensor, actions)
        return actions.numpy(), log_probs.numpy()

    def compute_gae(self, rewards, values, dones, next_value):
        """计算GAE优势估计"""
        advantages = torch.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            last_gae = advantages[t]

        returns = advantages + values
        return advantages, returns

    def update(self, batch):
        """
        更新策略（CTDE方式）

        参数:
            batch: 包含以下字段的字典
                - obs: 局部观测 [batch_size, obs_dim]
                - actions: 动作 [batch_size, action_dim]
                - old_log_probs: 旧对数概率 [batch_size]
                - advantages: 优势估计 [batch_size]
                - returns: 回报 [batch_size]
                - global_states: 全局状态 [batch_size, global_state_dim]
        """
        obs = batch['obs']
        actions = batch['actions']
        old_log_probs = batch['old_log_probs']
        advantages = batch['advantages']
        returns = batch['returns']
        global_states = batch['global_states']

        # 归一化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Actor更新（使用局部观测）
        new_log_probs = self.actor.get_log_prob(obs, actions)
        entropy = self.actor.get_entropy(obs)
        ratio = torch.exp(new_log_probs - old_log_probs)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon,
                           1 + self.clip_epsilon) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        entropy_loss = -entropy.mean()

        # Critic更新（使用全局状态）
        values = self.critic(global_states)
        value_loss = F.mse_loss(values, returns)

        # 总损失
        total_loss = (actor_loss
                     + self.value_coef * value_loss
                     + self.entropy_coef * entropy_loss)

        # 梯度更新
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            self.max_grad_norm
        )
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        # 计算监控指标
        with torch.no_grad():
            approx_kl = (old_log_probs - new_log_probs).mean()
            clip_frac = (torch.abs(ratio - 1) > self.clip_epsilon).float().mean()

        return {
            'actor_loss': actor_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.mean().item(),
            'approx_kl': approx_kl.item(),
            'clip_fraction': clip_frac.item()
        }

    def save(self, path: str):
        """保存模型"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict()
        }, path)

    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
```

### 5.4 训练主循环

```python
class CTDE_Trainer:
    """CTDE-PPO训练器"""

    def __init__(self, env, config):
        self.env = env
        self.config = config

        # 计算状态维度
        obs_dim = 11  # 局部观测维度
        action_dim = 3  # 动作维度
        num_agents = config['num_uavs']
        global_state_dim = obs_dim * num_agents + 10  # 全局状态维度

        # 创建CTDE-PPO算法
        self.algorithm = CTDE_PPO(
            obs_dim=obs_dim,
            action_dim=action_dim,
            num_agents=num_agents,
            global_state_dim=global_state_dim,
            config=config
        )

        # 经验缓冲区
        self.buffer = RolloutBuffer(
            buffer_size=config['buffer_size'],
            obs_dim=obs_dim,
            action_dim=action_dim,
            num_agents=num_agents
        )

    def train(self):
        """主训练循环"""
        for epoch in range(self.config['num_epochs']):

            # ========== 阶段1：经验收集 ==========
            global_state = self.env.reset()
            episode_rewards = []

            for step in range(self.config['steps_per_epoch']):
                # 获取各无人机局部观测
                observations = self._get_local_observations(global_state)

                # 各无人机独立选择动作
                actions, log_probs = self.algorithm.select_actions(observations)

                # 集中式Critic评估
                with torch.no_grad():
                    global_state_tensor = torch.FloatTensor(
                        self._encode_global_state(global_state)
                    )
                    value = self.algorithm.critic(global_state_tensor).item()

                # 环境执行联合动作
                next_global_state, reward, done, info = self.env.step(actions)

                # 存储经验
                self.buffer.store(
                    observations=observations,
                    actions=actions,
                    rewards=reward,
                    values=value,
                    log_probs=log_probs,
                    dones=done,
                    global_state=self._encode_global_state(global_state)
                )

                global_state = next_global_state
                if done:
                    episode_rewards.append(info['episode_reward'])
                    global_state = self.env.reset()

            # ========== 阶段2：GAE计算 ==========
            with torch.no_grad():
                last_value = self.algorithm.critic(
                    torch.FloatTensor(self._encode_global_state(global_state))
                ).item()

            advantages, returns = self.algorithm.compute_gae(
                rewards=self.buffer.rewards[:self.buffer.ptr],
                values=self.buffer.values[:self.buffer.ptr],
                dones=self.buffer.dones[:self.buffer.ptr],
                next_value=last_value
            )

            # ========== 阶段3：策略更新 ==========
            for update_epoch in range(self.config['num_update_epochs']):
                # 随机打乱数据
                indices = np.random.permutation(self.buffer.ptr)

                for start in range(0, self.buffer.ptr, self.config['batch_size']):
                    end = start + self.config['batch_size']
                    batch_indices = indices[start:end]

                    batch = {
                        'obs': torch.FloatTensor(
                            self.buffer.observations[batch_indices]
                        ),
                        'actions': torch.FloatTensor(
                            self.buffer.actions[batch_indices]
                        ),
                        'old_log_probs': torch.FloatTensor(
                            self.buffer.log_probs[batch_indices]
                        ),
                        'advantages': advantages[batch_indices],
                        'returns': returns[batch_indices],
                        'global_states': torch.FloatTensor(
                            self.buffer.global_states[batch_indices]
                        )
                    }

                    metrics = self.algorithm.update(batch)

            # ========== 阶段4：学习率更新 ==========
            self.algorithm.scheduler.step()

            # ========== 阶段5：日志记录 ==========
            if epoch % self.config['log_interval'] == 0:
                self._log_metrics(epoch, episode_rewards, metrics)

            # ========== 阶段6：模型保存 ==========
            if epoch % self.config['save_interval'] == 0:
                self.algorithm.save(f"checkpoints/ctde_ppo_epoch_{epoch}.pt")

    def _get_local_observations(self, global_state):
        """从全局状态提取各无人机的局部观测"""
        # 实现细节取决于环境的状态表示
        pass

    def _encode_global_state(self, global_state):
        """编码全局状态为Critic输入"""
        # 将全局状态编码为固定长度向量
        pass

    def _log_metrics(self, epoch, episode_rewards, metrics):
        """记录训练指标"""
        print(f"Epoch {epoch}")
        print(f"  平均回报: {np.mean(episode_rewards):.2f}")
        print(f"  Actor损失: {metrics['actor_loss']:.4f}")
        print(f"  价值损失: {metrics['value_loss']:.4f}")
        print(f"  策略熵: {metrics['entropy']:.4f}")
        print(f"  KL散度: {metrics['approx_kl']:.4f}")
        print(f"  裁剪比例: {metrics['clip_fraction']:.4f}")
```

### 5.5 分散式执行器

```python
class DecentralizedExecutor:
    """分散式执行器"""

    def __init__(self, actor_path, num_agents, obs_dim=11, action_dim=3,
                 enable_comm=False, comm_config=None):
        """
        参数:
            actor_path: 训练好的Actor模型路径
            num_agents: 无人机数量
            obs_dim: 局部观测维度
            action_dim: 动作维度
            enable_comm: 是否启用通信
            comm_config: 通信配置
        """
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # 加载训练好的Actor
        self.actor = ActorNetwork(obs_dim, action_dim)
        checkpoint = torch.load(actor_path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.actor.eval()

        # 通信模块（可选）
        self.enable_comm = enable_comm
        if enable_comm and comm_config:
            self.comm_modules = [
                EventTriggeredComm(
                    uav_id=i,
                    d_safe_warn=comm_config.get('d_safe_warn', 3.0),
                    d_safe=comm_config.get('d_safe', 10.0),
                    battery_warn=comm_config.get('battery_warn', 0.2)
                )
                for i in range(num_agents)
            ]
        else:
            self.comm_modules = None

    def execute_step(self, observations, current_time=0.0):
        """
        执行一步决策

        参数:
            observations: 各无人机局部观测 [num_agents, obs_dim]
            current_time: 当前时间

        返回:
            actions: 各无人机动作 [num_agents, action_dim]
        """
        obs_tensor = torch.FloatTensor(observations)

        # 各无人机独立决策
        with torch.no_grad():
            actions = self.actor.get_action(obs_tensor, deterministic=True)

        # 通信处理（可选）
        if self.enable_comm and self.comm_modules:
            for i in range(self.num_agents):
                events = self.comm_modules[i].check_events(
                    observations[i], {}, current_time
                )
                if events:
                    # 触发通信
                    pass

        return actions.numpy()

    def run_episode(self, env, max_steps=1000):
        """
        运行一个完整的episode

        参数:
            env: 环境
            max_steps: 最大步数

        返回:
            total_reward: 总奖励
            steps: 步数
            success: 是否成功
        """
        global_state = env.reset()
        total_reward = 0.0

        for step in range(max_steps):
            # 获取局部观测
            observations = self._get_local_observations(global_state)

            # 执行决策
            actions = self.execute_step(observations, current_time=step * 0.5)

            # 环境步进
            global_state, reward, done, info = env.step(actions)
            total_reward += reward

            if done:
                break

        return total_reward, step + 1, info.get('success', False)

    def _get_local_observations(self, global_state):
        """从全局状态提取局部观测"""
        # 实现细节取决于环境
        pass
```

---

## 六、参数汇总

### 6.1 PPO算法参数

| 参数 | 符号 | 值 | 说明 |
|------|------|-----|------|
| 学习率 | alpha | 3 x 10^-4 | Adam优化器初始学习率 |
| 最小学习率 | alpha_min | 1 x 10^-5 | 余弦退火最低学习率 |
| 折扣因子 | gamma | 0.99 | 未来奖励折扣系数 |
| GAE参数 | lambda | 0.95 | 优势估计平滑系数 |
| 裁剪参数 | epsilon | 0.2 | 策略更新裁剪范围 |
| 价值损失系数 | c_1 | 0.5 | Critic损失权重 |
| 熵正则化系数 | c_2 | 0.01 | 探索鼓励权重 |
| 最大梯度范数 | max_norm | 0.5 | 梯度裁剪阈值 |
| 批次大小 | B | 256 | 每次更新的样本数 |
| 每轮更新次数 | K | 4 | 每个epoch对数据的重复使用次数 |
| 训练轮次 | N_epochs | 1000 | 总训练epoch数 |
| 每轮采样步数 | T | 2048 | 每个epoch与环境交互的步数 |

### 6.2 CTDE框架参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 无人机数量 N | 3-6 | 编队中无人机数量 |
| 局部观测维度 | 11 | 每架无人机的观测维度 |
| 全局状态维度 | 43-76 | N x 11 + 10 (共享信息) |
| 动作维度 | 3 | 三维加速度控制 |
| Actor隐藏层 | [256, 128, 64] | Actor网络结构 |
| Critic隐藏层 | [256, 128, 64] | Critic网络结构 |

### 6.3 通信参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 通信频率 | 1 Hz | 低频通信 |
| 通信消息大小 | ~51 字节 | 单条消息大小 |
| 最大通信距离 | 500 m | 视距通信 |
| 碰撞预警距离 | 3 m | 紧急通信触发 |
| 安全距离 | 10 m | 高优先级通信触发 |
| 电量预警阈值 | 20% | 高优先级通信触发 |
| 定期同步间隔 | 60 s | 低优先级通信触发 |

---

## 七、设计约束与数值稳定性

### 7.1 数值稳定性措施

| 问题 | 解决方案 | 参数 |
|------|---------|------|
| 梯度爆炸 | 梯度范数裁剪 | max_norm=0.5 |
| 价值估计爆炸 | 价值裁剪 | clip范围=epsilon |
| 概率比过大 | 策略裁剪 | epsilon=0.2 |
| 对数概率NaN | 对数标准差截断 | log_std in [-20, 2] |
| tanh饱和 | 动作值截断 | clamp(-0.999, 0.999) |
| 除零错误 | 小常数 | eps=1e-8 |

### 7.2 训练约束

| 约束 | 值 | 说明 |
|------|-----|------|
| 最大episode长度 | 1000步 | 防止无限循环 |
| 最小batch大小 | 64 | 保证梯度估计稳定 |
| 最大策略更新 | 4轮 | 防止过拟合 |
| 早停耐心值 | 50 epochs | 防止无效训练 |

---

## 八、与其他模块的接口

### 8.1 与环境模块的接口

```python
# CTDE-PPO与环境交互
global_state = env.reset()  # 返回全局状态
observations = get_local_obs(global_state)  # 提取局部观测 [N, 11]
actions = ppo.select_actions(observations)  # 返回 [N, 3]
next_global_state, reward, done, info = env.step(actions)
```

### 8.2 与任务分配模块的接口

```python
# 任务分配结果作为环境输入
allocation = adaptive_allocator.allocate(orders, drones, current_time)
env.set_task_allocation(allocation.allocation)
```

### 8.3 与奖励函数模块的接口

```python
# 全局奖励计算
reward, reward_info = reward_calculator.calculate(
    positions=positions,          # [N, 3]
    prev_positions=prev_positions,
    actions=actions,              # [N, 3]
    prev_actions=prev_actions,
    goal_pos=goal_pos,
    goal_positions=goal_positions,
    formation_offsets=formation_offsets,
    obstacles=obstacles,
    cargo_loaded=cargo_loaded,
    cargo_delivered=cargo_delivered
)
```

---

## 九、总结

### 9.1 设计要点

1. **共享Actor网络**：所有无人机使用同一套策略参数，提高参数效率和泛化能力
2. **集中式Critic**：训练时利用全局状态评估联合动作价值，解决信用分配问题
3. **全局状态聚合**：直接拼接所有无人机的局部观测和共享信息，信息无损
4. **局部观测设计**：11维观测包含自身状态、目标信息、障碍物信息、邻居信息
5. **通信机制**：支持低带宽通信、事件触发通信、分层通信三种模式

### 9.2 创新点

1. **参数共享的多智能体PPO**：N架无人机共享Actor参数，提高参数效率和泛化能力
2. **集中式Critic**：训练时利用全局状态评估联合动作价值，解决信用分配问题
3. **事件触发通信**：降低通信开销，仅在必要时交换信息
4. **分层通信架构**：支持大规模无人机集群的可扩展通信
5. **与现有模块的无缝集成**：与任务分配、奖励函数、状态/动作空间设计文档保持一致

### 9.3 适用场景

| 场景 | 无人机数量 | 通信模式 | 说明 |
|------|-----------|---------|------|
| 小规模配送 | 3-5架 | 无通信/低带宽 | 基础CTDE，无需通信 |
| 中规模配送 | 5-10架 | 事件触发 | 碰撞预警时通信 |
| 大规模配送 | 10+架 | 分层通信 | 区域协调 |

---

## 附录：参考文献

1. Schulman, J., et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347, 2017.
2. Schulman, J., et al. "High-dimensional continuous control using generalized advantage estimation." ICLR, 2016.
3. Lowe, R., et al. "Multi-agent actor-critic for mixed cooperative-competitive environments." NeurIPS, 2017.
4. Rashid, T., et al. "QMIX: Monotonic value function factorisation for deep multi-agent reinforcement learning." ICML, 2018.
5. Foerster, J., et al. "Counterfactual multi-agent policy gradients." AAAI, 2018.
6. Mao, H., et al. "Learning transferable multi-agent communication with contrastive learning." NeurIPS, 2021.
7. Jiang, J., & Lu, Z. "Learning attentional communication for multi-agent cooperation." NeurIPS, 2018.
