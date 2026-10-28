# PPO算法与CTDE框架设计文档

> 项目：基于深度强化学习的多无人机协同物流配送路径规划
> 模块：PPO算法设计 + CTDE框架设计
> 版本：v2.0
> 创建日期：2026-07-01
> 更新日期：2026-07-01

---

## 第一部分：PPO算法设计

---

### 一、算法原理

#### 1.1 策略梯度方法

PPO（Proximal Policy Optimization）属于策略梯度（Policy Gradient）方法家族。与基于值函数的方法（如DQN）不同，策略梯度方法直接参数化策略 $\pi_\theta(a|s)$，通过梯度上升优化期望累积回报。

**策略梯度定理**：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot A_t \right]
$$

其中：
- $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \gamma^t r_t\right]$：策略的期望累积回报
- $\pi_\theta(a_t | s_t)$：参数化策略，在状态 $s_t$ 下选择动作 $a_t$ 的概率密度
- $A_t$：优势函数（Advantage Function），衡量动作 $a_t$ 相对于平均水平的优劣
- $\gamma \in [0, 1)$：折扣因子

**优势函数定义**：

$$
A_t = Q^{\pi}(s_t, a_t) - V^{\pi}(s_t)
$$

其中：
- $Q^{\pi}(s_t, a_t) = \mathbb{E}\left[\sum_{l=0}^{\infty} \gamma^l r_{t+l} \mid s_t, a_t\right]$：动作价值函数
- $V^{\pi}(s_t) = \mathbb{E}_{a \sim \pi}\left[Q^{\pi}(s_t, a)\right]$：状态价值函数

**直观理解**：$A_t > 0$ 表示动作 $a_t$ 优于平均水平，应增大其概率；$A_t < 0$ 表示动作 $a_t$ 劣于平均水平，应减小其概率。

#### 1.2 重要性采样

在on-policy算法中，策略更新需要使用当前策略采集的数据。然而，如果每次更新后丢弃旧数据，采样效率极低。重要性采样（Importance Sampling）允许使用旧策略 $\pi_{\theta_{old}}$ 采集的数据来更新新策略 $\pi_\theta$。

**重要性采样比率**：

$$
r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{old}}(a_t | s_t)}
$$

**策略梯度的等价形式**：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_{\theta_{old}}} \left[ r_t(\theta) \cdot \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot A_t \right]
$$

**问题**：当新旧策略差异过大时，重要性采样比率 $r_t(\theta)$ 的方差会急剧增大，导致训练不稳定甚至发散。

#### 1.3 裁剪目标函数

PPO的核心创新是通过裁剪机制限制策略更新幅度，解决重要性采样的方差问题。

**PPO裁剪目标函数**：

$$
L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]
$$

其中 $\epsilon$ 为裁剪参数（本项目取 $\epsilon = 0.2$）。

**裁剪机制的两种情况**：

**情况1：$\hat{A}_t > 0$（动作优于平均）**

$$
L = \min\left(r_t, \; 1+\epsilon\right) \cdot \hat{A}_t
$$

含义：即使新策略大幅增加好动作的概率（$r_t \gg 1$），也只按 $(1+\epsilon) \hat{A}_t$ 计算梯度，防止过度更新。

**情况2：$\hat{A}_t < 0$（动作劣于平均）**

$$
L = \min\left(r_t, \; 1-\epsilon\right) \cdot \hat{A}_t
$$

注意 $\hat{A}_t < 0$，取最小值实际上是选择绝对值更小的负值，即限制策略不要过度减少差动作的概率。

**裁剪机制示意图**：

```
L^CLIP
  ^
  |        /
  |       /  (A > 0, 斜率被限制)
  |      /
  |     /
  |    /
  |   /
  +--/----------+-----> r_t
  | /  1-ε   1  1+ε
  |/
  |  (A < 0, 斜率被限制)
  |
```

**总损失函数**：

$$
L(\theta, \phi) = -L^{CLIP}(\theta) + c_1 L^{VF}(\phi) - c_2 H[\pi_\theta]
$$

其中：
- $L^{CLIP}(\theta)$：策略裁剪目标（最大化）
- $L^{VF}(\phi)$：价值函数损失（最小化）
- $H[\pi_\theta]$：熵正则化（鼓励探索）
- $c_1 = 0.5$：价值损失系数
- $c_2 = 0.01$：熵正则化系数

---

### 二、网络结构设计

#### 2.1 Actor网络（策略网络）

Actor网络将状态映射为动作分布参数（均值 $\mu$ 和标准差 $\sigma$），用于采样连续动作。

**网络架构**：

```
输入层: 状态向量 s ∈ R^D (D = 11, 见状态空间设计文档)
    |
    v
全连接层1: Linear(D, 256) + BatchNorm + ReLU
    |
    v
全连接层2: Linear(256, 128) + BatchNorm + ReLU
    |
    v
全连接层3: Linear(128, 64) + ReLU
    |
    v
┌──────────────────────────────────────────────────┐
│ 输出分支1: Linear(64, 3) → μ (动作均值)           │
│ 输出分支2: Linear(64, 3) → log_σ (对数标准差)     │
└──────────────────────────────────────────────────┘
    |
    v
重参数化采样: u ~ N(μ, σ²)
    |
    v
动作映射: a = a_max * tanh(u)
```

**数学表达**：

隐藏层前向传播：

$$
\begin{aligned}
h_1 &= \text{ReLU}(\text{BN}(W_1 s + b_1)) \\
h_2 &= \text{ReLU}(\text{BN}(W_2 h_1 + b_2)) \\
h_3 &= \text{ReLU}(W_3 h_2 + b_3)
\end{aligned}
$$

输出层计算：

$$
\begin{aligned}
\mu &= W_\mu h_3 + b_\mu \\
\log\sigma &= \text{clamp}(W_\sigma h_3 + b_\sigma, -20, 2) \\
\sigma &= \exp(\log\sigma)
\end{aligned}
$$

**参数维度**：

| 层 | 输入维度 | 输出维度 | 参数数量 | 说明 |
|-----|---------|---------|---------|------|
| 全连接层1 | 11 | 256 | 11 x 256 + 256 = 3,072 | 含BatchNorm参数 |
| 全连接层2 | 256 | 128 | 256 x 128 + 128 = 32,896 | 含BatchNorm参数 |
| 全连接层3 | 128 | 64 | 128 x 64 + 64 = 8,256 | |
| 均值输出层 | 64 | 3 | 64 x 3 + 3 = 195 | |
| 标准差输出层 | 64 | 3 | 64 x 3 + 3 = 195 | |

**总参数量**: 约 44,614

**激活函数选择说明**：
- **ReLU**：保证梯度在正区间稳定传播，计算效率高，适合策略学习
- **对数标准差截断**：$\log\sigma \in [-20, 2]$，对应 $\sigma \in [2 \times 10^{-9}, 7.39]$，防止数值不稳定

#### 2.2 Critic网络（价值网络）

Critic网络将状态映射为标量状态价值 $V(s)$，用于计算优势函数和指导策略更新。

**网络架构**：

```
输入层: 状态向量 s ∈ R^D
    |
    v
全连接层1: Linear(D, 256) + Tanh
    |
    v
全连接层2: Linear(256, 128) + Tanh
    |
    v
全连接层3: Linear(128, 64) + Tanh
    |
    v
输出层: Linear(64, 1) → V(s)
```

**数学表达**：

$$
\begin{aligned}
h_1 &= \text{Tanh}(W_1 s + b_1) \\
h_2 &= \text{Tanh}(W_2 h_1 + b_2) \\
h_3 &= \text{Tanh}(W_3 h_2 + b_3) \\
V(s) &= W_v h_3 + b_v
\end{aligned}
$$

**激活函数选择说明**：
- **Tanh**：输出有界（[-1, 1]），避免价值估计爆炸，提高训练稳定性
- 与Actor使用不同激活函数，体现两个网络的不同职责

#### 2.3 权重初始化

采用正交初始化（Orthogonal Initialization），保证梯度在深层网络中的稳定传播：

$$
W \sim \text{Orthogonal}(\sqrt{2}) \quad \text{（隐藏层，配合ReLU）}
$$

$$
W \sim \text{Orthogonal}(1.0) \quad \text{（隐藏层，配合Tanh）}
$$

$$
W \sim \text{Orthogonal}(0.01) \quad \text{（输出层）}
$$

偏置初始化为零：$b = \mathbf{0}$

对数标准差初始化：$\log\sigma_0 = -0.5 \Rightarrow \sigma_0 \approx 0.607$，控制初始探索程度。

#### 2.4 网络结构对比表

| 参数 | Actor网络 | Critic网络 |
|------|----------|-----------|
| 隐藏层维度 | [256, 128, 64] | [256, 128, 64] |
| 激活函数 | ReLU | Tanh |
| BatchNorm | 前两层使用 | 不使用 |
| 输出维度 | 3 + 3 (mu和log_sigma) | 1 |
| 权重初始化 | 正交(0.01) | 正交(1.0) |
| 总参数量 | ~44,614 | ~42,497 |

---

### 三、超参数设置

#### 3.1 核心超参数表

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
| 每轮更新次数 | K | 10 | 每个epoch对数据的重复使用次数 |
| 训练轮次 | N_epochs | 1000 | 总训练epoch数 |
| 每轮采样步数 | T | 2048 | 每个epoch与环境交互的步数 |

#### 3.2 GAE参数详解

GAE（Generalized Advantage Estimation）通过参数 lambda 控制偏差-方差权衡：

$$
\hat{A}_t^{GAE(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
$$

其中TD误差为：

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

**递推计算**（从后向前，便于实现）：

$$
\hat{A}_t = \delta_t + \gamma \lambda \hat{A}_{t+1}
$$

**lambda 的作用**：
- lambda = 0：$\hat{A}_t = \delta_t$，低方差但高偏差（单步TD误差）
- lambda = 1：$\hat{A}_t = \sum_{l=0}^{\infty} \gamma^l \delta_{t+l}$，高方差但低偏差（蒙特卡洛回报）
- lambda = 0.95：折中选择，兼顾偏差与方差

#### 3.3 学习率衰减策略

采用余弦退火（Cosine Annealing）策略：

$$
\alpha_t = \alpha_{min} + \frac{1}{2}(\alpha_{max} - \alpha_{min})\left(1 + \cos\left(\frac{t}{T_{max}}\pi\right)\right)
$$

参数设置：
- $\alpha_{max} = 3 \times 10^{-4}$：初始学习率
- $\alpha_{min} = 1 \times 10^{-5}$：最小学习率
- $T_{max} = 1000$：总训练轮次

**学习率衰减曲线**：

```
学习率 α
  ^
  |___
  |   \
  |    \
  |     \
  |      \
  |       \___
  +------------+-> epoch
  0          1000
```

#### 3.4 Adam优化器参数

$$
\theta_{t+1} = \theta_t - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon_{adam}}
$$

| 参数 | 值 | 说明 |
|------|-----|------|
| alpha | 3 x 10^-4 | 学习率 |
| beta_1 | 0.9 | 一阶矩估计衰减率 |
| beta_2 | 0.999 | 二阶矩估计衰减率 |
| epsilon_adam | 10^-8 | 数值稳定性常数 |

---

### 四、训练策略

#### 4.1 经验回放

PPO采用on-policy经验收集方式，每个训练epoch与环境交互采集新数据。

**经验缓冲区结构**：

```python
class RolloutBuffer:
    """PPO经验回放缓冲区"""
    def __init__(self, buffer_size, state_dim, action_dim):
        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)
        self.ptr = 0
        self.size = buffer_size
```

**经验收集流程**：

```
算法：经验收集
输入：当前策略 pi_theta，环境 env
输出：经验批次 {(s_t, a_t, r_t, s_{t+1}, log_pi_t, V_t, done_t)}

1. 初始化状态 s_0 = env.reset()
2. for t = 0, 1, ..., T-1 do
3.     计算动作分布：mu, sigma = Actor(s_t)
4.     采样动作：a_t ~ N(mu, sigma^2)
5.     计算对数概率：log_pi_t = log pi_theta(a_t|s_t)
6.     计算状态价值：V_t = Critic(s_t)
7.     执行动作：s_{t+1}, r_t, done_t = env.step(a_t)
8.     存储经验：buffer.store(s_t, a_t, r_t, s_{t+1}, log_pi_t, V_t, done_t)
9.     if done_t then
10.        s_{t+1} = env.reset()
11.    end if
12. end for
13. 计算GAE优势估计和回报
14. 返回经验批次
```

#### 4.2 梯度裁剪

采用梯度范数裁剪（Gradient Norm Clipping），防止梯度爆炸：

$$
\hat{g} = g \cdot \frac{c}{\max(\|g\|_2, c)}
$$

其中：
- $g$：原始梯度向量
- $c = 0.5$：最大梯度范数阈值
- $\hat{g}$：裁剪后的梯度

**实现**：

```python
torch.nn.utils.clip_grad_norm_(
    list(self.actor.parameters()) + list(self.critic.parameters()),
    max_norm=0.5
)
```

#### 4.3 学习率衰减

每个epoch结束后更新学习率：

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=1000, eta_min=1e-5
)
# 每个epoch结束后调用
scheduler.step()
```

#### 4.4 优先级经验回放

基于TD误差的优先级采样，提高数据利用效率：

**优先级计算**：

$$
p_i = (|\delta_i| + \epsilon_{priority})^{\alpha_{priority}}
$$

**采样概率**：

$$
P(i) = \frac{p_i}{\sum_k p_k}
$$

**重要性采样权重**：

$$
w_i = \left(\frac{1}{N \cdot P(i)}\right)^{\beta}
$$

其中 beta 从 0.4 线性增长到 1.0，逐步消除采样偏差。

---

### 五、PPO完整训练算法伪代码

```
算法：PPO主训练循环
输入：环境 env，网络参数 theta, phi，超参数集合
输出：训练后的策略 pi_theta*

1. 初始化：
    - 创建Actor网络，参数 theta
    - 创建Critic网络，参数 phi
    - 创建经验回放缓冲区 buffer，容量 T = 2048
    - 创建Adam优化器，学习率 alpha = 3e-4
    - 初始化余弦退火学习率调度器

2. for epoch = 1, 2, ..., N_epochs do
3.
4.     // 阶段1：经验收集
5.     for t = 0, 1, ..., T-1 do
6.         mu, sigma = Actor(s_t)
7.         a_t ~ N(mu, sigma^2)
8.         log_pi_t = log N(a_t; mu, sigma^2)
9.         V_t = Critic(s_t)
10.        s_{t+1}, r_t, done_t = env.step(a_t)
11.        buffer.store(s_t, a_t, r_t, V_t, log_pi_t, done_t)
12.        if done_t then s_{t+1} = env.reset()
13.    end for
14.
15.    // 阶段2：GAE计算
16.    V_T = Critic(s_T)  // 最后一个状态的价值
17.    for t = T-1, T-2, ..., 0 do
18.        delta_t = r_t + gamma * V_{t+1} * (1 - done_t) - V_t
19.        A_t = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}
20.        R_t = A_t + V_t
21.    end for
22.    归一化优势：A = (A - mean(A)) / (std(A) + 1e-8)
23.
24.    // 阶段3：策略更新（K轮）
25.    for k = 1, 2, ..., K do
26.        将buffer随机分成 mini-batches (batch_size=256)
27.        for each mini-batch B do
28.            // Actor损失
29.            new_log_probs = Actor.get_log_prob(s, a)
30.            ratio = exp(new_log_probs - old_log_probs)
31.            surr1 = ratio * A
32.            surr2 = clamp(ratio, 1-eps, 1+eps) * A
33.            L_clip = -min(surr1, surr2).mean()
34.            entropy = Actor.get_entropy(s)
35.            actor_loss = L_clip - c2 * entropy.mean()
36.
37.            // Critic损失
38.            V_pred = Critic(s)
39.            value_loss = MSE(V_pred, R)
40.            critic_loss = c1 * value_loss
41.
42.            // 梯度更新
43.            total_loss = actor_loss + critic_loss
44.            optimizer.zero_grad()
45.            total_loss.backward()
46.            clip_grad_norm_(parameters, max_norm=0.5)
47.            optimizer.step()
48.        end for
49.    end for
50.
51.    // 阶段4：学习率更新
52.    scheduler.step()
53.
54.    // 阶段5：日志记录
55.    记录：平均回报、策略熵、价值损失、概率比统计
56.
57.    // 阶段6：模型保存
58.    if epoch % save_interval == 0 then
59.        保存模型检查点
60.    end if
61.
62. end for
63.
64. 返回训练后的策略 pi_theta*
```

---

## 第二部分：CTDE框架设计

---

### 六、CTDE框架概述

#### 6.1 什么是CTDE

CTDE（Centralized Training with Decentralized Execution，集中式训练分散式执行）是多智能体强化学习（MARL）中的主流训练范式。其核心思想是：

- **训练阶段**：利用全局信息（所有智能体的状态、动作、奖励）训练策略和价值网络
- **执行阶段**：每个智能体仅基于局部观测独立决策，无需全局信息

这种设计完美契合实际无人机配送场景：训练可以在拥有完整信息的仿真环境中进行，而部署时每架无人机只能获取自身的传感器数据。

#### 6.2 CTDE与本项目的契合

| 方面 | 训练阶段（集中式） | 执行阶段（分散式） |
|------|-------------------|-------------------|
| 状态信息 | 全局状态（所有无人机位置、速度、目标等） | 局部观测（自身位置、速度、最近障碍物等） |
| 决策主体 | 集中式Critic统一评估 | 独立Actor各自决策 |
| 奖励信号 | 全局奖励（考虑所有无人机） | 无（执行时不计算奖励） |
| 通信需求 | 无限制（仿真环境） | 低带宽、事件触发 |

#### 6.3 框架总体架构

```
                    ┌─────────────────────────────────────────────────┐
                    │              集中式训练阶段                      │
                    │                                                 │
                    │   全局状态 s_global ──> 集中式Critic ──> V(s)   │
                    │         |                                       │
                    │         v                                       │
                    │   ┌──────────────────────────────────────┐     │
                    │   │          共享Actor网络                │     │
                    │   │   s_local_i --> a_i (i=1,...,N)      │     │
                    │   └──────────────────────────────────────┘     │
                    │         |                                       │
                    │         v                                       │
                    │   环境交互 --> 全局奖励 R --> 更新网络           │
                    └─────────────────────────────────────────────────┘

                    ┌─────────────────────────────────────────────────┐
                    │              分散式执行阶段                      │
                    │                                                 │
                    │   UAV_1: s_local_1 --> Actor_1 --> a_1          │
                    │   UAV_2: s_local_2 --> Actor_2 --> a_2          │
                    │   UAV_3: s_local_3 --> Actor_3 --> a_3          │
                    │   ...                                           │
                    │   UAV_N: s_local_N --> Actor_N --> a_N          │
                    │                                                 │
                    │   （各无人机独立决策，无需通信）                  │
                    └─────────────────────────────────────────────────┘
```

---

### 七、集中式训练设计

#### 7.1 共享Actor网络

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
        # 与单机PPO的Actor结构相同
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

#### 7.2 全局状态输入Critic

集中式Critic在训练时接收全局状态信息，能够评估联合动作的价值。这是CTDE的核心优势：训练时利用全局信息，执行时不需要。

**全局状态定义**：

$$
s_{global} = \begin{bmatrix} o_1 \\ o_2 \\ \vdots \\ o_N \\ \mathbf{g}_{shared} \end{bmatrix} \in \mathbb{R}^{N \cdot D_{obs} + D_{shared}}
$$

其中：
- $o_i \in \mathbb{R}^{D_{obs}}$：无人机 $i$ 的局部观测（11维）
- $\mathbf{g}_{shared}$：共享的全局信息（目标位置、障碍物信息等）
- $N$：无人机数量
- $D_{obs} = 11$：局部观测维度

**集中式Critic网络结构**：

```
全局状态输入: s_global ∈ R^(N*11 + D_shared)
    |
    v
全连接层1: Linear(D_global, 256) + ReLU
    |
    v
全连接层2: Linear(256, 128) + ReLU
    |
    v
全连接层3: Linear(128, 64) + ReLU
    |
    v
输出层: Linear(64, 1) → V(s_global)
```

**数学表达**：

$$
V_\phi(s_{global}) = W_v \cdot \text{ReLU}(W_3 \cdot \text{ReLU}(W_2 \cdot \text{ReLU}(W_1 s_{global} + b_1) + b_2) + b_3) + b_v
$$

**与独立Critic的对比**：

| 特性 | 集中式Critic | 独立Critic |
|------|-------------|-----------|
| 输入 | 全局状态 | 局部观测 |
| 评估范围 | 联合动作价值 | 单智能体动作价值 |
| 信息量 | 完整 | 局部 |
| 训练阶段 | 使用 | 使用 |
| 执行阶段 | 不使用 | 不使用 |

#### 7.3 全局奖励函数

训练时使用全局奖励信号，考虑所有无人机的整体表现：

$$
R_{global} = \sum_{i=1}^{N} R_i + R_{cooperation}
$$

其中：
- $R_i$：无人机 $i$ 的个体奖励（详见奖励函数设计文档）
- $R_{cooperation}$：协同奖励项

**协同奖励设计**：

$$
R_{cooperation} = R_{task\_completion} + R\_collision\_avoidance + R\_formation
$$

各分量定义：

| 分量 | 公式 | 说明 |
|------|------|------|
| 任务完成 | $R_{task} = \sum_{k=1}^{K} 50 \cdot \mathbb{1}[\text{delivery}_k \text{ done}]$ | 完成配送任务的总奖励 |
| 避碰协同 | $R_{collision} = -200 \cdot \mathbb{1}[\text{collision}]$ | 碰撞惩罚 |
| 队形保持 | $R_{form} = -1.0 \cdot \frac{1}{N}\sum_i \|p_i - p_{des}^i\|$ | 编队偏差惩罚 |

**全局奖励的优势**：
- 避免"信用分配"问题：每架无人机都能从全局成功中获益
- 鼓励协同行为：无人机学会互相配合而非各自为政
- 简化算法设计：无需设计复杂的局部奖励分解机制

---

### 八、分散式执行设计

#### 8.1 独立Actor网络

在执行阶段，每架无人机加载相同的Actor网络参数，但基于各自的局部观测独立做出决策。

**局部观测定义**：

$$
o_i = \begin{bmatrix} x_i \\ y_i \\ z_i \\ v_{x,i} \\ v_{y,i} \\ v_{z,i} \\ d_{goal,i} \\ d_{obs,i} \\ d_{uav,i} \\ E_{remain,i} \\ p_{delivery,i} \end{bmatrix} \in \mathbb{R}^{11}
$$

**各维度说明**：

| 维度 | 符号 | 含义 | 取值范围 |
|------|------|------|---------|
| 1-3 | (x, y, z) | 三维位置 | [0,100] x [0,100] x [10,120] |
| 4-6 | (vx, vy, vz) | 三维速度 | [-15, 15] m/s |
| 7 | d_goal | 到目标距离 | [0, 150] m |
| 8 | d_obs | 到最近障碍物距离 | [0, 100] m |
| 9 | d_uav | 到最近其他无人机距离 | [0, 100] m |
| 10 | E_remain | 剩余电量比例 | [0, 1] |
| 11 | p_delivery | 配送进度 | {0, 1} 独热编码 |

**执行阶段决策流程**：

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

#### 8.2 局部观测与全局状态的关系

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
(自身+邻居)   (自身+邻居)      (自身+邻居)
```

**局部观测的信息层次**：

| 信息层次 | 内容 | 是否包含 |
|---------|------|---------|
| 自身状态 | 位置、速度、电量 | 是 |
| 目标信息 | 到目标距离、配送进度 | 是 |
| 障碍物信息 | 到最近障碍物距离 | 是 |
| 其他无人机 | 到最近无人机距离 | 是（简化） |
| 其他无人机完整状态 | 所有无人机位置速度 | 否 |
| 全局目标位置 | 所有目标坐标 | 否 |

---

### 九、通信机制设计

#### 9.1 低带宽通信协议

在实际部署中，无人机之间的通信带宽有限。设计低带宽通信协议：

**通信消息格式**：

```python
@dataclass
class CommMessage:
    """无人机通信消息"""
    uav_id: str                    # 无人机ID
    position: np.ndarray           # 位置 (3,)
    velocity: np.ndarray           # 速度 (3,)
    battery_level: float           # 电量 [0, 1]
    task_status: str               # 任务状态
    timestamp: float               # 时间戳

    def to_bytes(self) -> bytes:
        """序列化为字节（约32字节）"""
        # 位置(12B) + 速度(12B) + 电量(4B) + ID(4B) = 32字节
        pass
```

**带宽需求分析**：

| 参数 | 值 |
|------|-----|
| 单条消息大小 | ~32 字节 |
| 通信频率 | 1 Hz（每秒1次） |
| 无人机数量 | 3 架 |
| 总带宽需求 | 32 x 3 = 96 字节/秒 = 768 bps |

#### 9.2 事件触发通信

并非每个时间步都需要通信。设计事件触发机制，仅在必要时进行信息交换：

**触发条件**：

| 触发事件 | 条件 | 优先级 |
|---------|------|--------|
| 碰撞风险 | $d_{uav} < d_{safe}$ (10m) | 高 |
| 任务完成 | 到达目标点 | 中 |
| 电量预警 | $E_{remain} < 0.2$ | 高 |
| 新订单 | 系统分配新任务 | 中 |
| 定期同步 | 每60秒 | 低 |

**事件触发通信算法**：

```
算法：事件触发通信
输入：当前状态 s_t，事件队列 events
输出：是否触发通信，通信内容

1. 检查碰撞风险：
   if min(d_uav_i) < 10m then
       触发通信，发送位置和速度
       优先级 = HIGH

2. 检查任务事件：
   if 任务完成 or 新订单 then
       触发通信，发送任务状态
       优先级 = MEDIUM

3. 检查电量：
   if E_remain < 0.2 then
       触发通信，发送电量和位置
       优先级 = HIGH

4. 定期同步：
   if time_since_last_comm > 60s then
       触发通信，发送完整状态
       优先级 = LOW

5. 返回通信决策和内容
```

#### 9.3 通信对训练的影响

在训练阶段，通信信息可以作为Critic的额外输入：

$$
s_{global}^{comm} = s_{global} \oplus \mathbf{m}_{comm}
$$

其中 $\mathbf{m}_{comm}$ 为通信获得的邻居信息向量。

**通信增强的状态表示**：

$$
s_{global}^{comm} = \begin{bmatrix} o_1 \\ o_2 \\ \vdots \\ o_N \\ \mathbf{g}_{shared} \\ \mathbf{m}_{1 \to all} \\ \mathbf{m}_{2 \to all} \\ \vdots \\ \mathbf{m}_{N \to all} \end{bmatrix}
$$

---

### 十、CTDE训练算法伪代码

```
算法：CTDE-PPO多智能体训练
输入：环境 env（多无人机），Actor参数 theta，Critic参数 phi
输出：训练后的共享策略 theta*

1. 初始化：
    - 创建共享Actor网络，参数 theta
    - 创建集中式Critic网络，参数 phi
    - 创建经验缓冲区 buffer
    - 创建Adam优化器

2. for epoch = 1, 2, ..., N_epochs do
3.
4.     // ========== 阶段1：分散式经验收集 ==========
5.     初始化：s_global = env.reset()
6.     for t = 0, 1, ..., T-1 do
7.         // 各无人机独立决策
8.         for i = 1, 2, ..., N do
9.             o_i = get_local_obs(s_global, i)
10.            mu_i, sigma_i = Actor_theta(o_i)
11.            a_i ~ N(mu_i, sigma_i^2)
12.            a_i_mapped = tanh(a_i) * a_max
13.        end for
14.
15.        // 环境执行联合动作
16.        s_global', R_global, done = env.step([a_1, ..., a_N])
17.
18.        // 计算集中式Critic的价值估计
19.        V_t = Critic_phi(s_global)
20.
21.        // 存储经验
22.        for i = 1, ..., N do
23.            buffer.store(o_i, a_i, R_global, V_t, log_pi_i, done)
24.        end for
25.
26.        s_global = s_global'
27.        if done then s_global = env.reset()
28.    end for
29.
30.    // ========== 阶段2：GAE计算 ==========
31.    V_T = Critic_phi(s_global_T)
32.    for t = T-1, T-2, ..., 0 do
33.        delta_t = R_t + gamma * V_{t+1} * (1-done_t) - V_t
34.        A_t = delta_t + gamma * lambda * (1-done_t) * A_{t+1}
35.        R_t = A_t + V_t
36.    end for
37.    归一化优势
38.
39.    // ========== 阶段3：策略更新 ==========
40.    for k = 1, 2, ..., K do
41.        随机采样 mini-batches
42.        for each batch B do
43.            // --- Actor更新（使用局部观测） ---
44.            对batch中每个样本：
45.                new_log_pi_i = Actor_theta(o_i).log_prob(a_i)
46.            ratio = exp(new_log_pi - old_log_pi)
47.            surr1 = ratio * A
48.            surr2 = clamp(ratio, 1-eps, 1+eps) * A
49.            actor_loss = -min(surr1, surr2).mean()
50.            actor_loss -= c2 * entropy.mean()
51.
52.            // --- Critic更新（使用全局状态） ---
53.            V_pred = Critic_phi(s_global)
54.            critic_loss = c1 * MSE(V_pred, R)
55.
56.            // --- 梯度更新 ---
57.            total_loss = actor_loss + critic_loss
58.            optimizer.zero_grad()
59.            total_loss.backward()
60.            clip_grad_norm_(parameters, max_norm=0.5)
61.            optimizer.step()
62.        end for
63.    end for
64.
65.    // ========== 阶段4：学习率更新 ==========
66.    scheduler.step()
67.
68.    // ========== 阶段5：日志记录 ==========
69.    记录：全局回报、各无人机回报、碰撞率、任务完成率
70.
71. end for
72.
73. 返回训练后的共享策略 theta*
```

---

### 十一、完整Python代码框架

#### 11.1 Actor网络实现

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

#### 11.2 Critic网络实现

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

#### 11.3 CTDE-PPO算法实现

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

#### 11.4 训练主循环

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

---

### 十二、参数汇总

#### 12.1 PPO算法参数

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
| 每轮更新次数 | K | 10 | 每个epoch对数据的重复使用次数 |
| 训练轮次 | N_epochs | 1000 | 总训练epoch数 |
| 每轮采样步数 | T | 2048 | 每个epoch与环境交互的步数 |

#### 12.2 CTDE框架参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 无人机数量 N | 3 | 编队中无人机数量 |
| 局部观测维度 | 11 | 每架无人机的观测维度 |
| 全局状态维度 | 43 | N x 11 + 10 (共享信息) |
| 动作维度 | 3 | 三维加速度控制 |
| Actor隐藏层 | [256, 128, 64] | Actor网络结构 |
| Critic隐藏层 | [256, 128, 64] | Critic网络结构 |
| 通信频率 | 1 Hz | 事件触发通信频率 |
| 通信消息大小 | ~32 字节 | 单条消息大小 |

---

### 十三、设计约束与数值稳定性

#### 13.1 数值稳定性措施

| 问题 | 解决方案 | 参数 |
|------|---------|------|
| 梯度爆炸 | 梯度范数裁剪 | max_norm=0.5 |
| 价值估计爆炸 | 价值裁剪 | clip范围=epsilon |
| 概率比过大 | 策略裁剪 | epsilon=0.2 |
| 对数概率NaN | 对数标准差截断 | log_std in [-20, 2] |
| tanh饱和 | 动作值截断 | clamp(-0.999, 0.999) |
| 除零错误 | 小常数 | eps=1e-8 |

#### 13.2 训练约束

| 约束 | 值 | 说明 |
|------|-----|------|
| 最大episode长度 | 1000步 | 防止无限循环 |
| 最小batch大小 | 64 | 保证梯度估计稳定 |
| 最大策略更新 | 10轮 | 防止过拟合 |
| 早停耐心值 | 50 epochs | 防止无效训练 |

---

### 十四、与其他模块的接口

#### 14.1 与环境模块的接口

```python
# CTDE-PPO与环境交互
global_state = env.reset()  # 返回全局状态
observations = get_local_obs(global_state)  # 提取局部观测 [N, 11]
actions = ppo.select_actions(observations)  # 返回 [N, 3]
next_global_state, reward, done, info = env.step(actions)
```

#### 14.2 与任务分配模块的接口

```python
# 任务分配结果作为环境输入
allocation = adaptive_allocator.allocate(orders, drones, current_time)
env.set_task_allocation(allocation.allocation)
```

#### 14.3 与奖励函数模块的接口

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

### 十五、总结

#### 15.1 设计要点

1. **PPO算法**：采用裁剪目标函数限制策略更新幅度，结合GAE优势估计平衡偏差与方差
2. **网络结构**：Actor使用ReLU激活（3层全连接256-128-64），Critic使用Tanh激活（同结构）
3. **超参数**：学习率3e-4（余弦退火），gamma=0.99，lambda=0.95，epsilon=0.2
4. **训练策略**：梯度裁剪(0.5)，经验回放，学习率衰减
5. **CTDE框架**：共享Actor + 集中式Critic，训练用全局信息，执行用局部观测
6. **通信机制**：低带宽事件触发通信，碰撞风险和任务事件触发

#### 15.2 创新点

1. **参数共享的多智能体PPO**：N架无人机共享Actor参数，提高参数效率和泛化能力
2. **集中式Critic**：训练时利用全局状态评估联合动作价值，解决信用分配问题
3. **事件触发通信**：降低通信开销，仅在必要时交换信息
4. **与现有模块的无缝集成**：与任务分配、奖励函数、状态/动作空间设计文档保持一致

---

### 附录：参考文献

1. Schulman, J., et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347, 2017.
2. Schulman, J., et al. "High-dimensional continuous control using generalized advantage estimation." ICLR, 2016.
3. Lowe, R., et al. "Multi-agent actor-critic for mixed cooperative-competitive environments." NeurIPS, 2017.
4. Rashid, T., et al. "QMIX: Monotonic value function factorisation for deep multi-agent reinforcement learning." ICML, 2018.
5. Foerster, J., et al. "Counterfactual multi-agent policy gradients." AAAI, 2018.
