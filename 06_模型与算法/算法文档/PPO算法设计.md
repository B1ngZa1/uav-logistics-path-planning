# PPO算法设计文档

> 项目：基于深度强化学习的多无人机协同物流配送路径规划
> 模块：PPO算法设计（Proximal Policy Optimization）
> 版本：v1.0
> 创建日期：2026-07-01

---

## 一、算法概述

### 1.1 PPO算法原理

PPO（Proximal Policy Optimization）是Schulman等人于2017年提出的策略梯度强化学习算法。其核心思想是通过裁剪机制限制策略更新幅度，从而在保证训练稳定性的同时实现高效的策略优化。

**策略梯度基本框架**：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot A_t \right]
$$

其中：
- $J(\theta)$：策略的期望回报
- $\pi_\theta(a_t | s_t)$：策略网络在状态 $s_t$ 下选择动作 $a_t$ 的概率
- $A_t$：优势函数，衡量动作 $a_t$ 相对于平均水平的优劣

**PPO的改进**：

PPO通过裁剪目标函数限制新旧策略之间的差异，避免过大的策略更新导致训练不稳定：

$$
L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) A_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]
$$

其中 $r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{old}}(a_t | s_t)}$ 为新旧策略的概率比，$\epsilon$ 为裁剪参数。

### 1.2 Actor-Critic架构

PPO采用Actor-Critic架构：
- **Actor（策略网络）**：输出动作分布，决定在给定状态下选择什么动作
- **Critic（价值网络）**：评估状态价值，为Actor提供学习信号

**Actor-Critic更新流程**：

```
状态 s_t → Actor网络 → 动作分布 π(a|s_t) → 采样动作 a_t
    ↓
环境执行 a_t → 获得奖励 r_t、新状态 s_{t+1}
    ↓
计算优势函数 A_t = r_t + γV(s_{t+1}) - V(s_t)
    ↓
更新Actor：最大化裁剪目标 L^CLIP
更新Critic：最小化价值损失 L^VF
```

### 1.3 优势函数计算

采用GAE（Generalized Advantage Estimation）计算优势函数，平衡偏差与方差：

$$
\hat{A}_t^{GAE(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
$$

其中TD误差为：

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

展开形式：

$$
\hat{A}_t = \delta_t + (\gamma\lambda)\delta_{t+1} + (\gamma\lambda)^2\delta_{t+2} + \cdots
$$

**参数说明**：
- $\gamma = 0.99$：折扣因子，平衡即时奖励与未来奖励
- $\lambda = 0.95$：GAE参数，控制优势估计的偏差-方差权衡

**递推计算**（从后向前）：

$$
\hat{A}_t = \delta_t + \gamma \lambda \hat{A}_{t+1}
$$

---

## 二、网络结构设计

### 2.1 Actor网络

Actor网络将状态映射为动作分布参数（均值和标准差），用于采样连续动作。

**网络架构**：

```
输入层: 状态向量 s ∈ R^D (D = 10 + M + 1)
    |
    v
全连接层1: Linear(D, 256)
    |
    v
激活函数: ReLU
    |
    v
全连接层2: Linear(256, 128)
    |
    v
激活函数: ReLU
    |
    v
全连接层3: Linear(128, 64)
    |
    v
激活函数: ReLU
    |
    v
┌─────────────────────────────────────────┐
│ 输出分支1: Linear(64, 3) → μ (动作均值)   │
│ 输出分支2: Linear(64, 3) → log_σ (对数标准差)│
└─────────────────────────────────────────┘
    |
    v
动作采样: a ~ N(μ, σ²)
    |
    v
动作映射: a_mapped = a_max * tanh(a)
```

**数学表达**：

隐藏层计算：

$$
\begin{aligned}
h_1 &= \text{ReLU}(W_1 s + b_1) \\
h_2 &= \text{ReLU}(W_2 h_1 + b_2) \\
h_3 &= \text{ReLU}(W_3 h_2 + b_3)
\end{aligned}
$$

输出层计算：

$$
\begin{aligned}
\mu &= W_\mu h_3 + b_\mu \\
\log\sigma &= W_\sigma h_3 + b_\sigma
\end{aligned}
$$

**参数维度**：

| 层 | 输入维度 | 输出维度 | 参数数量 |
|-----|---------|---------|---------|
| 全连接层1 | D | 256 | D × 256 + 256 |
| 全连接层2 | 256 | 128 | 256 × 128 + 128 |
| 全连接层3 | 128 | 64 | 128 × 64 + 64 |
| 均值输出层 | 64 | 3 | 64 × 3 + 3 |
| 标准差输出层 | 64 | 3 | 64 × 3 + 3 |

对于复杂场景（M=20，D=31）：

$$
\text{总参数} = (31 \times 256 + 256) + (256 \times 128 + 128) + (128 \times 64 + 64) + 2 \times (64 \times 3 + 3) = 42,502
$$

### 2.2 Critic网络

Critic网络将状态映射为标量状态价值 $V(s)$，用于计算优势函数和指导策略更新。

**网络架构**：

```
输入层: 状态向量 s ∈ R^D
    |
    v
全连接层1: Linear(D, 256)
    |
    v
激活函数: Tanh
    |
    v
全连接层2: Linear(256, 128)
    |
    v
激活函数: Tanh
    |
    v
全连接层3: Linear(128, 64)
    |
    v
激活函数: Tanh
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
- Actor使用ReLU：保证梯度在正区间的稳定传播，适合策略学习
- Critic使用Tanh：输出有界，避免价值估计爆炸，提高训练稳定性

### 2.3 权重初始化

采用正交初始化（Orthogonal Initialization），保证梯度在深层网络中的稳定传播：

$$
W \sim \text{Orthogonal}(\sqrt{2}) \quad \text{（隐藏层）}
$$

$$
W \sim \text{Orthogonal}(0.01) \quad \text{（输出层）}
$$

偏置初始化为零：

$$
b = \mathbf{0}
$$

对数标准差初始化为特定值，控制初始探索程度：

$$
\log\sigma_0 = -0.5 \quad \Rightarrow \quad \sigma_0 \approx 0.607
$$

### 2.4 网络结构参数汇总

| 参数 | Actor网络 | Critic网络 |
|------|----------|-----------|
| 隐藏层维度 | [256, 128, 64] | [256, 128, 64] |
| 激活函数 | ReLU | Tanh |
| 输出维度 | 3 + 3 (μ和log_σ) | 1 |
| 权重初始化 | 正交初始化 | 正交初始化 |
| 参数数量（M=20） | ~42,502 | ~42,497 |

---

## 三、超参数设置

### 3.1 核心超参数

| 参数 | 符号 | 值 | 说明 |
|------|------|-----|------|
| 学习率 | $\alpha$ | $3 \times 10^{-4}$ | Adam优化器学习率 |
| 折扣因子 | $\gamma$ | 0.99 | 未来奖励折扣系数 |
| GAE参数 | $\lambda$ | 0.95 | 优势估计平滑系数 |
| 裁剪参数 | $\epsilon$ | 0.2 | 策略更新裁剪范围 |
| 批次大小 | $B$ | 256 | 每次更新的样本数 |
| 训练轮次 | $N_{epochs}$ | 1000 | 总训练epoch数 |
| 每轮更新次数 | $K$ | 4 | 每个epoch对数据的重复使用次数 |
| 经验池大小 | $N_{buffer}$ | 10000 | 经验回放缓冲区容量 |

### 3.2 Adam优化器参数

$$
\theta_{t+1} = \theta_t - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon_{adam}}
$$

| 参数 | 值 | 说明 |
|------|-----|------|
| $\alpha$ | $3 \times 10^{-4}$ | 学习率 |
| $\beta_1$ | 0.9 | 一阶矩估计衰减率 |
| $\beta_2$ | 0.999 | 二阶矩估计衰减率 |
| $\epsilon_{adam}$ | $10^{-8}$ | 数值稳定性常数 |

### 3.3 学习率衰减策略

采用余弦退火（Cosine Annealing）策略：

$$
\alpha_t = \alpha_{min} + \frac{1}{2}(\alpha_{max} - \alpha_{min})\left(1 + \cos\left(\frac{t}{T_{max}}\pi\right)\right)
$$

参数设置：
- $\alpha_{max} = 3 \times 10^{-4}$：初始学习率
- $\alpha_{min} = 1 \times 10^{-5}$：最小学习率
- $T_{max} = 1000$：总训练轮次

### 3.4 梯度裁剪

采用梯度范数裁剪（Gradient Norm Clipping）：

$$
\hat{g} = g \cdot \frac{c}{\max(\|g\|, c)}
$$

其中：
- $g$：原始梯度
- $c = 0.5$：最大梯度范数阈值
- $\hat{g}$：裁剪后的梯度

### 3.5 经验池配置

| 参数 | 值 | 说明 |
|------|-----|------|
| 缓冲区容量 | 10000 | 最大存储经验数 |
| 采样方式 | 优先级采样 | 基于TD误差的优先级 |
| 优先级指数 | $\alpha_{priority} = 0.6$ | 控制优先化程度 |
| 重要性采样指数 | $\beta_{priority} = 0.4$ | 初始偏差修正系数 |
| $\beta$衰减策略 | 线性增长到1.0 | 训练过程中逐步增大 |

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
w_i = \left(\frac{1}{N \cdot P(i)}\right)^{\beta_{priority}}
$$

归一化：

$$
\hat{w}_i = \frac{w_i}{\max_j w_j}
$$

---

## 四、PPO目标函数

### 4.1 裁剪目标函数

PPO的核心是裁剪目标函数，通过限制策略更新幅度保证训练稳定性：

$$
L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]
$$

其中概率比为：

$$
r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{old}}(a_t | s_t)}
$$

**裁剪机制详解**：

当 $\hat{A}_t > 0$（动作优于平均）：

$$
L = \min(r_t, 1+\epsilon) \cdot \hat{A}_t
$$

限制策略不要过度增加好动作的概率。

当 $\hat{A}_t < 0$（动作劣于平均）：

$$
L = \min(r_t, 1-\epsilon) \cdot \hat{A}_t
$$

（注意 $\hat{A}_t < 0$，取最小值实际上是限制策略不要过度减少差动作的概率）

**图示说明**：

```
L^CLIP
    ^
    |      /|
    |     / |  (A > 0)
    |    /  |
    |   /   |
    |  /    |
----+-/-----+-----> r_t
    |/      |
    |       |
    |  (A<0)|
    |       |
```

### 4.2 价值函数损失

Critic网络的损失函数采用裁剪的价值估计：

$$
L^{VF}(\phi) = \frac{1}{2} \mathbb{E}_t \left[ \max\left( (V_\phi(s_t) - V_t^{target})^2, \; (V_{clip}(s_t) - V_t^{target})^2 \right) \right]
$$

其中裁剪后的价值估计为：

$$
V_{clip}(s_t) = V_{\phi_{old}}(s_t) + \text{clip}(V_\phi(s_t) - V_{\phi_{old}}(s_t), -\epsilon, \epsilon)
$$

目标价值：

$$
V_t^{target} = \hat{A}_t + V_{\phi_{old}}(s_t)
$$

### 4.3 熵正则化

添加熵正则化项鼓励探索：

$$
H[\pi_\theta](s) = -\mathbb{E}_{a \sim \pi_\theta} [\log \pi_\theta(a|s)]
$$

对于高斯策略：

$$
H = \frac{1}{2} \log(2\pi e \sigma^2) = \frac{1}{2} + \frac{1}{2}\log(2\pi) + \log\sigma
$$

### 4.4 总损失函数

PPO的总损失函数为三项加权和：

$$
L(\theta, \phi) = -L^{CLIP}(\theta) + c_1 L^{VF}(\phi) - c_2 H[\pi_\theta]
$$

参数设置：

| 参数 | 值 | 说明 |
|------|-----|------|
| $c_1$ | 0.5 | 价值损失系数 |
| $c_2$ | 0.01 | 熵正则化系数 |

---

## 五、训练策略

### 5.1 经验收集

每个训练epoch开始时，使用当前策略与环境交互收集经验：

**收集流程**：

```
算法：经验收集
输入：当前策略 π_θ，环境 env
输出：经验批次 {(s_t, a_t, r_t, s_{t+1}, log_π_θ(a_t|s_t), V_φ(s_t))}

1. 初始化状态 s_0 = env.reset()
2. for t = 0, 1, ..., T-1 do
3.     计算动作分布：μ, σ = Actor(s_t)
4.     采样动作：a_t ~ N(μ, σ²)
5.     计算对数概率：log_π_t = log π_θ(a_t|s_t)
6.     计算状态价值：V_t = Critic(s_t)
7.     执行动作：s_{t+1}, r_t, done = env.step(a_t)
8.     存储经验：buffer.store(s_t, a_t, r_t, s_{t+1}, log_π_t, V_t)
9.     if done then
10.        s_{t+1} = env.reset()
11.    end if
12. end for
13. 计算GAE优势估计和回报
14. 返回经验批次
```

**经验池更新策略**：

- 采用滑动窗口策略，新经验替换最旧的经验
- 维持经验池大小为 $N_{buffer} = 10000$
- 每个epoch收集 $T = 2048$ 步经验

### 5.2 优势估计

采用GAE（Generalized Advantage Estimation）计算优势函数：

**算法**：

```
算法：GAE优势估计
输入：状态序列 {s_0, ..., s_T}，奖励序列 {r_0, ..., r_{T-1}}，
      价值序列 {V_0, ..., V_T}，参数 γ, λ
输出：优势估计 {A_0, ..., A_{T-1}}，回报 {R_0, ..., R_{T-1}}

1. 初始化优势 A = 0
2. for t = T-1, T-2, ..., 0 do
3.     if t == T-1 then
4.         V_{T} = V(s_T)  // 终止状态价值
5.     end if
6.     δ_t = r_t + γ * V_{t+1} - V_t  // TD误差
7.     A_t = δ_t + γ * λ * A  // GAE优势
8.     R_t = A_t + V_t  // 回报
9.     A = A_t  // 递推
10. end for
11. 返回 {A_0, ..., A_{T-1}}, {R_0, ..., R_{T-1}}
```

**数学表达**：

$$
\begin{aligned}
\delta_t &= r_t + \gamma V(s_{t+1}) - V(s_t) \\
\hat{A}_t &= \delta_t + \gamma\lambda \hat{A}_{t+1} \\
R_t &= \hat{A}_t + V(s_t)
\end{aligned}
$$

### 5.3 策略更新

对收集的经验进行多轮更新，采用mini-batch梯度下降：

**算法**：

```
算法：PPO策略更新
输入：经验批次 {(s_t, a_t, A_t, R_t, log_π_old_t)}，更新轮次 K
输出：更新后的策略参数 θ，价值参数 φ

1. 计算优势归一化：A = (A - mean(A)) / (std(A) + ε)
2. for k = 1, 2, ..., K do
3.     将经验随机分成 mini-batches
4.     for each mini-batch B do
5.         // Actor更新
6.         计算新策略概率：log_π_new = log π_θ(a|s)
7.         计算概率比：r = exp(log_π_new - log_π_old)
8.         计算裁剪目标：
9.             surr1 = r * A
10.            surr2 = clip(r, 1-ε, 1+ε) * A
11.            L_CLIP = -min(surr1, surr2).mean()
12.        计算熵：H = -π log π
13.        Actor损失 = L_CLIP - c2 * H
14.        梯度更新：θ = θ - α * ∇(Actor损失)
15.
16.        // Critic更新
17.        计算价值预测：V_pred = Critic(s)
18.        计算裁剪价值：
19.            V_clip = V_old + clip(V_pred - V_old, -ε, ε)
20.            L_VF = max((V_pred - R)², (V_clip - R)²).mean()
21.        Critic损失 = c1 * L_VF
22.        梯度更新：φ = φ - α * ∇(Critic损失)
23.     end for
24. end for
```

### 5.4 优先级采样

基于TD误差的优先级经验回放：

**优先级更新**：

$$
p_i = |\delta_i| + \epsilon_{priority}
$$

其中 $\epsilon_{priority} = 10^{-6}$ 保证所有经验都有被采样的机会。

**采样概率**：

$$
P(i) = \frac{p_i^{\alpha}}{\sum_k p_k^{\alpha}}
$$

**重要性采样权重**：

$$
w_i = \left(\frac{1}{N \cdot P(i)}\right)^{\beta}
$$

$\beta$ 从0.4线性增长到1.0，逐步消除采样偏差。

---

## 六、训练循环

### 6.1 主训练循环

```
算法：PPO主训练循环
输入：环境 env，网络参数 θ, φ，超参数集合
输出：训练后的策略 π_θ*

1. 初始化：
    - 创建Actor网络，参数 θ
    - 创建Critic网络，参数 φ
    - 创建经验回放缓冲区 D，容量 N_buffer = 10000
    - 创建Adam优化器，学习率 α = 3e-4
    - 初始化学习率调度器

2. for epoch = 1, 2, ..., N_epochs do
3.
4.     // 阶段1：经验收集
5.     初始化经验列表 trajectories = []
6.     for episode = 1, 2, ..., N_episodes_per_epoch do
7.         s = env.reset()
8.         episode_reward = 0
9.         for t = 1, 2, ..., T_max do
10.            // Actor推理
11.            μ, σ = Actor(s)
12.            a = sample_action(μ, σ)
13.            log_π = compute_log_prob(a, μ, σ)
14.            V = Critic(s)
15.
16.            // 环境交互
17.            s', r, done = env.step(a)
18.            episode_reward += r
19.
20.            // 存储经验
21.            trajectories.append((s, a, r, s', log_π, V, done))
22.            buffer.push((s, a, r, s', log_π, V, done))
23.
24.            s = s'
25.            if done then
26.                break
27.            end if
28.        end for
29.        记录episode_reward
30.    end for
31.
32.    // 阶段2：GAE计算
33.    计算GAE优势估计 {A_t} 和回报 {R_t}
34.    归一化优势：A = (A - mean) / (std + ε)
35.
36.    // 阶段3：策略更新
37.    for k = 1, 2, ..., K do
38.        从buffer中优先采样 mini-batches
39.        for each batch B do
40.            // 计算Actor损失
41.            L_clip = compute_clip_loss(B)
42.            entropy = compute_entropy(B)
43.            actor_loss = L_clip - c2 * entropy
44.
45.            // 计算Critic损失
46.            L_vf = compute_value_loss(B)
47.            critic_loss = c1 * L_vf
48.
49.            // 梯度裁剪与更新
50.            optimizer.zero_grad()
51.            total_loss = actor_loss + critic_loss
52.            total_loss.backward()
53.            clip_grad_norm_(parameters, max_norm=0.5)
54.            optimizer.step()
55.
56.            // 更新优先级
57.            update_priorities(B, new_td_errors)
58.        end for
59.    end for
60.
61.    // 阶段4：学习率更新
62.    scheduler.step()
63.
64.    // 阶段5：日志记录
65.    记录训练指标：
66.        - 平均episode回报
67.        - 策略熵
68.        - 价值损失
69.        - 策略损失
70.        - 概率比均值和标准差
71.        - 学习率
72.
73.    // 阶段6：模型保存
74.    if epoch % save_interval == 0 then
75.        保存模型检查点
76.    end if
77.
78.    // 阶段7：早停检查
79.    if 满足早停条件 then
80.        终止训练
81.    end if
82.
83. end for
84.
85. 返回训练后的策略 π_θ*
```

### 6.2 训练阶段时序

```
Epoch 1                     Epoch 2                     Epoch 3
    |                           |                           |
    v                           v                           v
[经验收集]                  [经验收集]                  [经验收集]
    |                           |                           |
    v                           v                           v
[GAE计算]                   [GAE计算]                   [GAE计算]
    |                           |                           |
    v                           v                           v
[策略更新 x4]               [策略更新 x4]               [策略更新 x4]
    |                           |                           |
    v                           v                           v
[日志记录]                  [日志记录]                  [日志记录]
```

### 6.3 经验池管理

**滑动窗口策略**：

```python
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, experience):
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, priorities=None):
        if priorities is not None:
            # 优先级采样
            probabilities = priorities / priorities.sum()
            indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        else:
            # 均匀采样
            indices = np.random.choice(len(self.buffer), batch_size)
        return [self.buffer[i] for i in indices]
```

---

## 七、收敛性分析

### 7.1 收敛性指标

监控以下指标判断训练是否收敛：

| 指标 | 符号 | 计算方式 | 收敛标准 |
|------|------|---------|---------|
| 平均回报 | $\bar{R}$ | 最近100个episode回报均值 | 连续50个epoch变化 < 1% |
| 成功率 | $P_{success}$ | 到达目标的episode比例 | > 80% |
| 策略熵 | $H$ | 动作分布的熵 | 稳定在0.5-2.0 |
| 价值损失 | $L_{VF}$ | Critic网络损失 | < 0.01 |
| 概率比 | $r$ | 新旧策略概率比 | 均值接近1.0，标准差 < 0.5 |

### 7.2 学习曲线分析

**理想学习曲线特征**：

```
回报 R
    ^
    |                    ___________
    |                   /
    |                  /
    |                 /
    |                /
    |               /
    |         _____/
    |        /
    |   ____/
    |  /
    | /
    +-----------------------------> 训练轮次
       探索期   学习期   收敛期
```

**阶段划分**：

1. **探索期**（Epoch 0-100）：回报波动大，策略熵高
2. **学习期**（Epoch 100-500）：回报快速上升，策略熵下降
3. **收敛期**（Epoch 500+）：回报稳定，策略熵稳定

### 7.3 收敛性监控算法

```python
class ConvergenceMonitor:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.rewards_history = []
        self.success_history = []

    def update(self, episode_reward, success):
        self.rewards_history.append(episode_reward)
        self.success_history.append(success)

    def check_convergence(self):
        if len(self.rewards_history) < self.window_size:
            return False, {}

        recent_rewards = self.rewards_history[-self.window_size:]
        recent_success = self.success_history[-self.window_size:]

        # 计算指标
        mean_reward = np.mean(recent_rewards)
        std_reward = np.std(recent_rewards)
        success_rate = np.mean(recent_success)

        # 计算变化率
        if len(self.rewards_history) >= 2 * self.window_size:
            prev_rewards = self.rewards_history[-2*self.window_size:-self.window_size]
            prev_mean = np.mean(prev_rewards)
            change_rate = abs(mean_reward - prev_mean) / (abs(prev_mean) + 1e-8)
        else:
            change_rate = float('inf')

        # 收敛判断
        converged = (
            change_rate < 0.01 and  # 回报变化 < 1%
            success_rate > 0.8 and  # 成功率 > 80%
            std_reward / (abs(mean_reward) + 1e-8) < 0.1  # 波动 < 10%
        )

        metrics = {
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'success_rate': success_rate,
            'change_rate': change_rate,
            'converged': converged
        }

        return converged, metrics
```

### 7.4 早停策略

防止过拟合和无效训练：

```python
class EarlyStopping:
    def __init__(self, patience=50, min_delta=0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_reward = -float('inf')

    def check(self, current_reward):
        if current_reward > self.best_reward + self.min_delta:
            self.best_reward = current_reward
            self.counter = 0
            return False  # 不停止
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True  # 停止训练
            return False
```

### 7.5 训练稳定性指标

监控以下指标确保训练稳定：

| 指标 | 健康范围 | 异常处理 |
|------|---------|---------|
| 概率比均值 | [0.8, 1.2] | 超出范围则减小学习率 |
| 概率比标准差 | < 0.5 | 超出则增大裁剪参数 |
| 梯度范数 | < 1.0 | 超出则检查梯度裁剪 |
| 策略熵 | [0.5, 2.0] | 过低则增大熵系数 |
| 价值估计误差 | < 0.1 | 超出则增大价值损失系数 |

---

## 八、奖励函数集成

### 8.1 奖励函数设计

PPO算法与奖励函数的集成采用加权求和结构：

$$
R_t = R_{goal} + P_{collision} + P_{proximity} + P_{formation} + P_{energy} + P_{time} + P_{smooth} + R_{delivery}
$$

各分量定义详见《奖励函数设计文档》。

### 8.2 奖励归一化

为提高训练稳定性，对奖励进行归一化处理：

$$
\hat{R}_t = \frac{R_t - \mu_R}{\sigma_R + \epsilon}
$$

其中 $\mu_R$ 和 $\sigma_R$ 为最近1000步的奖励均值和标准差。

### 8.3 奖励塑形

采用基于势能的奖励塑形，提供密集梯度信号：

$$
F(s, a, s') = \gamma \Phi(s') - \Phi(s)
$$

势函数：

$$
\Phi(s) = -c \cdot \| \mathbf{p}_t - \mathbf{p}_{goal} \|_2
$$

塑形强度系数 $c = 0.5$。

---

## 九、状态空间与动作空间集成

### 9.1 状态编码

PPO算法接收编码后的状态向量：

$$
\mathbf{s}_{encoded} = \begin{bmatrix} \hat{\mathbf{s}}_{cont} \\ \mathbf{o}_{delivery} \end{bmatrix} \in \mathbb{R}^{10 + (M+1)}
$$

其中：
- $\hat{\mathbf{s}}_{cont} \in [0, 1]^{10}$：归一化的连续状态
- $\mathbf{o}_{delivery} \in \{0, 1\}^{M+1}$：配送进度独热编码

状态编码详见《状态空间设计文档》。

### 9.2 动作映射

Actor网络输出的原始动作经过映射和约束处理：

$$
\mathbf{a}_t = \text{ConstraintHandler}(\mathbf{a}_{max} \odot \tanh(\mathbf{u}_t))
$$

其中 $\mathbf{u}_t \sim \mathcal{N}(\mu_t, \sigma_t^2)$ 为网络输出的高斯采样。

约束处理包括：
1. 加速度边界裁剪
2. Jerk限制
3. 速度约束
4. 位置边界

动作空间详见《动作空间设计文档》。

### 9.3 对数概率计算

对于高斯策略，对数概率为：

$$
\log \pi(\mathbf{u}_t | \mathbf{s}_t) = -\frac{1}{2} \sum_{i=1}^{3} \left[ \frac{(u_{i,t} - \mu_{i,t})^2}{\sigma_{i,t}^2} + \log(2\pi\sigma_{i,t}^2) \right]
$$

经过tanh变换的Jacobian修正：

$$
\log \pi(\mathbf{a}_t | \mathbf{s}_t) = \log \pi(\mathbf{u}_t | \mathbf{s}_t) - \sum_{i=1}^{3} \log(1 - \tanh^2(u_{i,t}))
$$

---

## 十、CTDE框架集成

### 10.1 集中训练分散执行

PPO算法在CTDE（Centralized Training with Decentralized Execution）框架下的应用：

**训练阶段**：
- 使用全局状态信息训练集中式Critic
- 各智能体共享经验池，加速学习
- 使用全局奖励信号更新策略

**执行阶段**：
- 各智能体仅使用局部观测做决策
- 无需通信，独立执行
- 符合实际部署场景

### 10.2 多智能体PPO更新

对于N架无人机，采用参数共享的Actor和独立的Critic：

$$
L_{total} = \sum_{i=1}^{N} L^{CLIP}(\theta_i) + c_1 \sum_{i=1}^{N} L^{VF}(\phi_i) - c_2 \sum_{i=1}^{N} H[\pi_{\theta_i}]
$$

共享策略参数：$\theta_1 = \theta_2 = \cdots = \theta_N = \theta$

---

## 十一、实现代码

### 11.1 Actor网络实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class ActorNetwork(nn.Module):
    """PPO Actor网络"""

    def __init__(self, state_dim, action_dim, hidden_dims=[256, 128, 64]):
        super().__init__()

        # 构建隐藏层
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        self.feature_net = nn.Sequential(*layers)

        # 输出层
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

        # 标准差初始化
        nn.init.constant_(self.log_std_head.weight, 0)
        nn.init.constant_(self.log_std_head.bias, -0.5)

    def forward(self, state):
        """前向传播"""
        features = self.feature_net(state)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        std = torch.exp(log_std)
        return mean, std

    def get_action(self, state, deterministic=False):
        """获取动作"""
        mean, std = self.forward(state)
        if deterministic:
            action = mean
        else:
            dist = Normal(mean, std)
            action = dist.sample()
        return action

    def get_log_prob(self, state, action):
        """计算对数概率"""
        mean, std = self.forward(state)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        return log_prob

    def get_entropy(self, state):
        """计算熵"""
        mean, std = self.forward(state)
        dist = Normal(mean, std)
        entropy = dist.entropy().sum(dim=-1)
        return entropy
```

### 11.2 Critic网络实现

```python
class CriticNetwork(nn.Module):
    """PPO Critic网络"""

    def __init__(self, state_dim, hidden_dims=[256, 128, 64]):
        super().__init__()

        # 构建隐藏层
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.Tanh())
            prev_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

        # 初始化
        self._init_weights()

    def _init_weights(self):
        """正交初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.constant_(module.bias, 0)

    def forward(self, state):
        """前向传播"""
        return self.network(state).squeeze(-1)
```

### 11.3 PPO算法实现

```python
class PPO:
    """PPO算法实现"""

    def __init__(self, state_dim, action_dim, config):
        self.config = config

        # 创建网络
        self.actor = ActorNetwork(state_dim, action_dim)
        self.critic = CriticNetwork(state_dim)

        # 创建优化器
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
        """更新策略"""
        states, actions, old_log_probs, advantages, returns = batch

        # 归一化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 计算新的对数概率和熵
        new_log_probs = self.actor.get_log_prob(states, actions)
        entropy = self.actor.get_entropy(states)

        # 计算概率比
        ratio = torch.exp(new_log_probs - old_log_probs)

        # 计算裁剪目标
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        # 计算熵正则化
        entropy_loss = -entropy.mean()

        # 计算价值损失
        values = self.critic(states)
        value_loss = F.mse_loss(values, returns)

        # 总损失
        total_loss = actor_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

        # 梯度更新
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        # 计算指标
        with torch.no_grad():
            approx_kl = (old_log_probs - new_log_probs).mean()
            clip_frac = (torch.abs(ratio - 1) > self.clip_epsilon).float().mean()

        metrics = {
            'actor_loss': actor_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.mean().item(),
            'approx_kl': approx_kl.item(),
            'clip_fraction': clip_frac.item()
        }

        return metrics

    def save(self, path):
        """保存模型"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict()
        }, path)

    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
```

---

## 十二、参数汇总表

### 12.1 网络结构参数

| 参数 | Actor | Critic |
|------|-------|--------|
| 隐藏层维度 | [256, 128, 64] | [256, 128, 64] |
| 激活函数 | ReLU | Tanh |
| 输出维度 | 6 (3+3) | 1 |
| 权重初始化 | 正交(0.01) | 正交(1.0) |

### 12.2 训练超参数

| 参数 | 符号 | 值 |
|------|------|-----|
| 学习率 | $\alpha$ | $3 \times 10^{-4}$ |
| 最小学习率 | $\alpha_{min}$ | $1 \times 10^{-5}$ |
| 折扣因子 | $\gamma$ | 0.99 |
| GAE参数 | $\lambda$ | 0.95 |
| 裁剪参数 | $\epsilon$ | 0.2 |
| 价值损失系数 | $c_1$ | 0.5 |
| 熵系数 | $c_2$ | 0.01 |
| 最大梯度范数 | $c$ | 0.5 |
| 批次大小 | $B$ | 256 |
| 更新轮次 | $K$ | 4 |
| 经验池大小 | $N_{buffer}$ | 10000 |
| 训练轮次 | $N_{epochs}$ | 1000 |

### 12.3 优先级采样参数

| 参数 | 符号 | 值 |
|------|------|-----|
| 优先级指数 | $\alpha_{priority}$ | 0.6 |
| 初始$\beta$ | $\beta_0$ | 0.4 |
| 最终$\beta$ | $\beta_1$ | 1.0 |
| 优先级常数 | $\epsilon_{priority}$ | $10^{-6}$ |

---

## 十三、设计约束与边界条件

### 13.1 数值稳定性

| 问题 | 解决方案 | 说明 |
|------|---------|------|
| 梯度爆炸 | 梯度裁剪(0.5) | 限制梯度范数 |
| 价值估计爆炸 | 价值裁剪 | 限制价值更新幅度 |
| 概率比过大 | 裁剪参数(0.2) | 限制策略更新幅度 |
| 除零错误 | 小常数ε | 各处添加1e-8 |
| 数值溢出 | float32 | 平衡精度与效率 |

### 13.2 训练约束

| 约束 | 值 | 说明 |
|------|-----|------|
| 最大episode长度 | 1000步 | 防止无限循环 |
| 最小batch大小 | 64 | 保证梯度估计稳定 |
| 最大策略更新 | 4轮 | 防止过拟合 |
| 早停耐心值 | 50 epochs | 防止无效训练 |

---

## 十四、与其他模块的接口

### 14.1 与环境模块的接口

```python
# PPO与环境交互
state = env.reset()  # 返回 shape=(11,)
action = ppo.actor.get_action(state)  # 返回 shape=(3,)
next_state, reward, done, info = env.step(action)
```

### 14.2 与状态编码器的接口

```python
# 状态编码后输入网络
encoded_state = encoder.encode(raw_state)
action = ppo.actor.get_action(encoded_state)
value = ppo.critic(encoded_state)
```

### 14.3 与奖励函数的接口

```python
# 奖励计算
reward, reward_info = reward_calculator.calculate(
    positions=positions,
    prev_positions=prev_positions,
    actions=actions,
    prev_actions=prev_actions,
    goal_pos=goal_pos,
    # ... 其他参数
)
```

---

## 十五、总结

### 15.1 设计要点

1. **Actor-Critic架构**：3层全连接网络，Actor输出动作均值和标准差，Critic输出状态价值
2. **PPO裁剪目标**：通过裁剪机制限制策略更新幅度，保证训练稳定性
3. **GAE优势估计**：平衡偏差与方差，提供准确的学习信号
4. **优先级经验回放**：基于TD误差的优先级采样，提高数据利用效率
5. **梯度裁剪与学习率衰减**：防止梯度爆炸，提高训练稳定性

### 15.2 创新点

1. **自适应平滑约束**：根据障碍物距离动态调整动作平滑程度
2. **课程学习集成**：支持分阶段训练，从简单到复杂逐步提升能力
3. **多目标奖励平衡**：8个奖励分量的加权组合，平衡效率、安全和能耗
4. **CTDE框架支持**：支持集中训练分散执行的多智能体场景

### 15.3 后续优化方向

1. **网络结构优化**：探索更深或更宽的网络结构
2. **自适应超参数**：根据训练进度动态调整超参数
3. **分布式训练**：支持多GPU并行训练
4. **在线学习**：支持部署后的在线微调

---

## 附录：参考文献

1. Schulman, J., et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347, 2017.
2. Schulman, J., et al. "High-dimensional continuous control using generalized advantage estimation." ICLR, 2016.
3. Mnih, V., et al. "Asynchronous methods for deep reinforcement learning." ICML, 2016.
4. Lillicrap, T. P., et al. "Continuous control with deep reinforcement learning." ICLR, 2016.
5. Ng, A. Y., et al. "Policy invariance under reward transformations: Theory and application to reward shaping." ICML, 1999.
