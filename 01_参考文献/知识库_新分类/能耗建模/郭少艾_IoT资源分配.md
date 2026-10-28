# 知识卡片：基于深度强化学习的认知物联网资源分配研究

> **来源**：郭少艾，吉林大学博士学位论文，2023年
> **分类**：能耗建模 / 认知物联网 / 深度强化学习资源分配
> **提取日期**：2026-07-13

---

## 第一章 绪论 / Introduction

### 1.1 研究背景

- **5G/6G网络挑战**：频谱资源紧张，IoT设备数量爆炸式增长，传统静态频谱分配效率低下
- **认知无线电（Cognitive Radio, CR）**：通过动态频谱接入（DSA）复用授权频谱，提高频谱效率
  - 三种频谱共享模式：Interweave（交织）、Underlay（底层）、Overlay（覆盖）
  - Underlay模式下次用户可与主用户同时传输，但须满足干扰温度约束
  - Interweave模式下次用户仅在频谱空闲时接入
- **认知物联网（Cognitive IoT）**：认知无线电与物联网的融合，解决大规模IoT设备的频谱和能耗问题
- **能量收集（Energy Harvesting, EH）**：从射频信号中收集能量，解决IoT设备供电问题
  - 射频能量收集（RF EH）：从环境中的无线信号转换为电能
  - 同时无线信息与功率传输（SWIPT）：在传输信息的同时传输能量
- **环境反向散射通信（Ambient Backscatter Communication, ABC）**：被动通信技术，利用环境射频信号传输数据，无需主动射频模块
- **无人机辅助通信（UAV-assisted Communication）**：无人机作为空中基站，灵活部署，增强覆盖

### 1.2 研究动机与意义

- 三大场景的资源分配问题均为NP-hard组合优化问题
- 传统优化方法（凸优化、博弈论）在动态环境下难以实时决策
- 深度强化学习（DRL）适用于动态、不确定环境下的实时决策
- 认知IoT中频谱接入、功率控制、时隙分配等资源分配问题可建模为马尔可夫决策过程（MDP）

### 1.3 研究内容

论文提出三个研究场景：
1. **场景A（第3章）**：具有射频能量收集的认知IoT网络 → Dueling-DQN + DDPG
2. **场景B（第4章）**：具有环境反向散射通信的射频供能认知IoT → DDPG（JCARA算法）
3. **场景C（第5章）**：认知卫星-无人机网络 → MADDPG（JTCPA算法）

### 1.4 主要贡献

- 提出基于Dueling-DQN的多时隙动态功率分配算法（MS-DPA）
- 提出基于DDPG的多时隙联合信道和功率分配算法（MS-CPA）
- 提出基于DDPG的联合信道和资源分配算法（JCARA）
- 提出基于MADDPG的联合轨迹控制和功率分配算法（JTCPA）

---

## 第二章 理论基础 / Theoretical Foundations

### 2.1 强化学习基础（RL）

- **马尔可夫决策过程（MDP）**：四元组 $(S, A, P_{sa}, R)$
  - $S$：状态空间
  - $A$：动作空间
  - $P_{sa}$：状态转移概率
  - $R$：奖励函数
- **贝尔曼方程（Bellman Equation）**：
  - 状态价值函数：$v_\pi(s) = \sum_a \pi(a|s) \sum_{s',r} p(s',r|s,a)[r + \gamma V_\pi(s')]$（式2.7）
  - 动作价值函数：$Q(s,a) = E[r_t + \gamma Q(S_{t+1}, A_{t+1}) | s=S_t, a=A_t]$（式3.10）
  - $\gamma \in [0,1)$：折扣因子
- **Q-learning算法**：基于时序差分（TD）的无模型算法
  - 更新规则：$Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$
  - $\alpha$：学习率
- **策略梯度（Policy Gradient）**：直接优化策略 $\pi_\theta(a|s)$
  - 目标函数：$J(\theta) = E_{\pi_\theta}[\sum_{t=0}^{T} \gamma^t r_t]$
  - 梯度：$\nabla_\theta J(\theta) = E_{\pi_\theta}[\nabla_\theta \log \pi_\theta(a|s) \cdot Q^{\pi}(s,a)]$

### 2.2 深度强化学习（DRL）

- **DQN（Deep Q-Network）**：用神经网络近似Q函数
  - 经验回放（Experience Replay）：存储 $(s, a, r, s')$ 到回放缓冲区，随机采样打破相关性
  - 目标网络（Target Network）：定期复制参数，稳定训练
  - 损失函数：$L(\theta) = E[(r + \gamma \max_{a'} Q(s',a';\theta^-) - Q(s,a;\theta))^2]$
- **Dueling-DQN**：分离状态价值 $V(s)$ 和优势函数 $A(s,a)$
  - $Q(s,a;\theta,\alpha,\beta) = V(s;\theta,\beta) + A(s,a;\theta,\alpha) - \frac{1}{|A|}\sum_{a'}A(s,a';\theta,\alpha)$（式3.13）
  - 优势：在状态价值高但动作差异小的场景中更高效
- **DDPG（Deep Deterministic Policy Gradient）**：连续动作空间的Actor-Critic算法
  - Actor网络 $\mu(s;\theta^\mu)$：输出确定性动作
  - Critic网络 $Q(s,a;\theta^Q)$：评估动作价值
  - 使用软更新（Soft Update）：$\theta' \leftarrow \tau\theta + (1-\tau)\theta'$
  - 使用Ornstein-Uhlenbeck噪声探索

### 2.3 多智能体强化学习（MARL）

- **马尔可夫博弈（Markov Game）**：扩展MDP到多智能体
  - 元组 $(N, S, \{A_i\}, \{R_i\}, T)$
  - $N$：智能体数量
  - $\{A_i\}$：各智能体的动作空间
  - $\{R_i\}$：各智能体的奖励函数
- **纳什均衡（Nash Equilibrium）**：各智能体策略互相最优响应
  - $\pi_i^* = \arg\max_{\pi_i} V_i(\pi_i, \pi_{-i}^*)$
- **MADDPG（Multi-Agent DDPG）**：集中训练分布式执行（CTDE）
  - 训练时Critic可访问所有智能体的观测和动作
  - 执行时每个Actor仅使用本地观测
  - 每个智能体 $i$ 有独立的Actor $\mu_i$ 和Critic $Q_i$
  - Critic输入：所有智能体的观测和动作 $Q_i(x, a_1, ..., a_N)$
- **部分可观测马尔可夫决策过程（POMDP）**：
  - 七元组 $(S, A, P_{sa}, R, \Omega, O, \gamma)$
  - $\Omega$：观测空间
  - $O$：观测概率函数
  - 实际中通过RNN/LSTM处理观测历史

---

## 第三章 具有射频能量收集的认知物联网 / RF Energy Harvesting Cognitive IoT

### 3.1 系统模型

- **网络拓扑**：一个主基站（PBS）、$N$ 个次用户（SU/认知IoT设备）、$M$ 个主用户（PU）
- **时隙结构**：每个时隙 $T$ 分为两阶段：
  - 能量收集阶段（持续时间 $\tau$）：SU从RF信号收集能量
  - 数据传输阶段（持续时间 $T-\tau$）：SU使用收集的能量传输数据
- **频谱共享**：Underlay模式，SU与PU共享频谱

### 3.2 能量收集模型

- **收集能量**：$E_i(t) = \eta(1-M_i(t))(T-\tau)[P_T(t)g_{pi}(t) + \sum_{j \neq i} P_j(t)g_{ji}(t)]$（式3.4）
  - $\eta$：能量转换效率
  - $M_i(t) \in \{0,1\}$：模式选择（0=收集，1=传输）
  - $P_T(t)$：主基站发射功率
  - $g_{pi}(t)$：PBS到SU的信道增益
  - $P_j(t)$：其他SU的发射功率
  - $g_{ji}(t)$：SU $j$ 到SU $i$ 的信道增益
- **电池约束**：$0 \leq B_i(t) \leq B_{\max}$

### 3.3 传输模型与吞吐量

- **香农吞吐量**：$r_t = \sum_{i=1}^{N} \frac{(T-\tau)M_i(t)W \log_2(1 + P_i(t)g_{ib}(t)/\sigma^2)}{T}$（式3.7）
  - $W$：带宽
  - $P_i(t)$：SU发射功率
  - $g_{ib}(t)$：SU到PBS的信道增益
  - $\sigma^2$：噪声功率
- **干扰约束**：$\sum_{i=1}^{N} M_i(t) P_i(t) g_{ip}(t) \leq I_{th}$（对主用户的干扰不超过门限）

### 3.4 优化问题 P1

- **目标**：最大化系统长期平均吞吐量
- **公式**：$\max_{M_i(t), P_i(t), \tau} \lim_{T_0 \to \infty} \frac{1}{T_0} \sum_{t=1}^{T_0} r_t$
- **约束**：
  - 电池容量约束
  - 功率非负约束
  - 干扰温度约束
  - 模式选择二元约束
- **挑战**：混合整数非线性规划（MINLP），NP-hard

### 3.5 算法一：MS-DPA（基于Dueling-DQN的多时隙动态功率分配）

- **状态空间**：$s_t = [h_{ib}(t), h_{ip}(t), B_i(t), P_T(t)]$（信道增益、电池状态、主基站功率）
- **动作空间**：离散化功率等级 $\{P_1, P_2, ..., P_K\}$ 和模式选择 $\{0,1\}$
- **奖励函数**：$r_t = r_t^{throughput} - \lambda \cdot I_{violation}$
  - 吞吐量奖励 + 干扰违反惩罚
- **网络结构**：Dueling-DQN分离 $V(s)$ 和 $A(s,a)$
- **关键设计**：
  - 优先经验回放（Prioritized Experience Replay）
  - $\epsilon$-greedy探索，$\epsilon$ 从1.0线性衰减到0.01

### 3.6 算法二：MS-CPA（基于DDPG的多时隙联合信道和功率分配）

- **状态空间**：连续信道增益、电池状态
- **动作空间**：连续功率值 + 离散信道选择（通过Gumbel-Softmax松弛）
- **奖励函数**：与MS-DPA相同
- **Actor-Critic结构**：
  - Actor：输出连续功率分配策略
  - Critic：评估状态-动作价值
- **关键设计**：联合优化信道分配和功率控制

### 3.7 仿真结果

- **参数设置**：$W=1$MHz，$\sigma^2=10^{-5}$W，$B_{\max}=20\sim200$mJ，$P_p=0.2\sim2$W，$N=10\sim15$，$L=2\sim20$，$T=1$s
- **MS-DPA性能**：
  - 相比Q-learning提升约15-20%吞吐量
  - Dueling结构在大规模动作空间中收敛更快
  - 优先经验回放加速训练约30%
- **MS-CPA性能**：
  - 连续动作空间下吞吐量优于离散化方案
  - 联合优化比分离优化提升约10-15%
- **收敛性**：约2000-3000个episode后收敛

---

## 第四章 环境反向散射通信的射频供能认知物联网 / ABC+RF-Powered Cognitive IoT

### 4.1 系统模型

- **网络拓扑**：一个主基站（PBS）、一个反向散射设备（BD）、$N$ 个射频供能认知IoT设备
- **三类节点**：
  - PBS：主用户发射机
  - BD：无源反向散射设备，利用PBS信号进行被动通信
  - 认知IoT设备：从RF信号收集能量，使用Underlay/Interweave模式传输
- **NOMA（非正交多址接入）**：多个用户共享同一资源，通过SIC（串行干扰消除）解码

### 4.2 ABC信号模型

- **反向散射信号**：$x_i(t) = \eta_i(t)\sqrt{\alpha_i(t)g_{pi}(t)} x_p(t) c_i(t)$（式4.5）
  - $\eta_i(t)$：反射系数
  - $\alpha_i(t)$：功率分裂比
  - $g_{pi}(t)$：PBS到BD的信道增益
  - $x_p(t)$：PBS发射的信号
  - $c_i(t)$：BD的调制信号（0或1）
- **SIC解码顺序**：信道条件好的先解码，消除其干扰后再解码弱信号

### 4.3 Underlay-Interweave混合模型

- **Underlay模式**：SU与PU同时传输，需满足干扰约束
- **Interweave模式**：仅在检测到频谱空闲时传输
- **混合模式**：部分SU用Underlay，部分用Interweave
- **模式选择**：由智能体动态决策

### 4.4 优化问题 P1-P4

- **P1**：最大化系统吞吐量（Underlay-Interweave + ABC）
- **P2**：最大化能量效率（bits/Joule）
- **P3**：最大化公平性（Jain's fairness index）
- **P4**：多目标优化（吞吐量-能效-公平性权衡）
- **约束条件**：
  - 功率约束
  - 干扰温度约束
  - 电池容量约束
  - 反向散射反射系数约束

### 4.5 算法三：JCARA（基于DDPG的联合信道和资源分配）

- **状态空间**：$s_t = [g_{ij}(t), B_i(t), \alpha_i(t), \eta_i(t)]$
- **动作空间**：连续功率 $P_i(t)$、功率分裂比 $\alpha_i(t)$、反射系数 $\eta_i(t)$
- **奖励函数**：$r_t = r_t^{throughput} - \lambda_1 I_{violation} - \lambda_2 E_{violation}$
- **关键设计**：
  - 三层神经网络（256-128-64）
  - Gumbel-Softmax处理混合离散-连续动作
  - 自适应学习率
  - 梯度裁剪（Gradient Clipping）

### 4.6 仿真结果

- **参数设置**：$W=1$MHz，$P_p=2\sim20$W，$P_b=10$W，$\varepsilon=0.5$，$\eta=0.9$，$N=3\sim10$，$I_{th}=2\times10^{-4}$W
- **JCARA性能**：
  - 相比随机接入提升约40-60%吞吐量
  - ABC设备的存在使系统吞吐量提升约20%
  - NOMA比TDMA提升约25-35%频谱效率
  - Underlay-Interweave混合模式比单一模式提升约15%
- **能效优化**：P2场景下能量效率提升约50%（相比吞吐量最大化方案）

---

## 第五章 认知卫星-无人机网络 / Cognitive Satellite-UAV Network

### 5.1 系统模型

- **网络拓扑**：
  - 卫星（Satellite）：提供大范围覆盖
  - $K$ 架无人机（UAV）：作为空中移动基站，$K=3$（仿真中）
  - $N$ 个地面认知IoT用户
  - 主用户（PU）网络
- **NOMA下行链路**：UAV使用NOMA同时服务多个IoT设备
- **UAV移动性**：UAV可调整飞行轨迹和高度

### 5.2 信道模型

- **UAV-地面信道**：视距（LoS）+ 非视距（NLoS）概率模型
  - $P_{LoS} = \frac{1}{1 + a \exp(-b(\theta - a))}$
  - $\theta$：仰角
  - $a, b$：环境参数
- **自由空间路径损耗**：$L_{FS} = 20\log_{10}(d) + 20\log_{10}(f) + 20\log_{10}(4\pi/c)$

### 5.3 UAV轨迹模型

- **轨迹更新**：$x_u(t) = x_u(t-1) + \tau v_u(t) \cos(\theta_u(t))$（式5.1）
- **速度约束**：$0 \leq v_u(t) \leq v_{\max}$
- **高度约束**：$H_{\min} \leq H_u \leq H_{\max}$
- **碰撞避免**：$\|x_u(t) - x_j(t)\| \geq d_{\min}, \forall u \neq j$

### 5.4 优化问题 OP1/OP2

- **OP1**：最大化最小用户速率（Max-Min Fairness）
  - $\max_{x_u, P_i} \min_i R_i$
  - 联合优化UAV轨迹和发射功率
- **OP2**：最大化系统总吞吐量
  - $\max_{x_u, P_i} \sum_i R_i$
- **约束条件**：
  - UAV轨迹约束（速度、高度、碰撞避免）
  - 功率约束
  - 干扰约束
  - 数据速率需求约束

### 5.5 算法四：JTCPA（基于MADDPG的联合轨迹控制和功率分配）

- **POMDP建模**：
  - 每个UAV为一个智能体
  - 观测空间：本地信道状态、位置、其他UAV的部分信息
  - 状态转移由环境动态决定（信道衰落、用户移动）
- **MADDPG框架**：
  - 每个UAV有独立的Actor和Critic
  - Critic集中训练：可访问所有UAV的观测和动作
  - Actor分布式执行：仅使用本地观测
- **动作空间**：连续动作 $(v_u, \theta_u, P_i)$（速度、方向角、功率）
- **奖励函数**：
  - OP1：$r_t = \min_i R_i - \lambda_{collision} \cdot \mathbb{1}_{collision}$
  - OP2：$r_t = \sum_i R_i - \lambda_{collision} \cdot \mathbb{1}_{collision}$
- **关键设计**：
  - 多智能体协作：通过集中Critic实现隐式协调
  - 噪声探索：使用Ornstein-Uhlenbeck过程
  - 经验回放缓冲区大小：$10^6$
  - 软更新参数 $\tau = 0.01$

### 5.6 仿真结果

- **参数设置**：$v_{\max}=30$m/s，$P_{\max}=5$W，$H_u=100$m，$B=4$MHz，$\tau=1$s，$T=40\sim180$s，$D=[0.5,2]$Mbit
- **JTCPA性能**：
  - 3架UAV协作比单UAV提升约80-120%系统吞吐量
  - MADDPG比独立DDPG提升约20-30%（多智能体协调优势）
  - 轨迹优化比固定悬停提升约50-70%
  - 约5000个episode后收敛
- **轨迹分析**：
  - UAV倾向于飞向用户密集区域
  - 碰撞避免约束有效保证UAV安全间距
  - 较长的任务时间（$T$增大）允许更优轨迹规划

---

## 第六章 总结与展望 / Summary and Future Work

### 6.1 主要结论

1. **Dueling-DQN**适合离散动作空间下的频谱接入决策，分离状态价值和优势函数提高学习效率
2. **DDPG**适合连续动作空间下的功率控制，联合优化比分离优化更优
3. **MADDPG**适合多UAV协作场景，集中训练分布式执行框架有效解决非平稳性问题
4. **ABC技术**可显著提升认知IoT系统的能量效率和吞吐量
5. **NOMA+SIC**比传统TDMA/FDMA提升25-35%频谱效率

### 6.2 未来研究方向

1. **大规模MIMO+DRL**：天线阵列下的波束成形资源分配
2. **联邦学习+DRL**：分布式隐私保护训练
3. **元学习（Meta-Learning）**：快速适应新环境
4. **安全DRL**：对抗攻击下的鲁棒决策
5. **数字孪生+DRL**：仿真到真实的迁移
6. **6G网络**：太赫兹频段、智能超表面（RIS）辅助通信
7. **多目标优化**：帕累托最优的资源分配策略

---

## 关键公式速查 / Key Formulas

| 编号 | 公式 | 说明 |
|------|------|------|
| 2.7 | $v_\pi(s) = \sum_a \pi(a\|s) \sum_{s',r} p(s',r\|s,a)[r + \gamma V_\pi(s')]$ | 贝尔曼方程 |
| 3.4 | $E_i(t) = \eta(1-M_i(t))(T-\tau)[P_T(t)g_{pi}(t) + \sum_j P_j(t)g_{ji}(t)]$ | 射频能量收集 |
| 3.7 | $r_t = \sum_i \frac{(T-\tau)M_i(t)W\log_2(1+P_i(t)g_{ib}(t)/\sigma^2)}{T}$ | 香农吞吐量 |
| 3.10 | $Q(s,a) = E[r_t + \gamma Q(S_{t+1}, A_{t+1}) \| s=S_t, a=A_t]$ | Q函数定义 |
| 3.13 | $Q(s,a) = V(s;\theta,\beta) + A(s,a;\theta,\alpha) - \frac{1}{\|A\|}\sum_{a'}A(s,a';\theta,\alpha)$ | Dueling-DQN |
| 4.5 | $x_i(t) = \eta_i(t)\sqrt{\alpha_i(t)g_{pi}(t)} x_p(t) c_i(t)$ | ABC信号模型 |
| 5.1 | $x_u(t) = x_u(t-1) + \tau v_u(t)\cos(\theta_u(t))$ | UAV轨迹更新 |

## 算法对比速查 / Algorithm Comparison

| 算法 | 场景 | 动作空间 | 核心技术 | 关键优势 |
|------|------|----------|----------|----------|
| MS-DPA | 射频EH认知IoT | 离散 | Dueling-DQN + 优先经验回放 | 大规模离散动作空间高效学习 |
| MS-CPA | 射频EH认知IoT | 连续 | DDPG + Gumbel-Softmax | 联合信道和功率分配 |
| JCARA | ABC+RF供能认知IoT | 混合 | DDPG + 自适应学习率 | 多维资源联合优化 |
| JTCPA | 卫星-UAV网络 | 连续 | MADDPG + POMDP | 多UAV协作轨迹和功率优化 |

## 参数表速查 / Parameter Tables

### 场景A（第3章）参数

| 参数 | 含义 | 取值 |
|------|------|------|
| $W$ | 带宽 | 1 MHz |
| $\sigma^2$ | 噪声功率 | $10^{-5}$ W |
| $B_{\max}$ | 最大电池容量 | 20-200 mJ |
| $P_p$ | 主基站功率 | 0.2-2 W |
| $N$ | 次用户数 | 10-15 |
| $L$ | 主用户数 | 2-20 |
| $T$ | 时隙长度 | 1 s |
| $\eta$ | 能量转换效率 | 0.5-0.9 |

### 场景B（第4章）参数

| 参数 | 含义 | 取值 |
|------|------|------|
| $W$ | 带宽 | 1 MHz |
| $P_p$ | PBS发射功率 | 2-20 W |
| $P_b$ | 反向散射设备功率 | 10 W |
| $\varepsilon$ | 功率分裂比 | 0.5 |
| $\eta$ | 反射系数 | 0.9 |
| $N$ | IoT设备数 | 3-10 |
| $I_{th}$ | 干扰门限 | $2\times10^{-4}$ W |

### 场景C（第5章）参数

| 参数 | 含义 | 取值 |
|------|------|------|
| $v_{\max}$ | UAV最大速度 | 30 m/s |
| $P_{\max}$ | 最大发射功率 | 5 W |
| $H_u$ | UAV飞行高度 | 100 m |
| $B$ | 系统带宽 | 4 MHz |
| $\tau$ | 时隙长度 | 1 s |
| $T$ | 任务时间 | 40-180 s |
| $D$ | 数据需求 | 0.5-2 Mbit |
| $K$ | UAV数量 | 3 |

---

## 与本项目的关联 / Relevance to UAV Delivery Project

### 直接借鉴点

1. **DRL算法框架**：论文中的Dueling-DQN、DDPG、MADDPG框架可直接用于无人机配送的资源分配
2. **状态空间设计**：将信道增益、电池状态等映射为无人机位置、电量、订单状态
3. **奖励函数设计**：吞吐量+干扰惩罚的复合奖励结构，可映射为配送收益+能耗惩罚
4. **多智能体协调**：MADDPG的CTDE框架适用于多无人机协同配送
5. **轨迹优化**：UAV轨迹优化方法（式5.1）可直接用于配送路径规划
6. **能量建模**：RF能量收集模型可借鉴为无人机电池消耗建模

### 方法论映射

| 论文概念 | → | 无人机配送映射 |
|----------|---|----------------|
| 频谱接入决策 | → | 订单分配决策 |
| 功率控制 | → | 速度/高度控制 |
| 能量收集 | → | 充电站补能 |
| 干扰约束 | → | 禁飞区/碰撞避免 |
| 吞吐量最大化 | → | 配送效率最大化 |
| 多UAV协作 | → | 多无人机协同配送 |

### 差异点

- 论文关注通信资源分配（频谱、功率），本项目关注物流资源分配（路径、订单）
- 论文的信道模型为随机衰落，本项目的环境约束为确定性（建筑、禁飞区）
- 论文使用连续动作空间（DDPG），本项目L1/L2使用离散动作空间（Q-learning/DQN）
- 论文的POMDP建模（部分可观测）可借鉴用于多无人机场景（信息不完全共享）

---

*生成时间：2026-07-13 | 提取工具：pdftotext -layout | 论文总页数：约130页（5697行文本）*
