# 基于深度强化学习的多无人机协同物流配送路径规划 -- 奖励函数设计文档（详细版）

> **说明**：本文档为奖励函数设计的**详细版**，包含完整的数学公式推导（LaTeX 格式）、参数标定详细流程、奖励归一化方法（Running Mean / 自适应权重 / 裁剪）、稀疏奖励处理（势能塑形 / 进度奖励 / ICM 好奇心）、课程学习 4 阶段策略，以及完整的 Python 代码实现（RewardCalculator、RewardNormalizer、CurriculumManager、RewardMonitor、FormationGenerator）。
>
> 如仅需快速查阅各奖励分量的定义和核心参数，请参阅概要版文档：`06_模型与算法\算法文档\奖励函数设计.md`。
>
> 两份文档的核心参数保持一致，如有冲突以本文档（详细版）为准。

- **版本**：v2.0
- **日期**：2026-07-01
- **状态**：完整设计
- **作者**：算法设计Agent

---

## 1. 总体设计思路

### 1.1 设计目标

奖励函数是连接任务目标与智能体行为的核心桥梁。本项目奖励函数设计遵循以下原则：

1. **任务导向**：引导无人机完成配送任务（到达取货点 -> 取货 -> 飞往配送点 -> 投放）
2. **安全约束**：严格惩罚碰撞行为，保持无人机间安全间距
3. **效率优化**：鼓励快速、低能耗完成任务
4. **协同协作**：维持编队队形，避免冲突
5. **可训练性**：通过密集奖励信号和课程学习解决稀疏奖励问题

### 1.2 整体奖励结构

奖励函数采用**加权求和**结构，将8个奖励分量组合为标量奖励信号：

$$
R_t = \underbrace{R_{goal}}_{\text{目标到达}} + \underbrace{P_{collision}}_{\text{碰撞惩罚}} + \underbrace{P_{proximity}}_{\text{间距惩罚}} + \underbrace{P_{formation}}_{\text{队形惩罚}} + \underbrace{P_{energy}}_{\text{能耗惩罚}} + \underbrace{P_{time}}_{\text{时间惩罚}} + \underbrace{P_{smooth}}_{\text{平滑惩罚}} + \underbrace{R_{delivery}}_{\text{配送奖励}}
$$

分量分类：

| 类别 | 分量 | 触发方式 |
|------|------|----------|
| 事件触发型 | $R_{goal}$, $P_{collision}$, $R_{delivery}$ | 条件满足时一次性触发 |
| 连续型（每步） | $P_{proximity}$, $P_{formation}$, $P_{energy}$, $P_{time}$, $P_{smooth}$ | 每步持续计算 |

---

## 2. 各奖励分量定义与数学公式

### 2.1 目标到达奖励 R_goal

**含义**：无人机到达目标点时获得一次性正奖励，鼓励智能体将"到达目标"作为首要任务。

**数学公式**：

$$
R_{goal} =
\begin{cases}
+200, & \text{if } \| \mathbf{p}_t - \mathbf{p}_{goal} \|_2 \leq d_{goal} \\
0, & \text{otherwise}
\end{cases}
$$

**距离引导奖励**（辅助稠密信号）：

$$
R_{dist} = c_{dist} \cdot \left( d_{t-1} - d_t \right)
$$

其中 $d_t = \| \mathbf{p}_t - \mathbf{p}_{goal} \|_2$ 为当前步到目标的距离，$c_{dist} = 1.0$ 为距离奖励系数。

**变量说明**：
- $\mathbf{p}_t \in \mathbb{R}^3$：无人机在时刻 $t$ 的三维位置
- $\mathbf{p}_{goal} \in \mathbb{R}^3$：目标点坐标
- $d_{goal} = 5$ m：目标到达判定距离
- $\| \cdot \|_2$：欧几里得距离

**设计考量**：
- 阈值5米兼顾GPS定位精度和实际降落场地大小
- +200的奖励值确保智能体将"到达目标"作为首要任务
- 仅触发一次，防止智能体在目标点附近反复获取奖励
- 距离引导奖励在每步提供连续梯度信号，解决稀疏奖励问题

---

### 2.2 碰撞惩罚 P_collision

**含义**：发生碰撞时施加一次性大额惩罚，与目标奖励等值反向，确保安全与目标同等重要。

**数学公式**：

$$
P_{collision} =
\begin{cases}
-200, & \text{if collision detected} \\
0, & \text{otherwise}
\end{cases}
$$

**碰撞检测条件**（满足任一即判定碰撞）：

条件1 -- 无人机间碰撞：

$$
\text{collision}_{ij} \iff \| \mathbf{p}_t^{(i)} - \mathbf{p}_t^{(j)} \|_2 < r_{collision}
$$

条件2 -- 无人机与障碍物碰撞：

$$
\text{collision}_{obs} \iff \min_{\mathbf{o} \in \mathcal{O}} \| \mathbf{p}_t - \mathbf{o}_{pos} \|_2 < r_{collision} + r_{obs}
$$

条件3 -- 越界碰撞：

$$
\text{collision}_{bound} \iff \exists k \in \{x,y,z\}: p_t^{(k)} \notin [lb_k, \, ub_k]
$$

**安全距离渐进惩罚**（在碰撞发生前提供预警信号）：

$$
P_{safe} = -\lambda_{safe} \cdot \sum_{j \neq i} \max\left(0, \; \frac{d_{safe\_warn} - d_{ij}}{d_{safe\_warn} - r_{collision}}\right)^2
$$

其中 $d_{safe\_warn} = 3.0$ m 为预警距离，$\lambda_{safe} = 10.0$。

**变量说明**：
- $r_{collision} = 1.0$ m：碰撞判定半径
- $r_{obs}$：障碍物半径
- $\mathcal{O}$：环境中所有障碍物集合
- $[lb_k, ub_k]$：空间边界

**设计考量**：
- 与 $R_{goal}$ 等值反向（+200 vs -200），平衡安全与目标
- 碰撞后可选择终止episode或继续训练（建议课程学习阶段逐步放开）
- 安全距离渐进惩罚提供连续梯度，避免智能体在碰撞边界反复试探

---

### 2.3 间距惩罚 P_proximity

**含义**：当两架无人机距离小于安全距离阈值时，施加连续惩罚，防止多机过于接近。

**数学公式**：

$$
P_{proximity} = -100 \times \sum_{i=1}^{N} \sum_{j=i+1}^{N} \max\left(0, \; 1 - \frac{\| \mathbf{p}_t^{(i)} - \mathbf{p}_t^{(j)} \|_2}{d_{safe}}\right)
$$

其中 $d_{safe} = 10.0$ m。

**惩罚强度随距离变化**：

| 距离 $d_{ij}$ (m) | 惩罚系数 | 单对惩罚值 |
|:---:|:---:|:---:|
| 0 | 1.0 | -100 |
| 2 | 0.8 | -80 |
| 5 | 0.5 | -50 |
| 8 | 0.2 | -20 |
| 10 | 0.0 | 0 |
| >10 | 0.0 | 0 |

**设计考量**：
- 采用线性衰减而非阶跃函数，提供平滑梯度信号
- 惩罚随距离连续变化，避免智能体在阈值边界震荡
- 当 $N=1$（单机）时，此项恒为0
- 与碰撞惩罚形成两级防护：间距惩罚（软约束）-> 碰撞惩罚（硬约束）

---

### 2.4 队形保持惩罚 P_formation

**含义**：惩罚无人机偏离期望队形位置的程度，维持编队飞行的结构化行为。

**数学公式**：

$$
P_{formation} = -\lambda_{form} \times \frac{1}{N} \sum_{i=1}^{N} \| \mathbf{p}_t^{(i)} - \mathbf{p}_{desired}^{(i)} \|_2
$$

其中 $\lambda_{form} = 1.0$。

**期望队形位置计算**：

$$
\mathbf{p}_{desired}^{(i)} = \mathbf{p}_{center} + \mathbf{offset}^{(i)}
$$

$$
\mathbf{p}_{center} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{p}_t^{(i)}
$$

**三种队形偏移定义**：

V字队形（推荐）：

$$
\mathbf{offset}^{(i)} = \left( -|i - \lfloor N/2 \rfloor| \cdot d_x, \; (i - \lfloor N/2 \rfloor) \cdot d_y, \; 0 \right)
$$

直线队形：

$$
\mathbf{offset}^{(i)} = \left( 0, \; (i - \lfloor N/2 \rfloor) \cdot d_y, \; 0 \right)
$$

菱形队形：根据位置索引查表获取偏移量。

**变量说明**：
- $\lambda_{form} = 1.0$：队形偏差惩罚权重
- $N$：编队中无人机数量
- $d_x, d_y$：编队间距参数（建议值 8.0 m）

**设计考量**：
- 归一化处理（除以 $N$）使惩罚不随编队规模变化
- 可根据任务阶段动态调整：巡航阶段强调队形，配送阶段放宽约束
- 当 $N=1$ 时此项恒为0

---

### 2.5 能耗惩罚 P_energy

**含义**：对无人机每步的能量消耗施加惩罚，鼓励节能飞行，延长续航时间。

**数学公式**：

$$
P_{energy} = -\lambda_{energy} \times \frac{1}{N} \sum_{i=1}^{N} E_t^{(i)}
$$

其中 $\lambda_{energy} = 0.5$。

**完整能耗模型**：

$$
E_t^{(i)} = \alpha \cdot \| \mathbf{a}_t^{(i)} \|_2^2 + \beta \cdot \| \mathbf{v}_t^{(i)} \|_2^2 + \gamma \cdot | h_t^{(i)} - h_{t-1}^{(i)} |
$$

**简化版本**（用于快速训练）：

$$
E_t^{(i)} = \| \mathbf{a}_t^{(i)} \|_2^2
$$

**变量说明**：
- $\lambda_{energy} = 0.5$：能耗惩罚权重
- $\alpha = 1.0$：加速度能耗系数
- $\beta = 0.1$：速度维持能耗系数
- $\gamma = 2.0$：高度变化能耗系数（爬升耗能大）
- $\mathbf{a}_t^{(i)} \in \mathbb{R}^3$：无人机 $i$ 在时刻 $t$ 的加速度/推力指令

**自适应能耗系数**（考虑电池电量）：

$$
\lambda_{energy}^{adaptive} = \lambda_{energy} \cdot \left(1 + \eta_{batt} \cdot \frac{E_{max} - E_{remain}}{E_{max}}\right)
$$

其中 $\eta_{batt} = 2.0$，电量越低时能耗惩罚越大，促使无人机尽快完成任务或返航。

**设计考量**：
- 二次方形式使大幅动作受到更重惩罚
- 电量低时增大权重，鼓励返航或快速完成
- 鼓励巡航飞行而非急加速/急减速

---

### 2.6 时间惩罚 P_time

**含义**：每步施加固定时间成本，鼓励高效完成任务，避免无意义的徘徊。

**数学公式**：

$$
P_{time} = -\lambda_{time} = -2
$$

**累积时间惩罚**（episode层面）：

$$
P_{time}^{total} = -\lambda_{time} \times T_{episode} = -2 \times T_{episode}
$$

**超时终止惩罚**：

$$
P_{timeout} =
\begin{cases}
-100, & \text{if } T_{episode} \geq T_{max} \\
0, & \text{otherwise}
\end{cases}
$$

其中 $T_{max} = 1000$ 步为最大episode长度。

**变量说明**：
- $\lambda_{time} = 2$：每步时间惩罚值
- $T_{episode}$：episode总步数

**设计考量**：
- 固定惩罚值使得智能体在最优路径与最短路径间做权衡
- 与能耗惩罚配合：时间惩罚鼓励快速到达，能耗惩罚约束不要过于激进
- 超时终止防止无效探索

---

### 2.7 平滑惩罚 P_smooth

**含义**：惩罚动作的突变，鼓励平稳飞行，减少机械磨损和货物晃动。

**数学公式**：

$$
P_{smooth} = -\lambda_{smooth} \times \frac{1}{N} \sum_{i=1}^{N} \| \mathbf{a}_t^{(i)} - \mathbf{a}_{t-1}^{(i)} \|_2^2
$$

其中 $\lambda_{smooth} = 0.1$。

**加加速度（Jerk）惩罚**（更严格的平滑约束）：

$$
P_{jerk} = -\lambda_{jerk} \times \frac{1}{N} \sum_{i=1}^{N} \| \mathbf{j}_t^{(i)} \|_2^2
$$

其中 $\mathbf{j}_t = (\mathbf{a}_t - \mathbf{a}_{t-1}) / \Delta t$ 为加加速度。

**变量说明**：
- $\lambda_{smooth} = 0.1$：动作变化惩罚权重
- $\mathbf{a}_t^{(i)} - \mathbf{a}_{t-1}^{(i)}$：连续两步的动作差（jerk的离散近似）

**设计考量**：
- 权重较小（0.1），作为辅助约束而非主要信号
- 有助于减少实际飞行中的机械磨损和货物晃动
- 首步（$t=0$）时此项为0

---

### 2.8 配送完成奖励 R_delivery

**含义**：完成单个配送任务时获得奖励，鼓励智能体完成多任务配送。

**数学公式**：

$$
R_{delivery} = \sum_{k=1}^{K} r_k \cdot \mathbb{1}\left[\text{delivery}_k \text{ completed at step } t\right]
$$

**配送完成判定**：

$$
\text{delivery}_k \text{ completed} \iff \| \mathbf{p}_t - \mathbf{p}_{dest}^{(k)} \|_2 \leq d_{drop} \; \land \; \text{cargo}_k = \text{loaded}
$$

**递增奖励策略**（鼓励完成所有任务）：

$$
r_k = R_{delivery} \cdot \left(1 + \eta_{progress} \cdot \frac{k}{K}\right)
$$

其中 $R_{delivery} = 50$，$\eta_{progress} = 0.5$，后续任务给予更高奖励。

**全部配送完成的额外奖励**：

$$
R_{all\_done} =
\begin{cases}
+300, & \text{if all } K \text{ deliveries completed} \\
0, & \text{otherwise}
\end{cases}
$$

**变量说明**：
- $\mathbf{p}_{dest}^{(k)}$：第 $k$ 个配送任务的目的地坐标
- $d_{drop} = 3.0$ m：投放/降落判定距离
- $K$：总配送任务数

**设计考量**：
- 50的基础奖励低于到达奖励（200），引导智能体优先完成所有配送而非单一任务
- 递增奖励策略防止智能体只完成容易的任务
- 全部完成的额外大奖（300）提供终极目标激励

---

## 3. 完整奖励函数汇总

### 3.1 总奖励公式

$$
R_t = \underbrace{200 \cdot \mathbb{1}[\text{goal reached}]}_{R_{goal}}
+ \underbrace{1.0 \cdot (d_{t-1} - d_t)}_{R_{dist}}
- \underbrace{200 \cdot \mathbb{1}[\text{collision}]}_{P_{collision}}
- \underbrace{10 \cdot \sum_{j} \max\left(0, \frac{3 - d_{ij}}{2}\right)^2}_{P_{safe}}
- \underbrace{100 \cdot \sum_{i<j} \max\left(0, 1 - \frac{d_{ij}}{10}\right)}_{P_{proximity}}
- \underbrace{1.0 \cdot \frac{1}{N}\sum_{i} \| \mathbf{p}_t^{(i)} - \mathbf{p}_{des}^{(i)} \|}_{P_{formation}}
- \underbrace{0.5 \cdot \frac{1}{N}\sum_{i} E_t^{(i)}}_{P_{energy}}
- \underbrace{2}_{P_{time}}
- \underbrace{0.1 \cdot \frac{1}{N}\sum_{i} \| \Delta\mathbf{a}_t^{(i)} \|^2}_{P_{smooth}}
+ \underbrace{\sum_k r_k \cdot \mathbb{1}[\text{delivery}_k]}_{R_{delivery}}
+ \underbrace{300 \cdot \mathbb{1}[\text{all done}]}_{R_{all\_done}}
- \underbrace{100 \cdot \mathbb{1}[\text{timeout}]}_{P_{timeout}}
$$

### 3.2 参数汇总表

| 分量 | 符号 | 类型 | 基准值 | 触发条件 |
|------|------|------|--------|----------|
| 目标到达奖励 | $R_{goal}$ | 事件触发型 | +200 | 距目标 <= 5m |
| 距离引导奖励 | $R_{dist}$ | 连续型 | +1.0/step | 距离减小时 |
| 碰撞惩罚 | $P_{collision}$ | 事件触发型 | -200 | 碰撞检测触发 |
| 安全距离惩罚 | $P_{safe}$ | 连续型 | -10 (max) | 距 < 3m |
| 间距惩罚 | $P_{proximity}$ | 连续型 | -100 (max) | 距离 < 10m |
| 队形保持惩罚 | $P_{formation}$ | 连续型 | -1.0/step | 偏差存在时 |
| 能耗惩罚 | $P_{energy}$ | 连续型 | -0.5/step | 持续存在 |
| 时间惩罚 | $P_{time}$ | 连续型 | -2.0/step | 每步固定 |
| 平滑惩罚 | $P_{smooth}$ | 连续型 | -0.1/step | 动作变化时 |
| 配送完成奖励 | $R_{delivery}$ | 事件触发型 | +50 | 货物送达时 |
| 全部完成奖励 | $R_{all\_done}$ | 事件触发型 | +300 | 所有任务完成 |
| 超时惩罚 | $P_{timeout}$ | 事件触发型 | -100 | 步数超限 |

### 3.3 奖励量级分析

在一个典型的episode中（假设500步完成配送，3架无人机，5个配送任务）：

| 分量 | 单次量级 | 预计触发次数 | 预计总贡献 | 占比估计 |
|------|----------|-------------|-----------|----------|
| $R_{goal}$ | +200 | 1 | +200 | 12% |
| $R_{dist}$ | +0.5/step | 500步 | +250 | 15% |
| $P_{collision}$ | -200 | 0（理想） | 0 | 0% |
| $P_{safe}$ | -0.5/step | ~100步 | -50 | 3% |
| $P_{proximity}$ | -5/step | ~50步 | -250 | 15% |
| $P_{formation}$ | -2/step | 500步 | -1000 | (可关闭) |
| $P_{energy}$ | -0.5/step | 500步 | -250 | 15% |
| $P_{time}$ | -2/step | 500步 | -1000 | (主要负向) |
| $P_{smooth}$ | -0.05/step | 500步 | -25 | 1% |
| $R_{delivery}$ | +50 | 5 | +250 | 15% |
| $R_{all\_done}$ | +300 | 1 | +300 | 18% |

**平衡性校验**：
- 正向总奖励（最优路径）：+200 + 250 + 250 + 300 = +1000
- 负向总惩罚（最优路径）：-50 - 250 - 250 - 1000 - 25 = -1575
- 净奖励（最优路径）：约 -575
- 净奖励（随机/碰撞路径）：远低于 -1000

确保最优策略的回报显著高于其他策略。

---

## 4. 奖励归一化方法

### 4.1 问题分析

不同奖励分量的量级差异巨大（事件触发型可达200，连续型仅0.1-2），直接相加会导致：
1. 大量级分量主导梯度信号，小量级分量被淹没
2. 超参数调整困难，需要反复试错
3. 训练不稳定，不同阶段的奖励分布差异大

### 4.2 在线Running Mean归一化

对每个奖励分量维护在线统计量，进行z-score归一化：

$$
\hat{r}_i^{(t)} = \frac{r_i^{(t)} - \mu_i^{(t)}}{\sigma_i^{(t)} + \epsilon}
$$

**在线统计量更新**（Welford算法）：

$$
\mu_i^{(t)} = \mu_i^{(t-1)} + \frac{r_i^{(t)} - \mu_i^{(t-1)}}{t}
$$

$$
(\sigma_i^{(t)})^2 = (\sigma_i^{(t-1)})^2 + \frac{(r_i^{(t)} - \mu_i^{(t-1)})(r_i^{(t)} - \mu_i^{(t)}) - (\sigma_i^{(t-1)})^2}{t}
$$

其中 $\epsilon = 10^{-8}$ 为数值稳定项。

**归一化后的总奖励**：

$$
R_t^{norm} = \sum_{i} w_i \cdot \hat{r}_i^{(t)}
$$

其中 $w_i$ 为人工设定的相对重要性权重。

**推荐权重配置**：

$$
\mathbf{w} = [w_{goal}, w_{dist}, w_{collision}, w_{safe}, w_{proximity}, w_{formation}, w_{energy}, w_{time}, w_{smooth}, w_{delivery}]
$$
$$
\mathbf{w} = [3.0, \; 1.0, \; 3.0, \; 1.0, \; 1.5, \; 0.5, \; 0.5, \; 1.0, \; 0.3, \; 2.0]
$$

### 4.3 自适应权重调整（Reward Component Balancing）

基于各分量对总梯度的实际贡献比例，动态调整权重：

**目标贡献比**：

| 分量 | 目标贡献占比 |
|------|-------------|
| $R_{goal} + R_{delivery} + R_{all\_done}$ | 30% |
| $P_{collision} + P_{safe}$ | 20% |
| $P_{proximity}$ | 15% |
| $R_{dist}$ | 15% |
| $P_{energy}$ | 10% |
| $P_{time}$ | 7% |
| $P_{formation}$ | 2% |
| $P_{smooth}$ | 1% |

**自适应调整规则**：

$$
w_i^{(t+1)} = w_i^{(t)} \cdot \exp\left(\eta_w \cdot \left(\frac{\text{target}_i}{\text{actual}_i^{(t)}} - 1\right)\right)
$$

其中 $\eta_w = 0.01$ 为调整速率，$\text{actual}_i^{(t)}$ 为分量 $i$ 在最近 $T_w = 100$ 步内的实际贡献比例。

### 4.4 奖励裁剪（Reward Clipping）

防止极端奖励值导致训练不稳定：

**硬裁剪**：

$$
R_t^{clipped} = \text{clip}(R_t^{norm}, \; -R_{max}, \; R_{max})
$$

其中 $R_{max} = 10.0$。

**软裁剪**（tanh压缩）：

$$
R_t^{soft} = R_{max} \cdot \tanh\left(\frac{R_t^{norm}}{R_{max}}\right)
$$

**建议**：使用软裁剪，保留梯度信息的同时限制值域。

---

## 5. 稀疏奖励问题处理（Reward Shaping）

### 5.1 问题分析

本项目面临严重的稀疏奖励挑战：
- $R_{goal}$ 和 $R_{delivery}$ 仅在任务完成时触发
- 典型episode长度为500-1000步
- 智能体在初期几乎无法获得正向信号

### 5.2 基于势能的奖励塑形（Potential-Based Reward Shaping）

根据 Ng et al. (1999) 的理论，添加基于势函数的塑形奖励不会改变最优策略：

$$
F(s, a, s') = \gamma \Phi(s') - \Phi(s)
$$

**势函数设计**（基于到最近目标的距离）：

$$
\Phi(s) = -c \cdot \min_{k \in \text{remaining}} \| \mathbf{p}_t - \mathbf{p}_{dest}^{(k)} \|_2
$$

**塑形奖励**：

$$
F(s, a, s') = c \cdot \left( \| \mathbf{p}_t - \mathbf{p}_{target}^{nearest} \|_2 - \gamma \cdot \| \mathbf{p}_{t+1} - \mathbf{p}_{target}^{nearest} \|_2 \right)
$$

其中 $c = 0.5$ 为塑形强度系数，$\gamma = 0.99$ 为折扣因子。

**物理含义**：
- 靠近目标：$F > 0$（正奖励）
- 远离目标：$F < 0$（负奖励）
- 不变：$F = 0$

**多任务势函数**（贪心策略）：

$$
\Phi(s) = -c \cdot \min_{k \in \text{remaining}} \| \mathbf{p}_t - \mathbf{p}_{dest}^{(k)} \|_2
$$

**多任务势函数**（TSP启发式，考虑所有剩余任务）：

$$
\Phi(s) = -c \cdot \text{TSP\_length}(\mathbf{p}_t, \{\mathbf{p}_{dest}^{(k)} : k \in \text{remaining}\})
$$

### 5.3 基于进度的奖励塑形

定义任务进度指标 $\eta_t \in [0, 1]$：

$$
\eta_t = 1 - \frac{\| \mathbf{p}_t - \mathbf{p}_{goal} \|_2}{\| \mathbf{p}_0 - \mathbf{p}_{goal} \|_2}
$$

进度奖励：

$$
R_{progress} = c_p \cdot (\eta_t - \eta_{t-1})
$$

其中 $c_p = 5.0$ 为进度奖励系数。

### 5.4 基于中间目标的奖励

设置路径中间检查点，提供阶段性奖励：

$$
R_{waypoint} = \sum_{w=1}^{W} r_w \cdot \mathbb{1}\left[\| \mathbf{p}_t - \mathbf{p}_{waypoint}^{(w)} \|_2 \leq d_w\right]
$$

- $W$：检查点数量（建议3-5个）
- $r_w = 10$：每个检查点的奖励
- $d_w = 5$ m：检查点判定距离

**检查点生成方法**：
- 均匀插值：在起始点和目标点之间等距插入
- 路径规划辅助：使用A*等算法生成参考路径，在路径上设置检查点

### 5.5 探索增强策略

#### 5.5.1 内在好奇心模块（ICM）

添加基于预测误差的好奇心奖励：

$$
R_{icm} = \eta_{icm} \cdot \| \hat{\phi}(s_{t+1}) - \phi(s_{t+1}) \|_2^2
$$

- $\phi(s)$：状态编码器
- $\hat{\phi}(s_{t+1})$：基于 $(s_t, a_t)$ 的预测
- $\eta_{icm} = 0.1$：好奇心权重

#### 5.5.2 噪声增强探索

在训练初期使用参数空间噪声：

$$
\mathbf{a}_t = \pi_\theta(s_t) + \sigma_{noise} \cdot \mathbf{\epsilon}_t, \quad \mathbf{\epsilon}_t \sim \mathcal{N}(0, I)
$$

$$
\sigma_{noise}^{(epoch)} = \sigma_{init} \cdot e^{-epoch / T_\sigma} + \sigma_{final}
$$

- $\sigma_{init} = 0.3$
- $\sigma_{final} = 0.01$
- $T_\sigma = 30$

### 5.6 稀疏奖励处理策略汇总

| 策略 | 适用阶段 | 预期效果 | 实现复杂度 |
|------|---------|---------|-----------|
| 势能塑形 | 全阶段 | 提供密集梯度信号 | 低 |
| 进度奖励 | 全阶段 | 强化正向进展 | 低 |
| 中间检查点 | 阶段1-2 | 分解长期任务 | 中 |
| 距离引导奖励 | 全阶段 | 持续方向性信号 | 低 |
| ICM好奇心 | 阶段1-2 | 增强初期探索 | 高 |
| 噪声增强 | 全阶段 | 促进探索多样性 | 低 |

---

## 6. 课程学习（Curriculum Learning）策略

### 6.1 环境难度递进

分4个阶段逐步增加任务难度：

#### 阶段1：单机简单导航（100K steps）

| 项目 | 配置 |
|------|------|
| 无人机数量 | 1 |
| 障碍物数量 | 0 |
| 配送任务数 | 1 |
| 目标距离 | 10-50 m |
| 启用奖励 | $R_{goal}$, $R_{dist}$, $P_{time}$, 势能塑形 |
| 碰撞处理 | 重置位置（不终止episode） |

**训练目标**：学会基本的飞行和到达目标。

#### 阶段2：单机复杂环境（200K steps）

| 项目 | 配置 |
|------|------|
| 无人机数量 | 1 |
| 障碍物数量 | 5-10 |
| 配送任务数 | 1 |
| 目标距离 | 50-200 m |
| 启用奖励 | 阶段1 + $P_{collision}$, $P_{safe}$, $P_{energy}$ |
| 碰撞处理 | 终止episode |

**训练目标**：学会避障导航和能量管理。

#### 阶段3：多机协作（300K steps）

| 项目 | 配置 |
|------|------|
| 无人机数量 | 2-3 |
| 障碍物数量 | 5-10 |
| 配送任务数 | 2-3 |
| 目标距离 | 50-300 m |
| 启用奖励 | 阶段2 + $P_{proximity}$, $P_{formation}$ |
| 碰撞处理 | 终止episode |

**训练目标**：学会协调飞行、间距保持和队形维持。

#### 阶段4：完整配送任务（持续训练）

| 项目 | 配置 |
|------|------|
| 无人机数量 | 3-5 |
| 障碍物数量 | 10-20 |
| 配送任务数 | 3-8 |
| 目标距离 | 50-500 m |
| 启用奖励 | 全部8个分量 + $R_{delivery}$, $R_{all\_done}$, $P_{smooth}$ |
| 碰撞处理 | 终止episode |

**训练目标**：完成端到端多任务配送。

### 6.2 任务距离递进

在每个阶段内，逐步增加起始点到目标点的距离：

$$
d_{task}^{(epoch)} = d_{min} + (d_{max} - d_{min}) \cdot \min\left(1, \frac{epoch}{T_{ramp}}\right)
$$

- $d_{min} = 10$ m（近距离任务）
- $d_{max} = 500$ m（远距离任务）
- $T_{ramp} = 50$ 个epoch（递增周期）

### 6.3 奖励塑形强度递减

随着训练进行，逐步降低塑形奖励的权重，让智能体主要依赖原始稀疏奖励：

$$
c^{(epoch)} = c_{init} \cdot \left(\frac{T_{decay}}{T_{decay} + epoch}\right)
$$

- $c_{init} = 2.0$：初始塑形强度
- $T_{decay} = 20$：衰减时间常数

### 6.4 课程学习触发条件

自动判断是否进入下一阶段：

```python
def should_advance_stage(current_stage: int, metrics: dict) -> bool:
    """
    进入下一阶段的判定条件（全部满足）：
    1. 最近100个episode的成功率 > 阈值
    2. 平均episode长度 < 阶段阈值
    3. 碰撞率 < 阈值
    """
    if current_stage >= 4:  # 最大阶段
        return False

    success_rate = metrics["success_rate_recent_100"]
    avg_length = metrics["avg_episode_length"]
    collision_rate = metrics["collision_rate"]

    thresholds = {
        1: {"success": 0.80, "max_length": 200, "max_collision": 0.05},
        2: {"success": 0.75, "max_length": 400, "max_collision": 0.05},
        3: {"success": 0.70, "max_length": 600, "max_collision": 0.03},
    }

    if current_stage not in thresholds:
        return False

    t = thresholds[current_stage]
    return (success_rate > t["success"] and
            avg_length < t["max_length"] and
            collision_rate < t["max_collision"])
```

---

## 7. 参数标定方法

### 7.1 分层标定策略

采用"从简到繁、逐步叠加"的标定策略：

**第一层：核心任务（$R_{goal}$ + $R_{dist}$ + $P_{collision}$ + $P_{time}$）**
- 仅使用这4个分量进行初始训练
- 标定 $R_{goal}$ 和 $P_{collision}$ 的相对比例（建议1:1），确保安全与目标平衡
- 标定 $\lambda_{time}$ 使episode长度在合理范围内

**第二层：安全约束（+ $P_{proximity}$ + $P_{safe}$）**
- 加入间距惩罚，观察多机协作行为
- 调整阈值和权重，避免过度保守（目标：碰撞率 < 5%）

**第三层：编队与效率（+ $P_{formation}$ + $P_{energy}$ + $R_{delivery}$）**
- 加入队形、能耗和配送奖励
- 平衡各项权重

**第四层：平滑性（+ $P_{smooth}$ + $R_{all\_done}$）**
- 最后加入平滑惩罚作为微调
- 添加全部完成大奖

### 7.2 基于敏感度分析的标定

对每个参数进行网格搜索：

1. 固定其他参数，变化目标参数 $\lambda \in \{0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0\}$
2. 训练200个episode，记录关键指标：
   - 成功率（到达目标的比例）
   - 平均episode长度
   - 碰撞率
   - 队形偏差均值
   - 平均能耗
3. 选择使综合指标最优的参数值

**综合评价指标**：

$$
J = w_s \cdot \text{SuccessRate} - w_l \cdot \frac{\text{AvgLength}}{L_{max}} - w_c \cdot \text{CollisionRate} - w_f \cdot \frac{\text{AvgFormationError}}{E_{max}}
$$

- $w_s = 0.4$：成功率权重
- $w_l = 0.2$：效率权重
- $w_c = 0.3$：安全权重
- $w_f = 0.1$：队形权重

### 7.3 基于奖励分量贡献比的监控

训练过程中监控各分量实际贡献，动态调整权重使其趋近目标比例：

```python
class RewardMonitor:
    """奖励分量贡献监控器"""

    def __init__(self, target_ratios: dict, window_size: int = 100, adapt_rate: float = 0.01):
        self.target_ratios = target_ratios
        self.window_size = window_size
        self.adapt_rate = adapt_rate
        self.component_history = {k: [] for k in target_ratios}
        self.weights = {k: 1.0 for k in target_ratios}

    def update(self, reward_components: dict):
        """更新各分量历史记录"""
        for k, v in reward_components.items():
            if k in self.component_history:
                self.component_history[k].append(abs(v))
                if len(self.component_history[k]) > self.window_size:
                    self.component_history[k].pop(0)

    def compute_actual_ratios(self) -> dict:
        """计算各分量的实际贡献比"""
        means = {}
        total = 0.0
        for k, values in self.component_history.items():
            if values:
                m = np.mean(values)
                means[k] = m
                total += m
        if total == 0:
            return {k: 0.0 for k in means}
        return {k: v / total for k, v in means.items()}

    def adapt_weights(self) -> dict:
        """自适应调整权重"""
        actual = self.compute_actual_ratios()
        for k in self.weights:
            if k in actual and actual[k] > 0:
                ratio = self.target_ratios.get(k, 0.1) / actual[k]
                self.weights[k] *= np.exp(self.adapt_rate * (ratio - 1))
                self.weights[k] = np.clip(self.weights[k], 0.1, 10.0)
        return self.weights.copy()
```

### 7.4 推荐初始参数配置

```python
REWARD_CONFIG = {
    # ===== 事件触发奖励 =====
    "R_goal": 200.0,               # 目标到达奖励
    "R_delivery": 50.0,            # 单任务配送完成奖励
    "R_all_done": 300.0,           # 全部配送完成额外奖励
    "P_collision": -200.0,         # 碰撞惩罚
    "P_timeout": -100.0,           # 超时惩罚

    # ===== 安全约束 =====
    "P_proximity_max": -100.0,     # 间距惩罚最大值
    "d_safe": 10.0,                # 安全距离阈值 (m)
    "d_safe_warn": 3.0,            # 碰撞预警距离 (m)
    "lambda_safe": 10.0,           # 安全距离惩罚权重

    # ===== 连续惩罚权重 =====
    "c_dist": 1.0,                 # 距离引导奖励系数
    "lambda_form": 1.0,            # 队形惩罚权重
    "lambda_energy": 0.5,          # 能耗惩罚权重
    "lambda_time": 2.0,            # 时间惩罚权重
    "lambda_smooth": 0.1,          # 平滑惩罚权重

    # ===== 能耗模型参数 =====
    "alpha_accel": 1.0,            # 加速度能耗系数
    "beta_velocity": 0.1,          # 速度维持能耗系数
    "gamma_altitude": 2.0,         # 高度变化能耗系数
    "eta_batt": 2.0,               # 电量自适应系数

    # ===== 队形参数 =====
    "formation_type": "V",         # V字/直线/菱形
    "formation_spacing": 8.0,      # 编队间距 (m)

    # ===== 任务参数 =====
    "goal_threshold": 5.0,         # 目标到达判定距离 (m)
    "drop_threshold": 3.0,         # 配送完成判定距离 (m)
    "collision_radius": 1.0,       # 碰撞判定半径 (m)
    "eta_progress": 0.5,           # 递增配送奖励系数

    # ===== 奖励塑形 =====
    "shaping_c": 0.5,              # 势能塑形强度
    "shaping_c_p": 5.0,            # 进度奖励系数
    "shaping_c_init": 2.0,         # 初始塑形强度
    "shaping_T_decay": 20,         # 塑形衰减时间常数

    # ===== 课程学习 =====
    "curriculum_enabled": True,
    "curriculum_stage": 1,         # 初始阶段
    "d_task_min": 10.0,            # 最小任务距离 (m)
    "d_task_max": 500.0,           # 最大任务距离 (m)
    "T_ramp": 50,                  # 距离递增周期 (epoch)

    # ===== 归一化与裁剪 =====
    "normalize_rewards": True,
    "reward_clip_max": 10.0,       # 奖励裁剪上限
    "use_soft_clip": True,         # 使用软裁剪(tanh)

    # ===== Episode参数 =====
    "max_steps": 1000,             # 最大步数
    "gamma": 0.99,                 # 折扣因子
}
```

---

## 8. 奖励函数Python实现代码

### 8.1 核心奖励计算器

```python
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque


class RewardNormalizer:
    """在线奖励归一化器（Welford算法）"""

    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, value: float):
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.M2 += delta * delta2

    def normalize(self, value: float) -> float:
        if self.count < 2:
            return value
        variance = self.M2 / (self.count - 1)
        std = np.sqrt(variance + self.epsilon)
        return (value - self.mean) / std

    def reset(self):
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0


class RewardCalculator:
    """
    无人机物流配送奖励计算器

    8个奖励分量 + 势能塑形 + 归一化
    """

    def __init__(self, config: dict):
        self.config = config

        # ===== 事件触发奖励 =====
        self.R_goal = config.get("R_goal", 200.0)
        self.R_delivery = config.get("R_delivery", 50.0)
        self.R_all_done = config.get("R_all_done", 300.0)
        self.P_collision = config.get("P_collision", -200.0)
        self.P_timeout = config.get("P_timeout", -100.0)

        # ===== 安全约束 =====
        self.P_proximity_max = config.get("P_proximity_max", -100.0)
        self.d_safe = config.get("d_safe", 10.0)
        self.d_safe_warn = config.get("d_safe_warn", 3.0)
        self.lambda_safe = config.get("lambda_safe", 10.0)

        # ===== 连续惩罚权重 =====
        self.c_dist = config.get("c_dist", 1.0)
        self.lambda_form = config.get("lambda_form", 1.0)
        self.lambda_energy = config.get("lambda_energy", 0.5)
        self.lambda_time = config.get("lambda_time", 2.0)
        self.lambda_smooth = config.get("lambda_smooth", 0.1)

        # ===== 能耗模型参数 =====
        self.alpha_accel = config.get("alpha_accel", 1.0)
        self.beta_velocity = config.get("beta_velocity", 0.1)
        self.gamma_altitude = config.get("gamma_altitude", 2.0)
        self.eta_batt = config.get("eta_batt", 2.0)

        # ===== 队形参数 =====
        self.formation_type = config.get("formation_type", "V")
        self.formation_spacing = config.get("formation_spacing", 8.0)

        # ===== 任务参数 =====
        self.goal_threshold = config.get("goal_threshold", 5.0)
        self.drop_threshold = config.get("drop_threshold", 3.0)
        self.collision_radius = config.get("collision_radius", 1.0)
        self.eta_progress = config.get("eta_progress", 0.5)

        # ===== 奖励塑形参数 =====
        self.shaping_c = config.get("shaping_c", 0.5)
        self.gamma = config.get("gamma", 0.99)

        # ===== 归一化配置 =====
        self.normalize_rewards = config.get("normalize_rewards", True)
        self.reward_clip_max = config.get("reward_clip_max", 10.0)
        self.use_soft_clip = config.get("use_soft_clip", True)

        # ===== 内部状态 =====
        self.normalizers = {}
        self._prev_goal_reached = set()
        self._prev_delivery_done = set()

    def calculate(
        self,
        positions: np.ndarray,            # (N, 3) 无人机当前位置
        prev_positions: np.ndarray,        # (N, 3) 上一步位置
        actions: np.ndarray,              # (N, 3) 当前动作
        prev_actions: np.ndarray,         # (N, 3) 上一步动作
        velocities: np.ndarray,           # (N, 3) 当前速度
        goal_pos: np.ndarray,             # (3,) 主目标位置
        goal_positions: np.ndarray,       # (K, 3) 配送目标位置
        formation_offsets: np.ndarray,    # (N, 3) 期望队形偏移
        obstacles: np.ndarray,            # (M, 4) 障碍物 [x,y,z,radius]
        cargo_loaded: np.ndarray,         # (K,) 货物是否已装载
        cargo_delivered: np.ndarray,      # (K,) 货物是否已送达
        battery_ratios: np.ndarray,       # (N,) 各无人机剩余电量比例
        step_count: int,                  # 当前步数
        max_steps: int,                   # 最大步数
        boundary: np.ndarray,             # (3, 2) 空间边界 [[xmin,xmax],[ymin,ymax],[zmin,zmax]]
    ) -> Tuple[float, Dict[str, float]]:
        """
        计算总奖励和各分量

        返回:
            total_reward: 归一化后的总奖励
            reward_components: 各分量明细
        """
        N = positions.shape[0]
        K = goal_positions.shape[0]
        rewards = {}

        # ============================================================
        # 1. 目标到达奖励 R_goal
        # ============================================================
        dist_to_goal = np.linalg.norm(positions.mean(axis=0) - goal_pos)
        R_goal = self.R_goal if dist_to_goal <= self.goal_threshold else 0.0
        rewards["R_goal"] = R_goal

        # ============================================================
        # 2. 距离引导奖励 R_dist（辅助稠密信号）
        # ============================================================
        if prev_positions is not None:
            prev_dist = np.linalg.norm(prev_positions.mean(axis=0) - goal_pos)
            R_dist = self.c_dist * (prev_dist - dist_to_goal)
        else:
            R_dist = 0.0
        rewards["R_dist"] = R_dist

        # ============================================================
        # 3. 碰撞惩罚 P_collision
        # ============================================================
        collision, collision_details = self._check_collision(
            positions, obstacles, boundary
        )
        P_collision = self.P_collision if collision else 0.0
        rewards["P_collision"] = P_collision

        # ============================================================
        # 4. 安全距离惩罚 P_safe
        # ============================================================
        P_safe = self._safety_distance_penalty(positions, obstacles)
        rewards["P_safe"] = P_safe

        # ============================================================
        # 5. 间距惩罚 P_proximity
        # ============================================================
        P_proximity = self._proximity_penalty(positions)
        rewards["P_proximity"] = P_proximity

        # ============================================================
        # 6. 队形保持惩罚 P_formation
        # ============================================================
        if N > 1:
            center = positions.mean(axis=0)
            desired_pos = center + formation_offsets[:N]
            formation_errors = np.linalg.norm(positions - desired_pos, axis=1)
            P_formation = -self.lambda_form * formation_errors.mean()
        else:
            P_formation = 0.0
        rewards["P_formation"] = P_formation

        # ============================================================
        # 7. 能耗惩罚 P_energy
        # ============================================================
        energy_consumptions = self._compute_energy(actions, velocities, positions, prev_positions)
        # 自适应能耗系数（电量越低惩罚越大）
        avg_battery = battery_ratios.mean() if len(battery_ratios) > 0 else 1.0
        adaptive_lambda = self.lambda_energy * (1.0 + self.eta_batt * (1.0 - avg_battery))
        P_energy = -adaptive_lambda * energy_consumptions.mean()
        rewards["P_energy"] = P_energy

        # ============================================================
        # 8. 时间惩罚 P_time
        # ============================================================
        P_time = -self.lambda_time
        rewards["P_time"] = P_time

        # ============================================================
        # 9. 平滑惩罚 P_smooth
        # ============================================================
        if prev_actions is not None and step_count > 0:
            action_diff = actions - prev_actions
            P_smooth = -self.lambda_smooth * np.mean(np.linalg.norm(action_diff, axis=1) ** 2)
        else:
            P_smooth = 0.0
        rewards["P_smooth"] = P_smooth

        # ============================================================
        # 10. 配送完成奖励 R_delivery
        # ============================================================
        R_delivery = 0.0
        newly_delivered = []
        for k in range(K):
            dist = np.linalg.norm(positions.mean(axis=0) - goal_positions[k])
            if (dist <= self.drop_threshold and
                cargo_loaded[k] and not cargo_delivered[k] and
                k not in self._prev_delivery_done):
                # 递增奖励
                delivered_count = int(cargo_delivered.sum())
                r_k = self.R_delivery * (1.0 + self.eta_progress * delivered_count / max(K, 1))
                R_delivery += r_k
                newly_delivered.append(k)
        rewards["R_delivery"] = R_delivery

        # ============================================================
        # 11. 全部完成奖励 R_all_done
        # ============================================================
        all_done = cargo_delivered.all() or (cargo_delivered.sum() + len(newly_delivered)) >= K
        R_all_done = self.R_all_done if all_done and not cargo_delivered.all() else 0.0
        rewards["R_all_done"] = R_all_done

        # ============================================================
        # 12. 超时惩罚 P_timeout
        # ============================================================
        P_timeout = self.P_timeout if step_count >= max_steps else 0.0
        rewards["P_timeout"] = P_timeout

        # ============================================================
        # 13. 势能塑形奖励（不改变最优策略）
        # ============================================================
        if prev_positions is not None:
            remaining_targets = [goal_positions[k] for k in range(K) if not cargo_delivered[k]]
            if remaining_targets:
                # 使用最近目标的势函数
                prev_dists = [np.linalg.norm(prev_positions.mean(axis=0) - t) for t in remaining_targets]
                curr_dists = [np.linalg.norm(positions.mean(axis=0) - t) for t in remaining_targets]
                prev_min = min(prev_dists)
                curr_min = min(curr_dists)
                shaping = self.shaping_c * (self.gamma * (-curr_min) - (-prev_min))
            else:
                shaping = 0.0
        else:
            shaping = 0.0
        rewards["shaping"] = shaping

        # ============================================================
        # 更新内部状态
        # ============================================================
        for k in newly_delivered:
            self._prev_delivery_done.add(k)
        if dist_to_goal <= self.goal_threshold:
            self._prev_goal_reached.add("main")

        # ============================================================
        # 奖励归一化与裁剪
        # ============================================================
        raw_total = sum(rewards.values())

        if self.normalize_rewards:
            total_reward = self._normalize_and_clip(rewards)
        else:
            total_reward = raw_total

        return total_reward, rewards

    def reset_episode(self):
        """重置episode状态"""
        self._prev_goal_reached.clear()
        self._prev_delivery_done.clear()

    # ============================================================
    # 内部方法
    # ============================================================

    def _check_collision(
        self,
        positions: np.ndarray,
        obstacles: np.ndarray,
        boundary: np.ndarray
    ) -> Tuple[bool, dict]:
        """检测碰撞（无人机间、障碍物、越界）"""
        N = len(positions)
        details = {"uav_collision": False, "obs_collision": False, "boundary_violation": False}

        # 无人机间碰撞
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < self.collision_radius * 2:
                    details["uav_collision"] = True
                    return True, details

        # 障碍物碰撞
        for i in range(N):
            for obs in obstacles:
                obs_pos = obs[:3]
                obs_r = obs[3]
                dist = np.linalg.norm(positions[i] - obs_pos)
                if dist < (self.collision_radius + obs_r):
                    details["obs_collision"] = True
                    return True, details

        # 越界碰撞
        for i in range(N):
            for dim in range(3):
                if positions[i, dim] < boundary[dim, 0] or positions[i, dim] > boundary[dim, 1]:
                    details["boundary_violation"] = True
                    return True, details

        return False, details

    def _safety_distance_penalty(
        self,
        positions: np.ndarray,
        obstacles: np.ndarray
    ) -> float:
        """安全距离渐进惩罚（碰撞预警）"""
        penalty = 0.0
        N = len(positions)

        # 无人机间预警
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < self.d_safe_warn:
                    ratio = (self.d_safe_warn - dist) / (self.d_safe_warn - self.collision_radius * 2)
                    penalty -= self.lambda_safe * max(0, ratio) ** 2

        # 障碍物预警
        for i in range(N):
            for obs in obstacles:
                obs_pos = obs[:3]
                obs_r = obs[3]
                dist = np.linalg.norm(positions[i] - obs_pos)
                warn_dist = self.d_safe_warn + obs_r
                if dist < warn_dist:
                    ratio = (warn_dist - dist) / (warn_dist - (self.collision_radius + obs_r))
                    penalty -= self.lambda_safe * max(0, ratio) ** 2

        return penalty

    def _proximity_penalty(self, positions: np.ndarray) -> float:
        """计算无人机间距惩罚"""
        penalty = 0.0
        N = len(positions)
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.linalg.norm(positions[i] - positions[j])
                penalty += max(0, 1.0 - dist / self.d_safe)
        return self.P_proximity_max * penalty

    def _compute_energy(
        self,
        actions: np.ndarray,
        velocities: np.ndarray,
        positions: np.ndarray,
        prev_positions: Optional[np.ndarray]
    ) -> np.ndarray:
        """计算各无人机能耗"""
        # 加速度能耗
        e_accel = np.linalg.norm(actions, axis=1) ** 2

        # 速度维持能耗
        e_velocity = self.beta_velocity * np.linalg.norm(velocities, axis=1) ** 2

        # 高度变化能耗
        if prev_positions is not None:
            altitude_change = np.abs(positions[:, 2] - prev_positions[:, 2])
            e_altitude = self.gamma_altitude * altitude_change
        else:
            e_altitude = np.zeros(len(positions))

        return self.alpha_accel * e_accel + e_velocity + e_altitude

    def _normalize_and_clip(self, rewards: Dict[str, float]) -> float:
        """归一化并裁剪各分量，返回总奖励"""
        normalized_total = 0.0

        for name, value in rewards.items():
            # 初始化归一化器
            if name not in self.normalizers:
                self.normalizers[name] = RewardNormalizer()
            normalizer = self.normalizers[name]

            # 更新统计量
            normalizer.update(value)

            # 归一化
            norm_value = normalizer.normalize(value)

            # 裁剪
            if self.use_soft_clip:
                clipped = self.reward_clip_max * np.tanh(norm_value / self.reward_clip_max)
            else:
                clipped = np.clip(norm_value, -self.reward_clip_max, self.reward_clip_max)

            normalized_total += clipped

        return normalized_total


# ============================================================
# 课程学习管理器
# ============================================================

class CurriculumManager:
    """课程学习管理器"""

    STAGES = {
        1: {
            "name": "单机简单导航",
            "num_uavs": 1,
            "num_obstacles": 0,
            "num_targets": 1,
            "d_task_range": (10, 50),
            "enabled_rewards": ["R_goal", "R_dist", "P_time", "shaping"],
            "collision_terminate": False,
            "target_steps": 100_000,
        },
        2: {
            "name": "单机复杂环境",
            "num_uavs": 1,
            "num_obstacles_range": (5, 10),
            "num_targets": 1,
            "d_task_range": (50, 200),
            "enabled_rewards": ["R_goal", "R_dist", "P_collision", "P_safe",
                                "P_energy", "P_time", "shaping"],
            "collision_terminate": True,
            "target_steps": 200_000,
        },
        3: {
            "name": "多机协作",
            "num_uavs_range": (2, 3),
            "num_obstacles_range": (5, 10),
            "num_targets_range": (2, 3),
            "d_task_range": (50, 300),
            "enabled_rewards": ["R_goal", "R_dist", "P_collision", "P_safe",
                                "P_proximity", "P_formation", "P_energy", "P_time", "shaping"],
            "collision_terminate": True,
            "target_steps": 300_000,
        },
        4: {
            "name": "完整配送任务",
            "num_uavs_range": (3, 5),
            "num_obstacles_range": (10, 20),
            "num_targets_range": (3, 8),
            "d_task_range": (50, 500),
            "enabled_rewards": "all",
            "collision_terminate": True,
            "target_steps": None,  # 持续训练
        },
    }

    def __init__(self, initial_stage: int = 1):
        self.current_stage = initial_stage
        self.total_steps = 0
        self.episode_count = 0
        self.recent_successes = deque(maxlen=100)
        self.recent_lengths = deque(maxlen=100)
        self.recent_collisions = deque(maxlen=100)

    def get_stage_config(self) -> dict:
        """获取当前阶段配置"""
        return self.STAGES[self.current_stage]

    def update(self, success: bool, episode_length: int, collision: bool):
        """更新训练统计"""
        self.episode_count += 1
        self.total_steps += episode_length
        self.recent_successes.append(float(success))
        self.recent_lengths.append(episode_length)
        self.recent_collisions.append(float(collision))

    def should_advance(self) -> bool:
        """判断是否应进入下一阶段"""
        if self.current_stage >= 4:
            return False
        if len(self.recent_successes) < 50:
            return False

        metrics = {
            "success_rate": np.mean(self.recent_successes),
            "avg_length": np.mean(self.recent_lengths),
            "collision_rate": np.mean(self.recent_collisions),
        }

        thresholds = {
            1: {"success": 0.80, "max_length": 200, "max_collision": 0.05},
            2: {"success": 0.75, "max_length": 400, "max_collision": 0.05},
            3: {"success": 0.70, "max_length": 600, "max_collision": 0.03},
        }

        t = thresholds[self.current_stage]
        return (metrics["success_rate"] > t["success"] and
                metrics["avg_length"] < t["max_length"] and
                metrics["collision_rate"] < t["max_collision"])

    def advance_stage(self):
        """进入下一阶段"""
        if self.current_stage < 4:
            self.current_stage += 1
            self.recent_successes.clear()
            self.recent_lengths.clear()
            self.recent_collisions.clear()
            print(f"[Curriculum] 进入阶段 {self.current_stage}: {self.STAGES[self.current_stage]['name']}")

    def get_task_distance(self, epoch: int, d_min: float = 10.0, d_max: float = 500.0, T_ramp: int = 50) -> float:
        """获取当前epoch的任务距离"""
        return d_min + (d_max - d_min) * min(1.0, epoch / T_ramp)

    def get_shaping_coefficient(self, epoch: int, c_init: float = 2.0, T_decay: int = 20) -> float:
        """获取当前epoch的塑形强度（随训练衰减）"""
        return c_init * (T_decay / (T_decay + epoch))


# ============================================================
# 奖励监控器
# ============================================================

class RewardMonitor:
    """奖励分量贡献监控器"""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.component_history = {}
        self.total_rewards = deque(maxlen=window_size)

    def log(self, reward_components: Dict[str, float], total_reward: float):
        """记录奖励分量"""
        for k, v in reward_components.items():
            if k not in self.component_history:
                self.component_history[k] = deque(maxlen=self.window_size)
            self.component_history[k].append(v)
        self.total_rewards.append(total_reward)

    def get_statistics(self) -> dict:
        """获取统计信息"""
        stats = {
            "total_reward_mean": np.mean(self.total_rewards) if self.total_rewards else 0.0,
            "total_reward_std": np.std(self.total_rewards) if self.total_rewards else 0.0,
        }
        for k, values in self.component_history.items():
            if values:
                abs_values = [abs(v) for v in values]
                stats[f"{k}_mean"] = np.mean(values)
                stats[f"{k}_abs_mean"] = np.mean(abs_values)
        return stats

    def get_contribution_ratios(self) -> dict:
        """获取各分量贡献比例"""
        abs_means = {}
        total = 0.0
        for k, values in self.component_history.items():
            if values:
                m = np.mean([abs(v) for v in values])
                abs_means[k] = m
                total += m
        if total == 0:
            return {k: 0.0 for k in abs_means}
        return {k: v / total for k, v in abs_means.items()}

    def print_report(self):
        """打印监控报告"""
        stats = self.get_statistics()
        ratios = self.get_contribution_ratios()

        print("=" * 60)
        print("奖励监控报告")
        print("=" * 60)
        print(f"总奖励均值: {stats['total_reward_mean']:.4f} +/- {stats['total_reward_std']:.4f}")
        print("-" * 60)
        print(f"{'分量':<15} {'均值':>10} {'贡献比':>10}")
        print("-" * 60)
        for k in sorted(ratios.keys()):
            mean_val = stats.get(f"{k}_mean", 0.0)
            ratio = ratios[k]
            print(f"{k:<15} {mean_val:>10.4f} {ratio:>9.1%}")
        print("=" * 60)
```

### 8.2 队形生成工具

```python
class FormationGenerator:
    """编队队形生成器"""

    @staticmethod
    def v_formation(n_uavs: int, spacing: float = 8.0) -> np.ndarray:
        """V字队形"""
        offsets = np.zeros((n_uavs, 3))
        center_idx = n_uavs // 2
        for i in range(n_uavs):
            offsets[i, 0] = -abs(i - center_idx) * spacing * 0.5   # 纵向（前进方向）
            offsets[i, 1] = (i - center_idx) * spacing              # 横向
        return offsets

    @staticmethod
    def line_formation(n_uavs: int, spacing: float = 8.0) -> np.ndarray:
        """直线队形"""
        offsets = np.zeros((n_uavs, 3))
        center_idx = n_uavs // 2
        for i in range(n_uavs):
            offsets[i, 1] = (i - center_idx) * spacing
        return offsets

    @staticmethod
    def diamond_formation(n_uavs: int, spacing: float = 8.0) -> np.ndarray:
        """菱形队形（适用于3或5架无人机）"""
        if n_uavs == 1:
            return np.zeros((1, 3))
        elif n_uavs == 2:
            return np.array([[0, -spacing / 2, 0], [0, spacing / 2, 0]])
        elif n_uavs == 3:
            return np.array([
                [-spacing * 0.5, 0, 0],
                [0, -spacing / 2, 0],
                [0, spacing / 2, 0],
            ])
        elif n_uavs == 5:
            return np.array([
                [-spacing, 0, 0],
                [-spacing / 2, -spacing / 2, 0],
                [-spacing / 2, spacing / 2, 0],
                [0, -spacing, 0],
                [0, spacing, 0],
            ])
        else:
            return FormationGenerator.v_formation(n_uavs, spacing)

    @staticmethod
    def get_formation(formation_type: str, n_uavs: int, spacing: float = 8.0) -> np.ndarray:
        """获取队形偏移量"""
        generators = {
            "V": FormationGenerator.v_formation,
            "line": FormationGenerator.line_formation,
            "diamond": FormationGenerator.diamond_formation,
        }
        generator = generators.get(formation_type, FormationGenerator.v_formation)
        return generator(n_uavs, spacing)
```

### 8.3 使用示例

```python
def training_loop_example():
    """训练循环示例"""

    # 初始化配置
    reward_config = {
        "R_goal": 200.0,
        "R_delivery": 50.0,
        "R_all_done": 300.0,
        "P_collision": -200.0,
        "P_proximity_max": -100.0,
        "d_safe": 10.0,
        "lambda_form": 1.0,
        "lambda_energy": 0.5,
        "lambda_time": 2.0,
        "lambda_smooth": 0.1,
        "shaping_c": 0.5,
        "gamma": 0.99,
        "normalize_rewards": True,
        "reward_clip_max": 10.0,
        "use_soft_clip": True,
    }

    reward_calculator = RewardCalculator(reward_config)
    curriculum = CurriculumManager(initial_stage=1)
    monitor = RewardMonitor(window_size=100)

    for epoch in range(1000):
        stage_config = curriculum.get_stage_config()

        # 根据课程学习阶段配置环境
        n_uavs = stage_config.get("num_uavs", 3)
        formation_offsets = FormationGenerator.get_formation("V", n_uavs)

        for episode in range(10):
            reward_calculator.reset_episode()
            total_reward = 0.0
            step = 0
            done = False

            # 模拟环境重置
            positions = np.random.randn(n_uavs, 3) * 10
            prev_positions = None
            prev_actions = None

            while not done and step < 1000:
                # 模拟动作选择
                actions = np.random.randn(n_uavs, 3) * 0.5
                velocities = actions * 0.1  # 简化

                # 计算奖励
                r, components = reward_calculator.calculate(
                    positions=positions,
                    prev_positions=prev_positions,
                    actions=actions,
                    prev_actions=prev_actions,
                    velocities=velocities,
                    goal_pos=np.array([50.0, 50.0, 20.0]),
                    goal_positions=np.array([[50.0, 50.0, 20.0]]),
                    formation_offsets=formation_offsets,
                    obstacles=np.array([[30.0, 30.0, 15.0, 5.0]]),
                    cargo_loaded=np.array([True]),
                    cargo_delivered=np.array([False]),
                    battery_ratios=np.array([0.8] * n_uavs),
                    step_count=step,
                    max_steps=1000,
                    boundary=np.array([[0, 100], [0, 100], [0, 50]]),
                )

                total_reward += r
                monitor.log(components, r)

                # 更新状态
                prev_positions = positions.copy()
                prev_actions = actions.copy()
                positions = positions + velocities * 0.1
                step += 1

                # 判断终止条件
                if step >= 1000 or np.linalg.norm(positions.mean(0) - np.array([50, 50, 20])) < 5:
                    done = True

            # 更新课程学习
            success = np.linalg.norm(positions.mean(0) - np.array([50, 50, 20])) < 5
            curriculum.update(success, step, collision=False)

        # 检查是否进入下一阶段
        if curriculum.should_advance():
            curriculum.advance_stage()

        # 定期打印监控报告
        if epoch % 10 == 0:
            monitor.print_report()


if __name__ == "__main__":
    training_loop_example()
```

---

## 9. 奖励监控与调试

### 9.1 关键监控指标

训练过程中应持续记录以下指标：

| 指标 | 计算方式 | 健康范围 | 异常处理 |
|------|---------|---------|---------|
| 总奖励均值 | 最近100episode均值 | 应持续上升 | 检查各分量权重 |
| 成功率 | 到达目标episode占比 | > 70% | 增大R_goal或塑形强度 |
| 碰撞率 | 碰撞episode占比 | < 5% | 增大P_collision权重 |
| 平均步数 | episode长度均值 | 递减 | 调整lambda_time |
| 各分量贡献 | 各分量占总奖励比例 | 见3.3节 | 调整权重平衡 |
| 队形偏差 | 平均队形偏差距离 | < 10m | 调整lambda_form |
| 能耗分布 | 平均能耗值 | 稳定 | 调整lambda_energy |

### 9.2 常见问题诊断

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| 智能体不尝试到达目标 | R_goal被时间惩罚淹没 | 增大R_goal或减小lambda_time |
| 智能体冒险冲向目标，频繁碰撞 | R_goal > \|P_collision\| | 增大P_collision或引入碰撞后额外惩罚 |
| 智能体保持距离但不移动 | 时间惩罚过小 | 增大lambda_time或减小P_proximity |
| 队形过紧或过松 | lambda_form设置不当 | 通过敏感度分析调整权重 |
| 训练初期无正向奖励 | 稀疏奖励问题严重 | 增大塑形强度或添加检查点奖励 |
| 训练后期性能下降 | 奖励hack（利用漏洞） | 检查奖励分量是否有非预期行为 |

---

## 10. 设计文档版本历史

| 版本 | 日期 | 变更内容 |
|------|------|----------|
| v1.0 | 2026-07-01 | 初始设计，8个基础奖励分量 |
| v2.0 | 2026-07-01 | 完整设计，增加归一化、课程学习、参数标定、完整代码框架 |
