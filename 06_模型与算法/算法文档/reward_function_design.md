# 无人机物流配送项目 -- 奖励函数设计文档

## 1. 总体设计思路

奖励函数采用**加权求和**的结构，将8个奖励分量组合为一个标量奖励信号。整体奖励公式如下：

$$
R_t = R_{goal} + P_{collision} + P_{proximity} + P_{formation} + P_{energy} + P_{time} + P_{smooth} + R_{delivery}
$$

其中：
- 前三项为**事件触发型奖励**（离散发生）
- 后五项为**连续型奖励/惩罚**（每步计算）
- 正奖励引导智能体完成目标，负奖励/惩罚约束安全与效率

---

## 2. 各奖励分量定义与数学公式

### 2.1 目标到达奖励 R_goal

**含义**：无人机到达目标点5米范围内时获得一次性正奖励。

**数学公式**：

$$
R_{goal} =
\begin{cases}
+200, & \text{if } \| \mathbf{p}_t - \mathbf{p}_{goal} \|_2 \leq 5 \\
0, & \text{otherwise}
\end{cases}
$$

**变量说明**：
- $\mathbf{p}_t \in \mathbb{R}^3$：无人机在时刻 $t$ 的三维位置
- $\mathbf{p}_{goal} \in \mathbb{R}^3$：目标点坐标
- $\| \cdot \|_2$：欧几里得距离

**设计考量**：
- 阈值5米考虑到GPS定位精度和实际降落场地大小
- 200的奖励值设置为较大的正向激励，确保智能体将"到达目标"作为首要任务
- 仅触发一次，防止智能体在目标点附近反复获取奖励

---

### 2.2 碰撞惩罚 P_collision

**含义**：发生碰撞时施加一次性大额惩罚。

**数学公式**：

$$
P_{collision} =
\begin{cases}
-200, & \text{if collision detected} \\
0, & \text{otherwise}
\end{cases}
$$

**碰撞检测条件**：

$$
\text{collision detected} \iff \exists i \neq j: \| \mathbf{p}_t^{(i)} - \mathbf{p}_t^{(j)} \|_2 < r_{safe}
$$

或者无人机与障碍物的距离小于安全半径：

$$
\text{collision detected} \iff \min_{\mathbf{o} \in \mathcal{O}} \| \mathbf{p}_t - \mathbf{o} \|_2 < r_{drone}
$$

**变量说明**：
- $r_{safe}$：碰撞判定半径（建议值1.0米）
- $r_{drone}$：无人机物理半径
- $\mathcal{O}$：环境中所有障碍物的集合

**设计考量**：
- 与 $R_{goal}$ 等值反向（+200 vs -200），确保安全与目标同等重要
- 碰撞后可选择终止episode或继续训练（建议初期终止，后期允许恢复）

---

### 2.3 间距惩罚 P_proximity

**含义**：当两架无人机距离小于10米安全距离时，施加连续惩罚。

**数学公式**：

$$
P_{proximity} = -100 \times \sum_{i<j} \max\left(0, 1 - \frac{\| \mathbf{p}_t^{(i)} - \mathbf{p}_t^{(j)} \|_2}{d_{safe}}\right)
$$

其中 $d_{safe} = 10$ 米。

**展开形式**（对于 $N$ 架无人机）：

$$
P_{proximity} = -100 \times \sum_{i=1}^{N} \sum_{j=i+1}^{N} \max\left(0, 1 - \frac{\| \mathbf{p}_t^{(i)} - \mathbf{p}_t^{(j)} \|_2}{10}\right)
$$

**惩罚强度随距离变化**：
- 距离 = 0米：惩罚系数 = 1.0，惩罚 = -100
- 距离 = 5米：惩罚系数 = 0.5，惩罚 = -50
- 距离 = 10米：惩罚系数 = 0.0，惩罚 = 0
- 距离 > 10米：无惩罚

**设计考量**：
- 采用线性衰减而非阶跃函数，提供平滑梯度信号
- 惩罚随距离连续变化，避免智能体在阈值边界震荡
- 当 $N=1$（单机）时，此项恒为0

---

### 2.4 队形保持惩罚 P_formation

**含义**：惩罚无人机偏离期望队形位置的程度。

**数学公式**：

$$
P_{formation} = -\lambda_{form} \times \frac{1}{N} \sum_{i=1}^{N} \| \mathbf{p}_t^{(i)} - \mathbf{p}_{desired}^{(i)} \|_2
$$

其中 $\lambda_{form} = 1.0$。

**期望队形位置计算**（以编队中心为基准）：

$$
\mathbf{p}_{desired}^{(i)} = \mathbf{p}_{center} + \mathbf{offset}^{(i)}
$$

$$
\mathbf{p}_{center} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{p}_t^{(i)}
$$

**常见队形偏移定义**：
- **V字队形**：$\mathbf{offset}^{(i)} = (i \cdot d_x, \; |i| \cdot d_y, \; 0)$
- **直线队形**：$\mathbf{offset}^{(i)} = (0, \; i \cdot d_y, \; 0)$
- **菱形队形**：根据位置索引查表获取偏移量

**变量说明**：
- $\lambda_{form} = 1.0$：队形偏差惩罚权重
- $N$：编队中无人机数量
- $d_x, d_y$：编队间距参数

**设计考量**：
- 归一化处理（除以N）使惩罚不随编队规模变化
- 可根据任务阶段动态调整：巡航阶段强调队形，配送阶段放宽约束
- 当 $N=1$ 时此项恒为0

---

### 2.5 能耗惩罚 P_energy

**含义**：对无人机每步的能量消耗施加惩罚，鼓励节能飞行。

**数学公式**：

$$
P_{energy} = -\lambda_{energy} \times \frac{1}{N} \sum_{i=1}^{N} E_t^{(i)}
$$

其中 $\lambda_{energy} = 0.5$。

**单机能耗模型**：

$$
E_t^{(i)} = \alpha \cdot \| \mathbf{a}_t^{(i)} \|_2^2 + \beta \cdot \| \mathbf{v}_t^{(i)} \|_2^2 + \gamma \cdot | h_t^{(i)} - h_{t-1}^{(i)} |
$$

**简化版本**（仅考虑动作能耗）：

$$
E_t^{(i)} = \| \mathbf{a}_t^{(i)} \|_2^2
$$

其中 $\mathbf{a}_t^{(i)} \in \mathbb{R}^3$ 为无人机 $i$ 在时刻 $t$ 的加速度/推力指令。

**变量说明**：
- $\lambda_{energy} = 0.5$：能耗惩罚权重
- $\alpha = 1.0$：加速度能耗系数
- $\beta = 0.1$：速度维持能耗系数
- $\gamma = 2.0$：高度变化能耗系数（爬升耗能）

**设计考量**：
- 二次方形式使大幅动作受到更重惩罚
- 可引入电池剩余电量作为自适应系数：电量低时增大 $\lambda_{energy}$
- 鼓励巡航飞行而非急加速/急减速

---

### 2.6 时间惩罚 P_time

**含义**：每步施加固定时间成本，鼓励高效完成任务。

**数学公式**：

$$
P_{time} = -\lambda_{time} = -2
$$

**累积时间惩罚**（episode层面）：

$$
P_{time}^{total} = -\lambda_{time} \times T_{episode} = -2 \times T_{episode}
$$

**变量说明**：
- $\lambda_{time} = 2$：每步时间惩罚值
- $T_{episode}$：episode总步数

**设计考量**：
- 固定惩罚值使得智能体在最优路径与最短路径间做权衡
- 与能耗惩罚配合：时间惩罚鼓励快速到达，能耗惩罚约束不要过于激进
- 可设置最大episode步数（如1000步），超过则强制终止并施加额外惩罚

---

### 2.7 平滑惩罚 P_smooth

**含义**：惩罚动作的突变，鼓励平稳飞行。

**数学公式**：

$$
P_{smooth} = -\lambda_{smooth} \times \frac{1}{N} \sum_{i=1}^{N} \| \mathbf{a}_t^{(i)} - \mathbf{a}_{t-1}^{(i)} \|_2^2
$$

其中 $\lambda_{smooth} = 0.1$。

**变量说明**：
- $\lambda_{smooth} = 0.1$：动作变化惩罚权重
- $\mathbf{a}_t^{(i)} - \mathbf{a}_{t-1}^{(i)}$：连续两步的动作差（jerk）

**设计考量**：
- 惩罚"加加速度"（jerk），这是影响飞行平稳性的关键指标
- 权重较小（0.1），作为辅助约束而非主要信号
- 有助于减少实际飞行中的机械磨损和货物晃动
- 首步（$t=0$）时此项为0

---

### 2.8 配送完成奖励 R_delivery

**含义**：完成单个配送任务时获得奖励。

**数学公式**：

$$
R_{delivery} =
\begin{cases}
+50, & \text{if delivery}_k \text{ completed} \\
0, & \text{otherwise}
\end{cases}
$$

**配送完成判定**：

$$
\text{delivery}_k \text{ completed} \iff \| \mathbf{p}_t - \mathbf{p}_{dest}^{(k)} \|_2 \leq d_{drop} \; \land \; \text{cargo}_k = \text{loaded}
$$

**变量说明**：
- $\mathbf{p}_{dest}^{(k)}$：第 $k$ 个配送任务的目的地坐标
- $d_{drop}$：投放/降落判定距离（建议值3米）
- $\text{cargo}_k$：货物状态标志

**设计考量**：
- 50的奖励低于到达奖励（200），引导智能体优先完成所有配送而非单一任务
- 多任务场景下，总配送奖励 = $50 \times K$（K为任务数量）
- 可设置递增奖励：后续配送任务给予更高奖励，鼓励完成所有任务

---

## 3. 完整奖励函数汇总

### 3.1 总奖励公式

$$
R_t = \underbrace{200 \cdot \mathbb{1}[\text{goal reached}]}_{R_{goal}}
- \underbrace{200 \cdot \mathbb{1}[\text{collision}]}_{P_{collision}}
- \underbrace{100 \cdot \sum_{i<j} \max\left(0, 1 - \frac{d_{ij}}{10}\right)}_{P_{proximity}}
- \underbrace{1.0 \cdot \frac{1}{N}\sum_{i} \| \mathbf{p}_t^{(i)} - \mathbf{p}_{des}^{(i)} \|}_{P_{formation}}
- \underbrace{0.5 \cdot \frac{1}{N}\sum_{i} E_t^{(i)}}_{P_{energy}}
- \underbrace{2}_{P_{time}}
- \underbrace{0.1 \cdot \frac{1}{N}\sum_{i} \| \Delta\mathbf{a}_t^{(i)} \|^2}_{P_{smooth}}
+ \underbrace{50 \cdot \mathbb{1}[\text{delivery done}]}_{R_{delivery}}
$$

### 3.2 参数汇总表

| 分量 | 符号 | 类型 | 基准值 | 触发条件 |
|------|------|------|--------|----------|
| 目标到达奖励 | $R_{goal}$ | 事件触发型 | +200 | 距目标 ≤ 5m |
| 碰撞惩罚 | $P_{collision}$ | 事件触发型 | -200 | 碰撞检测触发 |
| 间距惩罚 | $P_{proximity}$ | 连续型 | -100 (max) | 距离 < 10m |
| 队形保持惩罚 | $\lambda_{form}$ | 连续型 | -1.0/step | 偏差存在时 |
| 能耗惩罚 | $\lambda_{energy}$ | 连续型 | -0.5/step | 持续存在 |
| 时间惩罚 | $\lambda_{time}$ | 连续型 | -2.0/step | 每步固定 |
| 平滑惩罚 | $\lambda_{smooth}$ | 连续型 | -0.1/step | 动作变化时 |
| 配送完成奖励 | $R_{delivery}$ | 事件触发型 | +50 | 货物送达时 |

### 3.3 奖励量级分析

在一个典型的episode中（假设500步完成配送）：

| 分量 | 单次量级 | 预计触发次数 | 预计总贡献 |
|------|----------|-------------|-----------|
| $R_{goal}$ | +200 | 1 | +200 |
| $P_{collision}$ | -200 | 0（理想情况） | 0 |
| $P_{proximity}$ | -5 ~ -100 | ~20次 | -100 ~ -500 |
| $P_{formation}$ | -1 ~ -5/step | 500步 | -500 ~ -2500 |
| $P_{energy}$ | -0.1 ~ -2/step | 500步 | -50 ~ -1000 |
| $P_{time}$ | -2/step | 500步 | -1000 |
| $P_{smooth}$ | -0.01 ~ -0.5/step | 500步 | -5 ~ -250 |
| $R_{delivery}$ | +50 | 1 | +50 |

**关键观察**：
- 目标奖励（+200）+ 配送奖励（+50）= +250（正向锚点）
- 时间惩罚（-1000）是最主要的负向信号，确保效率
- 队形惩罚可能量级较大，需要根据实际训练效果调整

---

## 4. 参数标定方法

### 4.1 分层标定策略

采用"从简到繁、逐步叠加"的标定策略：

**第一层：核心任务（R_goal + P_collision + P_time）**
- 仅使用这3个分量进行初始训练
- 标定 $R_{goal}$ 和 $P_{collision}$ 的相对比例，确保安全与目标平衡
- 标定 $\lambda_{time}$ 使episode长度在合理范围内

**第二层：安全约束（+ P_proximity）**
- 加入间距惩罚，观察多机协作行为
- 调整阈值和权重，避免过度保守

**第三层：编队与效率（+ P_formation + P_energy + R_delivery）**
- 加入队形、能耗和配送奖励
- 平衡各项权重

**第四层：平滑性（+ P_smooth）**
- 最后加入平滑惩罚作为微调

### 4.2 自动标定方法

#### 4.2.1 基于归一化的权重平衡

对每个分量计算其在训练初期的统计量，归一化后加权：

$$
\hat{R}_i = \frac{R_i - \mu_i}{\sigma_i}
$$

$$
R_{total} = \sum_{i} w_i \cdot \hat{R}_i
$$

其中 $w_i$ 为人工设定的相对重要性权重。

#### 4.2.2 基于敏感度分析的标定

对每个参数进行网格搜索：

1. 固定其他参数，变化目标参数 $\lambda \in \{0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0\}$
2. 训练N个episode，记录关键指标：
   - 成功率（到达目标的比例）
   - 平均episode长度
   - 碰撞率
   - 队形偏差均值
3. 选择使综合指标最优的参数值

#### 4.2.3 基于奖励分量贡献比的调整

设定目标贡献比：

| 分量 | 目标贡献占比 |
|------|-------------|
| $R_{goal}$ | 25% |
| $P_{collision}$ | 25% |
| $P_{proximity}$ | 15% |
| $P_{formation}$ | 10% |
| $P_{energy}$ | 10% |
| $P_{time}$ | 10% |
| $P_{smooth}$ | 3% |
| $R_{delivery}$ | 2% |

训练过程中监控各分量实际贡献，动态调整权重使其趋近目标比例。

### 4.3 推荐初始参数配置

```python
reward_config = {
    # 事件触发奖励
    "R_goal": 200.0,
    "R_delivery": 50.0,
    "P_collision": -200.0,

    # 安全约束
    "P_proximity_max": -100.0,
    "d_safe": 10.0,           # 安全距离阈值（米）

    # 连续惩罚权重
    "lambda_form": 1.0,
    "lambda_energy": 0.5,
    "lambda_time": 2.0,
    "lambda_smooth": 0.1,

    # 队形参数
    "formation_type": "V",     # V字/直线/菱形
    "formation_spacing": 8.0,  # 编队间距（米）

    # 任务参数
    "goal_threshold": 5.0,     # 目标到达判定距离（米）
    "drop_threshold": 3.0,     # 配送完成判定距离（米）
    "collision_radius": 1.0,   # 碰撞判定半径（米）

    # Episode参数
    "max_steps": 1000,
    "max_penalty_on_timeout": -100.0,
}
```

---

## 5. 稀疏奖励问题处理

稀疏奖励是本项目的核心挑战之一：$R_{goal}$ 和 $R_{delivery}$ 只在任务完成时触发，在长episode中智能体难以建立因果关联。

### 5.1 奖励塑形（Reward Shaping）

#### 5.1.1 基于势能的奖励塑形（Potential-Based Reward Shaping）

根据 Ng et al. (1999) 的理论，添加基于势函数的塑形奖励不会改变最优策略：

$$
F(s, a, s') = \gamma \Phi(s') - \Phi(s)
$$

**势函数设计**（基于到目标的距离）：

$$
\Phi(s) = -c \cdot \| \mathbf{p}_t - \mathbf{p}_{goal} \|_2
$$

**塑形奖励**：

$$
F(s, a, s') = -c \cdot \gamma \cdot \| \mathbf{p}_{t+1} - \mathbf{p}_{goal} \|_2 + c \cdot \| \mathbf{p}_t - \mathbf{p}_{goal} \|_2
$$

其中 $c > 0$ 为塑形强度系数（建议值 $c = 0.5$）。

**物理含义**：
- 靠近目标：$F > 0$（正奖励）
- 远离目标：$F < 0$（负奖励）
- 不变：$F = 0$

**多任务势函数**：

对于K个配送任务，采用最大剩余任务距离：

$$
\Phi(s) = -c \cdot \min_{k \in \text{remaining}} \| \mathbf{p}_t - \mathbf{p}_{dest}^{(k)} \|_2
$$

#### 5.1.2 基于进度的奖励塑形

定义任务进度指标 $\eta_t \in [0, 1]$：

$$
\eta_t = 1 - \frac{\| \mathbf{p}_t - \mathbf{p}_{goal} \|_2}{\| \mathbf{p}_0 - \mathbf{p}_{goal} \|_2}
$$

进度奖励：

$$
R_{progress} = c_p \cdot (\eta_t - \eta_{t-1})
$$

其中 $c_p = 5.0$ 为进度奖励系数。

#### 5.1.3 基于中间目标的奖励

设置路径中间检查点，提供阶段性奖励：

$$
R_{waypoint} = \sum_{w=1}^{W} r_w \cdot \mathbb{1}[\| \mathbf{p}_t - \mathbf{p}_{waypoint}^{(w)} \|_2 \leq d_w]
$$

- $W$：检查点数量（建议3-5个）
- $r_w = 10$：每个检查点的奖励
- $d_w = 5$：检查点判定距离

**检查点生成方法**：
- 均匀插值：在起始点和目标点之间等距插入
- 路径规划辅助：使用A*等算法生成参考路径，在路径上设置检查点

### 5.2 课程学习（Curriculum Learning）

#### 5.2.1 环境难度递进

分4个阶段逐步增加任务难度：

**阶段1：单机简单导航**
- 场景：单架无人机，无障碍物，单目标点
- 目标：学会基本的飞行和到达目标
- 训练步数：100K steps
- 稀疏奖励处理：仅依赖势能塑形

**阶段2：单机复杂环境**
- 场景：单架无人机，有静态障碍物，单目标点
- 目标：学会避障导航
- 训练步数：200K steps
- 新增：$P_{proximity}$（与障碍物的距离惩罚）

**阶段3：多机协作**
- 场景：2-3架无人机，有障碍物，单配送任务
- 目标：学会协调飞行和队形保持
- 训练步数：300K steps
- 新增：$P_{formation}$、完整的 $P_{proximity}$

**阶段4：完整配送任务**
- 场景：多架无人机，完整环境，多配送任务
- 目标：完成端到端配送
- 训练步数：持续训练
- 新增：$R_{delivery}$、$P_{energy}$、$P_{smooth}$

#### 5.2.2 任务距离递进

在每个阶段内，逐步增加起始点到目标点的距离：

$$
d_{task}^{(epoch)} = d_{min} + (d_{max} - d_{min}) \cdot \min\left(1, \frac{epoch}{T_{ramp}}\right)
$$

- $d_{min} = 10$ 米（近距离任务）
- $d_{max} = 500$ 米（远距离任务）
- $T_{ramp} = 50$ 个epoch（递增周期）

#### 5.2.3 奖励塑形强度递减

随着训练进行，逐步降低塑形奖励的权重：

$$
c^{(epoch)} = c_{init} \cdot \left(\frac{T_{decay}}{T_{decay} + epoch}\right)
$$

- $c_{init} = 2.0$：初始塑形强度
- $T_{decay} = 20$：衰减时间常数

**目标**：训练后期让智能体主要依赖原始稀疏奖励，确保策略的最优性。

#### 5.2.4 课程学习触发条件

自动判断是否进入下一阶段：

```python
def should_advance_stage(current_stage, metrics):
    """
    进入下一阶段的条件（全部满足）：
    1. 最近100个episode的成功率 > 80%
    2. 平均episode长度 < 阶段阈值
    3. 碰撞率 < 5%
    """
    if current_stage >= max_stage:
        return False

    success_rate = metrics["success_rate_recent_100"]
    avg_length = metrics["avg_episode_length"]
    collision_rate = metrics["collision_rate"]

    thresholds = {
        1: {"success": 0.80, "max_length": 200, "max_collision": 0.05},
        2: {"success": 0.75, "max_length": 400, "max_collision": 0.05},
        3: {"success": 0.70, "max_length": 600, "max_collision": 0.03},
    }

    t = thresholds[current_stage]
    return (success_rate > t["success"] and
            avg_length < t["max_length"] and
            collision_rate < t["max_collision"])
```

### 5.3 探索增强策略

#### 5.3.1 内在好奇心模块（ICM）

添加基于预测误差的好奇心奖励：

$$
R_{icm} = \eta_{icm} \cdot \| \hat{\phi}(s_{t+1}) - \phi(s_{t+1}) \|_2^2
$$

- $\phi(s)$：状态编码器
- $\hat{\phi}(s_{t+1})$：基于 $(s_t, a_t)$ 的预测
- $\eta_{icm} = 0.1$：好奇心权重

#### 5.3.2 epsilon-贪心探索增强

在训练初期增大探索概率：

$$
\epsilon^{(epoch)} = \epsilon_{final} + (\epsilon_{init} - \epsilon_{final}) \cdot e^{-epoch / T_{\epsilon}}
$$

- $\epsilon_{init} = 0.3$
- $\epsilon_{final} = 0.01$
- $T_{\epsilon} = 30$

### 5.4 稀疏奖励处理策略汇总

| 策略 | 适用阶段 | 预期效果 | 实现复杂度 |
|------|---------|---------|-----------|
| 势能塑形 | 全阶段 | 提供密集梯度信号 | 低 |
| 进度奖励 | 全阶段 | 强化正向进展 | 低 |
| 中间检查点 | 阶段1-2 | 分解长期任务 | 中 |
| 课程学习（环境） | 全阶段 | 逐步提升能力 | 中 |
| 课程学习（距离） | 阶段3-4 | 渐进式挑战 | 低 |
| 塑形衰减 | 阶段3-4 | 保证策略最优性 | 低 |
| ICM好奇心 | 阶段1-2 | 增强初期探索 | 高 |

---

## 6. 奖励函数实现代码

```python
import numpy as np
from typing import Dict, List, Tuple, Optional


class RewardCalculator:
    """无人机物流配送奖励计算器"""

    def __init__(self, config: dict):
        self.config = config

        # 奖励参数
        self.R_goal = config.get("R_goal", 200.0)
        self.R_delivery = config.get("R_delivery", 50.0)
        self.P_collision = config.get("P_collision", -200.0)
        self.P_proximity_max = config.get("P_proximity_max", -100.0)

        # 权重参数
        self.lambda_form = config.get("lambda_form", 1.0)
        self.lambda_energy = config.get("lambda_energy", 0.5)
        self.lambda_time = config.get("lambda_time", 2.0)
        self.lambda_smooth = config.get("lambda_smooth", 0.1)

        # 阈值参数
        self.d_safe = config.get("d_safe", 10.0)
        self.goal_threshold = config.get("goal_threshold", 5.0)
        self.drop_threshold = config.get("drop_threshold", 3.0)
        self.collision_radius = config.get("collision_radius", 1.0)

        # 塑形参数
        self.shaping_c = config.get("shaping_c", 0.5)
        self.gamma = config.get("gamma", 0.99)

    def calculate(
        self,
        positions: np.ndarray,       # (N, 3) 无人机位置
        prev_positions: np.ndarray,   # (N, 3) 上一步位置
        actions: np.ndarray,         # (N, 3) 当前动作
        prev_actions: np.ndarray,    # (N, 3) 上一步动作
        goal_pos: np.ndarray,        # (3,) 目标位置
        goal_positions: np.ndarray,  # (K, 3) 配送目标位置
        formation_offsets: np.ndarray, # (N, 3) 期望队形偏移
        obstacles: np.ndarray,       # (M, 4) 障碍物 [x,y,z,radius]
        cargo_loaded: np.ndarray,    # (K,) 货物是否装载
        cargo_delivered: np.ndarray, # (K,) 货物是否已送达
    ) -> Tuple[float, Dict[str, float]]:

        N = positions.shape[0]
        rewards = {}

        # 1. 目标到达奖励
        dist_to_goal = np.linalg.norm(positions.mean(axis=0) - goal_pos)
        R_goal = self.R_goal if dist_to_goal <= self.goal_threshold else 0.0
        rewards["R_goal"] = R_goal

        # 2. 碰撞惩罚
        collision = self._check_collision(positions, obstacles)
        P_collision = self.P_collision if collision else 0.0
        rewards["P_collision"] = P_collision

        # 3. 间距惩罚
        P_proximity = self._proximity_penalty(positions)
        rewards["P_proximity"] = P_proximity

        # 4. 队形保持惩罚
        center = positions.mean(axis=0)
        desired_pos = center + formation_offsets[:N]
        formation_error = np.linalg.norm(positions - desired_pos, axis=1).mean()
        P_formation = -self.lambda_form * formation_error
        rewards["P_formation"] = P_formation

        # 5. 能耗惩罚
        energy = np.linalg.norm(actions, axis=1) ** 2
        P_energy = -self.lambda_energy * energy.mean()
        rewards["P_energy"] = P_energy

        # 6. 时间惩罚
        P_time = -self.lambda_time
        rewards["P_time"] = P_time

        # 7. 平滑惩罚
        action_diff = actions - prev_actions
        P_smooth = -self.lambda_smooth * np.linalg.norm(action_diff, axis=1).mean() ** 2
        rewards["P_smooth"] = P_smooth

        # 8. 配送完成奖励
        R_delivery = 0.0
        for k in range(len(goal_positions)):
            dist = np.linalg.norm(positions.mean(axis=0) - goal_positions[k])
            if dist <= self.drop_threshold and cargo_loaded[k] and not cargo_delivered[k]:
                R_delivery += self.R_delivery
        rewards["R_delivery"] = R_delivery

        # 势能塑形奖励
        if prev_positions is not None:
            prev_dist = np.linalg.norm(prev_positions.mean(axis=0) - goal_pos)
            curr_dist = dist_to_goal
            shaping = self.shaping_c * (self.gamma * (-curr_dist) - (-prev_dist))
            rewards["shaping"] = shaping
        else:
            rewards["shaping"] = 0.0

        total_reward = sum(rewards.values())
        return total_reward, rewards

    def _check_collision(self, positions, obstacles):
        """检测碰撞"""
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                if np.linalg.norm(positions[i] - positions[j]) < self.collision_radius:
                    return True
            for obs in obstacles:
                obs_pos = obs[:3]
                obs_r = obs[3]
                if np.linalg.norm(positions[i] - obs_pos) < (self.collision_radius + obs_r):
                    return True
        return False

    def _proximity_penalty(self, positions):
        """计算间距惩罚"""
        penalty = 0.0
        N = len(positions)
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.linalg.norm(positions[i] - positions[j])
                penalty += max(0, 1.0 - dist / self.d_safe)
        return self.P_proximity_max * penalty
```

---

## 7. 奖励监控与调试

### 7.1 关键监控指标

训练过程中应持续记录以下指标：

| 指标 | 计算方式 | 健康范围 | 异常处理 |
|------|---------|---------|---------|
| 总奖励均值 | 最近100episode均值 | 应持续上升 | 检查各分量权重 |
| 成功率 | 到达目标episode占比 | > 70% | 增大R_goal或塑形强度 |
| 碰撞率 | 碰撞episode占比 | < 5% | 增大P_collision权重 |
| 平均步数 | episode长度均值 | 递减 | 调整lambda_time |
| 各分量贡献 | 各分量占总奖励比例 | 见3.3节 | 调整权重平衡 |

### 7.2 常见问题诊断

**问题1：智能体不尝试到达目标**
- 原因：$R_{goal}$ 被时间惩罚淹没
- 解决：增大 $R_{goal}$ 或减小 $\lambda_{time}$

**问题2：智能体冒险冲向目标，频繁碰撞**
- 原因：$R_{goal} > |P_{collision}|$ 或探索不足
- 解决：增大 $P_{collision}$ 或引入碰撞后的额外惩罚

**问题3：智能体保持距离但不移动**
- 原因：时间惩罚过小，保守策略最优
- 解决：增大 $\lambda_{time}$ 或减小 $P_{proximity}$

**问题4：队形过紧或过松**
- 原因：$\lambda_{form}$ 设置不当
- 解决：通过敏感度分析调整权重

---

## 8. 设计文档版本信息

- **版本**：v1.0
- **日期**：2026-07-01
- **状态**：初始设计
- **后续迭代**：根据实际训练效果调整参数和策略
