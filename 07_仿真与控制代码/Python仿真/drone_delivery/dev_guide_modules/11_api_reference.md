"""
职责 / Responsibility: 完整API参考文档 -- 环境、Agent、特征处理、算法核心、工作流、附录
参考 / Reference: agent_q_learning/全部源码, environment/全部源码, feature/definition.py, model/model.py
领域适配 / Domain Adaptation: 代码级精确API文档，四段式格式 (签名/参数/返回/不变量)
"""

## 11. API 参考手册 / API Reference Manual

---

### 11.1 环境 API / Environment API

#### 11.1.1 Simple2DDeliveryEnv（L1）

**源码 / Source**: `environment/simple_delivery_env.py`

```python
class Simple2DDeliveryEnv(gym.Env):
    """问题层级一：2D固定商家配送环境 / Level 1: 2D fixed merchant delivery environment"""

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 render_mode: Optional[str] = None): ...
```

**核心方法 / Core Methods**

##### reset()

```python
def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None
          ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
```

| 参数 / Parameter | 类型 / Type | 说明 / Description |
|---|---|---|
| `seed` | `Optional[int]` | 随机种子 / Random seed |
| `options` | `Optional[Dict]` | 保留参数 / Reserved |

| 返回 / Returns | 类型 / Type | 说明 / Description |
|---|---|---|
| `obs` | `Dict[str, np.ndarray]` | 含 `position:(2,), order_status:(5,), carrying:int` |
| `info` | `Dict[str, Any]` | 含 `position, completed, carrying, current_order, steps, depot_reached` |

**不变量 / Invariants**:
- `obs["position"]` 初始化为 `depot`（默认 [30, 10]）
- `obs["order_status"]` 全零（5个订单均未完成）
- `obs["carrying"]` 为 0（空手状态）

##### step()

```python
def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
```

| 参数 / Parameter | 类型 / Type | 说明 / Description |
|---|---|---|
| `action` | `int` | 动作索引：0=上UP, 1=下DOWN, 2=左LEFT, 3=右RIGHT |

| 返回 / Returns | 类型 / Type | 说明 / Description |
|---|---|---|
| `obs` | `Dict` | 同 `reset` 返回格式 |
| `reward` | `float` | 奖励值 / Reward value |
| `terminated` | `bool` | 碰撞或全部完成时为 `True` |
| `truncated` | `bool` | 达到 `max_steps`（500）时为 `True` |
| `info` | `Dict` | 同 `reset` 的 info |

**奖励表 / Reward Table**:

| 事件 / Event | 奖励 / Reward |
|---|---|
| 完成一单配送 | +100.0 |
| 全部完成并返回配送中心 | +500.0 |
| 每步时间惩罚 | -1.0 |
| 碰撞障碍物 | -200.0 |

##### get_state_key()

```python
def get_state_key(self) -> int
```

| 返回 / Returns | 类型 / Type | 说明 / Description |
|---|---|---|
| 唯一状态键 | `int` | 编码方式：`(pos_idx << 6) \| (status << 1) \| carrying` |

**编码空间 / Encoding Space**: `position(40x40=1600) * order_status(2^5=32) * carrying(2) = 102,400`

##### render()

```python
def render(self) -> Optional[np.ndarray]
```

以 ASCII 网格形式渲染当前地图状态。`·` = 空地, `█` = 障碍物, `D` = 配送中心, `★` = 商家, `●` = 无人机。

##### close()

```python
def close(self) -> None
```

清理渲染资源。无参数，无返回值。

---

#### 11.1.2 Random2DDeliveryEnv（L2）

**源码 / Source**: `environment/random_delivery_env.py`

```python
class Random2DDeliveryEnv(gym.Env):
    """问题层级二：2D随机商家配送环境 / Level 2: 2D random merchant delivery environment"""

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 render_mode: Optional[str] = None): ...
```

**与 L1 的关键差异 / Key Differences from L1**:

| 特性 / Feature | L1 Simple2D | L2 Random2D |
|---|---|---|
| 动作空间 | Discrete(4) | Discrete(8) |
| 订单 | 5个固定 | 4-8个随机生成 |
| 观测格式 | `{position, order_status, carrying}` | `{position, vector(50,)}` |
| 携带限制 | 1单 | 最多2单 |
| 时间窗口 | 无 | 有（超时扣分） |
| `max_steps` | 500 | 800 |

##### reset()

```python
def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None
          ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
```

返回 `obs` 含：
- `position: (2,) int32` — 无人机坐标
- `vector: (50,) float32` — 归一化特征向量（位置2维 + 订单6维 x 4~8 + 携带+步数2维）

**不变量 / Invariants**:
- 每次 `reset` 调用 `_generate_orders(rng)` 生成4-8个随机订单
- 每个订单包含 `merchant, delivery_point, status, deadline, pickup_time, delivery_time`

##### step()

```python
def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
```

| 参数 / Parameter | 类型 / Type | 说明 / Description |
|---|---|---|
| `action` | `int` | 0-7: 0=上, 1=下, 2=左, 3=右, 4=左上, 5=右上, 6=左下, 7=右下 |

**奖励表 / Reward Table**:

| 事件 / Event | 奖励 / Reward |
|---|---|
| 按时完成配送 | +100.0 |
| 时间窗口内完成奖励 | +50.0 |
| 超时后完成（基础分） | +30.0 |
| 订单超时未完成 | -30.0 |
| 多单携带鼓励 | +0.5/step |
| 全部完成并返回 | +500.0 |
| 每步时间惩罚 | -1.0 |
| 碰撞（终止） | -200.0 |

---

### 11.2 Agent 接口 API / Agent Interface API

所有Agent实现以下9个标准方法，遵循Kaiwu（腾讯开悟）接口规范。
All Agents implement these 9 standard methods following Kaiwu interface specification.

**源码示例 / Source Example**: `drone_delivery/agent_q_learning/agent.py`

#### 11.2.1 `__init__`

```python
def __init__(self, agent_type: str = "player", device: Any = None,
             logger: Any = None, monitor: Any = None) -> None:
```

| 参数 / Parameter | 类型 / Type | 默认值 / Default | 说明 / Description |
|---|---|---|---|
| `agent_type` | `str` | `"player"` | 智能体类型 / Agent type |
| `device` | `Any` | `None` | 计算设备（表格型RL固定用CPU） |
| `logger` | `Any` | `None` | 日志记录器 / Logger instance |
| `monitor` | `Any` | `None` | 监控实例 / Monitor instance |

**不变量 / Invariants**:
- 创建 `SparseQTable(action_size=4)` 模型
- 创建 `Algorithm(gamma, learning_rate, action_size)` 算法实例
- `Algorithm.Q` 与 `self.model` 指向同一个 SparseQTable 实例
- `epsilon = Config.EPSILON_START`（默认 1.0）

#### 11.2.2 `reset`

```python
def reset(self, env_obs: Optional[Dict[str, Any]] = None) -> None:
```

| 参数 / Parameter | 类型 / Type | 说明 / Description |
|---|---|---|
| `env_obs` | `Optional[Dict]` | 环境初始观测（当前未使用） |

| 返回 / Returns | 说明 / Description |
|---|---|
| `None` | 无返回值 |

**不变量 / Invariants**:
- `self.episodes` 递增 1
- 调用 `_decay_epsilon()`: `epsilon = max(epsilon_min, epsilon * epsilon_decay)`

#### 11.2.3 `predict`

```python
def predict(self, list_obs_data) -> ActData:
```

| 参数 / Parameter | 类型 / Type | 说明 / Description |
|---|---|---|
| `list_obs_data` | `list[ObsData]` | 观测数据列表（使用第一个元素） |

| 返回 / Returns | 类型 / Type | 说明 / Description |
|---|---|---|
| `ActData` | `ActData` | 包含 `act: int` 和 `val: None` |

**不变量 / Invariants**:
- 概率 `epsilon`：从 `legal_actions` 中均匀随机选择（探索）
- 概率 `1-epsilon`：选择 `Q[state_key]` 中合法动作的最大值（利用），随机打破平局
- 每次调用 `train_steps += 1` 并衰减 `epsilon`

#### 11.2.4 `exploit`

```python
def exploit(self, env_obs) -> int:
```

| 参数 / Parameter | 类型 / Type | 说明 / Description |
|---|---|---|
| `env_obs` | `Dict` | 环境观测字典 / Environment observation dict |

| 返回 / Returns | 类型 / Type | 说明 / Description |
|---|---|---|
| `action` | `int` | 可执行的动作索引（0-3） |

**不变量 / Invariants**:
- 确定性贪心策略，**不使用**探索噪声
- 始终选择 Q 值最大的合法动作
- 内部调用 `observation_process()` 和 `action_process()`

#### 11.2.5 `learn`

```python
def learn(self, list_sample_data) -> dict:
```

| 参数 / Parameter | 类型 / Type | 说明 / Description |
|---|---|---|
| `list_sample_data` | `list[SampleData]` 或 `list[dict]` | 样本数据列表 |

| 返回 / Returns | 类型 / Type | 说明 / Description |
|---|---|---|
| 指标字典 | `dict` | 含 `td_error, lr, epsilon, train_steps, q_table_size, num_samples` |

**不变量 / Invariants**:
- 空输入返回空字典 `{}`
- 委托给 `self.algorithm.learn(list_sample_data)`
- 内部执行Q-learning更新: `Q(s,a) += lr * [r + γ * max_a' Q(s',a') * (1-done) - Q(s,a)]`

#### 11.2.6 `observation_process`

```python
def observation_process(self, env_obs) -> ObsData:
```

| 参数 / Parameter | 类型 / Type | 说明 / Description |
|---|---|---|
| `env_obs` | `Dict` | 环境观测字典 |

| 返回 / Returns | 类型 / Type | 说明 / Description |
|---|---|---|
| `ObsData` | `ObsData` | 含 `feature: int`（状态键）, `legal_actions: list[int]` |

**不变量 / Invariants**:
- `feature` 值域: `[0, 102399]`（17位编码）
- `legal_actions` 是 `[0,1,2,3]` 的子集
- 障碍物掩码在首次调用时惰性构建

#### 11.2.7 `action_process`

```python
def action_process(self, act_data) -> int:
```

| 参数 / Parameter | 类型 / Type | 说明 / Description |
|---|---|---|
| `act_data` | `ActData` | 动作数据对象 |

| 返回 / Returns | 类型 / Type | 说明 / Description |
|---|---|---|
| `action` | `int` | 可执行动作索引，范围 `[0, ACTION_SIZE-1]` |

**不变量 / Invariants**:
- `act_data.act` 为 `None` 时返回 0
- 钳位到 `[0, 3]` 范围

#### 11.2.8 `reward_shaping`

```python
def reward_shaping(self, reward, env_obs=None, prev_env_obs=None, action=None) -> float:
```

| 参数 / Parameter | 类型 / Type | 说明 / Description |
|---|---|---|
| `reward` | `float` | 原始环境奖励 |
| `env_obs` | `Optional[Dict]` | 当前观测（可选） |
| `prev_env_obs` | `Optional[Dict]` | 前一观测（可选） |
| `action` | `Optional[int]` | 执行的动作（可选） |

| 返回 / Returns | 类型 / Type | 说明 / Description |
|---|---|---|
| `shaped_reward` | `float` | 塑形后奖励 |

**不变量 / Invariants**:
- 当前为**直通版本**: `return float(reward) if reward is not None else 0.0`
- 保留了未来扩展入口（距离塑形、原地不动惩罚等）

#### 11.2.9 `save_model` / `load_model`

```python
def save_model(self, path=None, id="latest") -> None:
def load_model(self, path=None, id="latest") -> None:
```

| 参数 / Parameter | 类型 / Type | 默认值 / Default | 说明 / Description |
|---|---|---|---|
| `path` | `Optional[str]` | `"outputs/models"` | 保存/加载目录 |
| `id` | `str` | `"latest"` | 检查点标识符 |

**保存文件格式 / Save File Format**:
- 文件名: `{algo}_{id}.npy`（如 `q_learning_latest.npy`）
- 格式: `numpy .npy`（`allow_pickle=True`）
- 包含: Q表字典 + `epsilon` + `train_steps` + `episodes`

**不变量 / Invariants**:
- `save_model`: 创建目录（如果不存在），保存 `.npy` 文件
- `load_model`: 文件不存在时静默回退到空白Q表，不抛异常

---

### 11.3 数据结构 API / Data Structure API

**源码 / Source**: `drone_delivery/agent_q_learning/feature/definition.py`

#### 11.3.1 ObsData

```python
class ObsData:
    def __init__(self, feature: Optional[int] = None,
                 legal_actions: Optional[List[int]] = None,
                 raw_obs: Optional[Dict[str, Any]] = None):
```

| 属性 / Attribute | 类型 / Type | 说明 / Description |
|---|---|---|
| `feature` | `Optional[int]` | 状态键（Q表索引）/ State key |
| `legal_actions` | `List[int]` | 合法动作列表，默认 `[0,1,2,3]` |
| `raw_obs` | `Dict` | 原始环境观测（可选） |

#### 11.3.2 ActData

```python
class ActData:
    def __init__(self, act: Optional[int] = None, val: Optional[float] = None):
```

| 属性 / Attribute | 类型 / Type | 说明 / Description |
|---|---|---|
| `act` | `Optional[int]` | 动作索引（0=UP, 1=DOWN, 2=LEFT, 3=RIGHT） |
| `val` | `Optional[float]` | 关联值（Q值等，可选） |

#### 11.3.3 SampleData

```python
class SampleData:
    def __init__(self, obs: Optional[int] = None, _obs: Optional[int] = None,
                 act: Optional[int] = None, rew: float = 0.0,
                 done: bool = False, ret: float = 0.0):
```

| 属性 / Attribute | 类型 / Type | 说明 / Description |
|---|---|---|
| `obs` | `Optional[int]` | 当前状态键 / Current state key |
| `_obs` | `Optional[int]` | 下一状态键 / Next state key |
| `act` | `Optional[int]` | 执行的动作 / Action taken |
| `rew` | `float` | 获得的奖励 / Reward received |
| `done` | `bool` | 终止标志 / Termination flag |
| `ret` | `float` | 累计回报（MC用）/ Cumulative return |

---

### 11.4 特征处理 API / Feature Processing API

**源码 / Source**: `drone_delivery/agent_q_learning/feature/preprocessor.py`

#### 11.4.1 Preprocessor 类

```python
class Preprocessor:
    def __init__(self): ...
    def observation_process(self, env_obs: Dict[str, Any]) -> ObsData: ...
    def action_process(self, act_data: ActData) -> int: ...
```

**observation_process 步骤**:
1. 调用 `encode_state_key(env_obs)` 编码状态键
2. 调用 `compute_legal_actions(env_obs)` 计算合法动作
3. 返回 `ObsData(feature=state_key, legal_actions=legal_list)`

#### 11.4.2 状态编码函数

```python
def encode_state_key(env_obs: Dict[str, Any]) -> int
```

| 参数 / Parameter | 类型 / Type | 说明 / Description |
|---|---|---|
| `env_obs` | `Dict` | 环境观测字典，需含 `position, order_status, carrying` |

| 返回 / Returns | 类型 / Type | 说明 / Description |
|---|---|---|
| 状态键 | `int` | 唯一整数，范围 `[0, 102399]` |

**编码公式**: `state_key = (pos_idx << 6) | (order_status_bits << 1) | carrying`
- `pos_idx = x * 40 + y`（11位）
- `order_status_bits`: 5位二进制掩码
- `carrying`: 1位

#### 11.4.3 合法动作计算

```python
def compute_legal_actions(env_obs: Dict[str, Any]) -> List[int]
```

| 返回 / Returns | 类型 / Type | 说明 / Description |
|---|---|---|
| 合法动作 | `List[int]` | `[0,1,2,3]` 的子集 |

**不变量 / Invariants**:
- 检查4个方向的目标格是否为障碍物或越界
- 安全回退：所有方向被阻挡时返回 `[0,1,2,3]`

#### 11.4.4 reward_shaping 函数

```python
def reward_shaping(reward: float, env_obs=None, prev_env_obs=None, action=None) -> float
```

当前为直通版本。未来扩展方向：取货鼓励、距离塑形、原地不动惩罚、截断惩罚。

#### 11.4.5 sample_process 函数

```python
def sample_process(list_game_data: List[Any]) -> List[Dict[str, Any]]
```

| 参数 / Parameter | 类型 / Type | 说明 / Description |
|---|---|---|
| `list_game_data` | `List` | SampleData 对象或 dict 列表 |

| 返回 / Returns | 类型 / Type | 说明 / Description |
|---|---|---|
| 样本字典列表 | `List[Dict]` | 每项含 `state, action, reward, next_state, done` |

---

### 11.5 算法核心 API / Algorithm Core API

#### 11.5.1 Algorithm 类

**源码 / Source**: `drone_delivery/agent_q_learning/algorithm/algorithm.py`

```python
class Algorithm:
    def __init__(self, gamma: float = None, learning_rate: float = None,
                 action_size: int = None): ...
    def learn(self, list_sample_data: List[Any]) -> Dict[str, float]: ...
    def reset(self) -> None: ...
```

**learn() 不变量**:
- 遍历所有样本，对每个执行: `Q(s,a) += lr * [r + γ * max Q(s',a') * (1-done) - Q(s,a)]`
- `done=True` 时目标 = 即时奖励（无自举）
- 处理完毕后衰减 epsilon
- 返回: `{td_error, lr, epsilon, train_steps, q_table_size, num_samples}`

**reset() 不变量**:
- 重置 `epsilon = EPSILON_START`
- 重置 `train_steps = 0`
- 创建新的空白 `SparseQTable`

#### 11.5.2 SparseQTable

**源码 / Source**: `drone_delivery/agent_q_learning/model/model.py`

```python
class SparseQTable:
    def __init__(self, action_size: int = 4, default_q: float = 0.0): ...
```

| 属性 / Attribute | 类型 / Type | 说明 / Description |
|---|---|---|
| `action_size` | `int` | 动作空间大小（默认 4） |
| `default_q` | `float` | 未访问状态的默认Q值（默认 0.0） |
| `table` | `Dict[int, np.ndarray]` | 核心存储：状态键 → Q值向量 |
| `train_steps` | `int` | 累计更新次数 |

**索引操作 / Indexing Operations**:

```python
table[state_key]            # → np.ndarray, shape (action_size,)
table[state_key, action]    # → float, 标量Q值
table[state_key, :]         # → np.ndarray, 切片
table[state_key, action] = value  # → 设置Q值
```

**核心方法 / Core Methods**:

##### get_max_action

```python
def get_max_action(self, state_key: int, legal_actions: Optional[list] = None) -> int:
```

| 返回 / Returns | 说明 / Description |
|---|---|
| `int` | Q值最大的动作索引，多个最大值时随机打破平局 |

**不变量 / Invariants**:
- `legal_actions` 非空时，非法动作的Q值被设为 `-inf`
- 惰性初始化：未访问状态首次被创建为 `np.zeros(action_size)`

##### get_max_q

```python
def get_max_q(self, state_key: int, legal_actions: Optional[list] = None) -> float:
```

| 返回 / Returns | 说明 / Description |
|---|---|
| `float` | 最大Q值；所有动作均非法时返回 `default_q` |

##### update

```python
def update(self, state_key: int, action: int, value: float) -> float:
```

| 返回 / Returns | 说明 / Description |
|---|---|
| `float` | `abs(new_value - old_value)`（更新变化量） |

##### to_serializable / from_serializable

```python
def to_serializable(self) -> dict:
    # Returns: {"action_size", "default_q", "table": {state_key: [q_values]}, "train_steps"}

@classmethod
def from_serializable(cls, payload: dict, action_size=None, default_q=0.0) -> "SparseQTable":
```

**不变量 / Invariants**:
- `to_serializable()` 中所有 state key 转为 `int`，Q值转为 `list`
- `from_serializable()` 中 `arr.shape[0]` 必须等于 `action_size` 才恢复

##### save / load

```python
def save(self, filepath: str) -> None: ...
def load(self, filepath: str) -> None: ...
```

文件格式: `.npy`（numpy binary, `allow_pickle=True`）

---

### 11.6 工作流 API / Workflow API

#### 11.6.1 SampleManager（轨迹收集器）

**源码 / Source**: `drone_delivery/agent_q_learning/feature/definition.py`

```python
class SampleManager:
    def __init__(self): ...
    def add(self, obs=None, _obs=None, act=None, rew=0.0, done=False, ret=0.0) -> None: ...
    def process_last_frame(self) -> None: ...
    def get_game_data(self) -> List[SampleData]: ...
    def clear(self) -> None: ...
```

| 方法 / Method | 说明 / Description |
|---|---|
| `add(obs, _obs, act, rew, done, ret)` | 添加一条转移记录，自动从 ObsData 提取 feature |
| `process_last_frame()` | 反向传播折扣累计回报（MC用，γ=0.99） |
| `get_game_data()` | 返回本回合所有 `SampleData` 的列表副本 |
| `clear()` | 清空所有记录和 `_last_obs` |

**不变量 / Invariants**:
- `add` 接受 `ObsData` 或整数作为 `obs`/`_obs`
- `len(manager)` 返回当前样本数

#### 11.6.2 Config 配置类

**源码 / Source**: `drone_delivery/agent_q_learning/conf/conf.py`

```python
class Config:
    # Q-Learning 超参数
    GAMMA = 0.99                  # 折扣因子
    LEARNING_RATE = 0.1           # 学习率

    # Epsilon-greedy
    EPSILON_START = 1.0           # 初始探索率
    EPSILON_MIN = 0.05            # 最小探索率
    EPSILON_DECAY = 0.9995        # 衰减因子

    # 空间
    GRID_SIZE = 40                # 网格尺寸
    NUM_ORDERS = 5                # 订单数量
    STATE_SIZE = 102400           # 状态空间大小
    ACTION_SIZE = 4               # 动作空间大小

    # 环境
    MAX_STEPS = 500               # 每回合最大步数
    DEPOT = [30, 10]              # 配送中心坐标

    # 训练
    NUM_EPISODES = 10000          # 最大训练回合数
    EVAL_INTERVAL = 100           # 评估间隔
    EVAL_EPISODES = 10            # 每次评估回合数

    # 奖励
    DELIVERY_REWARD = 100.0       # 配送奖励
    SAFE_RETURN_BONUS = 500.0     # 安全返航奖励
    STEP_PENALTY = -1.0           # 每步惩罚
    COLLISION_PENALTY = -200.0    # 碰撞惩罚

    # 保存
    MODEL_SAVE_INTERVAL = 300     # 保存间隔(秒)
    MAX_SAVE_COUNT = 100          # 最大保存次数
    OBSERVATION_SHAPE = 55        # 观测特征维度
```

---

### 附录 A: 数据来源表 / Appendix A: Data Sources

| 数据项 / Data Item | 来源 / Source | 格式 / Format |
|---|---|---|
| 四区建筑轮廓 | OpenStreetMap (OSM) | GeoJSON → `data/cache/osm_reference_scenes/` |
| 武汉学院参考地图 | 实地采集 + OSM | `data/cache/reference_sources/` |
| 任务点坐标 | `scene_2d_manifest.json` | JSON, 12个任务点 |
| 3D场景资产 | 自动生成 | OBJ + MTL + 碰撞JSON |
| OSM原始几何 | 四区缓存 | `data/cache/osm_reference_scenes/` |

---

### 附录 B: 领域映射表 / Appendix B: Domain Mapping

| 峡谷漫步v2 / Gorge Walk v2 | 无人机配送 / Drone Delivery |
|---|---|
| `hero_pos` | `drone_pos` (x, y grid) |
| `treasure` | `delivery_order` |
| `treasure_state` (10-bit) | `order_status_mask` (5-bit) |
| `endpoint/start` | `depot` at [30, 10] |
| `obstacle` | `building / no_fly_zone` |
| `monster` | `dynamic_risk` |
| `max_step=1000` | `max_steps=500` |
| `treasure_score=100` | `delivery_reward=+100` |
| `end_score=150` | `safe_return_bonus=+500` |
| BFS 距离 | A* 距离（曼哈顿） |
| 4方向移动 | 保持不变（L1） |
| 8方向移动 | 保持不变（L2） |
| `legal_act` | 合法动作掩码 |
| `10-bit treasure_state` | `5-bit order_status` |

---

### 附录 C: 配置快速参考 / Appendix C: Config Quick Reference

| 参数 / Parameter | Q-learning | SARSA | Monte Carlo | DQN | A2C |
|---|---|---|---|---|---|
| `GAMMA` | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 |
| `LEARNING_RATE` | 0.1 | 0.1 | 0.01 | 0.0003 | 0.0003 |
| `EPSILON_START` | 1.0 | 1.0 | 1.0 | 1.0 | — |
| `EPSILON_MIN` | 0.05 | 0.05 | 0.05 | 0.05 | — |
| `EPSILON_DECAY` | 0.9995 | 0.9995 | 0.9995 | 0.9995 | — |
| `ACTION_SIZE` | 4 | 4 | 4 | 4 | 4 |
| `ACTION_TYPE` | 离散/Discrete | 离散/Discrete | 离散/Discrete | 离散/Discrete | 离散/Discrete |
| `HIDDEN_DIM` | — | — | — | 256 | 256 |
| `BATCH_SIZE` | — | — | — | 64 | — |

---

### 附录 D: 算法文档索引 / Appendix D: Algorithm Document Index

位于 `06_模型与算法/` 下的18篇算法文档：
Located in `06_模型与算法/`:

| 编号 | 文档 / Document | 内容 / Content |
|---|---|---|
| 01 | `01_算法体系总览.md` | 六算法框架概述 |
| 02 | `02_层级与算法映射.md` | L1-L4 → 算法映射关系 |
| 03 | `03_状态空间设计.md` | 状态编码、维度、稀疏性分析 |
| 04 | `04_动作空间设计.md` | 离散/连续动作空间设计 |
| 05 | `05_奖励函数设计.md` | 8分量奖励函数+归一化 |
| 06 | `06_Q_learning算法详解.md` | Q-learning数学推导+代码 |
| 07 | `07_SARSA算法详解.md` | SARSA数学推导+代码 |
| 08 | `08_蒙特卡洛算法详解.md` | MC算法+首次/每次访问 |
| 09 | `09_深度Q网络算法详解.md` | DQN/Dueling/Double/PER |
| 10 | `10_A2C算法详解.md` | A2C+GAE+并行环境 |
| 11 | `11_PPO算法详解.md` | PPO-Clip+CTDE+16维动作 |
| 12 | `12_经典RL算法对比.md` | Q/SARSA/MC对比分析 |
| 13 | `13_深度RL算法对比.md` | DQN/A2C/PPO对比分析 |
| 14 | `14_经典vs深度RL对比.md` | 两大范式对比 |
| 15 | `15_探索与利用策略.md` | ε-greedy/Boltzmann/UCB |
| 16 | `16_课程学习设计.md` | C1-C7课程配置+晋级条件 |
| 17 | `17_CTDE框架设计.md` | 集中训练分散执行 |
| 18 | `18_奖励归一化与塑形.md` | Welford/自适应权重/ICM |

---

### 附录 E: 版本历史 / Appendix E: Version History

| 版本 / Version | 日期 / Date | 变更 / Changes |
|---|---|---|
| v1.0 | 2026-06-15 | 初始版本，基础API + Q-learning/SARSA |
| v2.0 | 2026-07-01 | 新增DQN/A2C/PPO，四层级环境 |
| v3.0 | 2026-07-09 | 对齐Kaiwu结构，Agent九方法标准化 |
| v3.1 | 2026-07-11 | 模块化拆分，新增附录D（文档索引） |

---

> **下一篇 / Next**: [12. 常见问题与排错 / Troubleshooting](./12_troubleshooting.md)
