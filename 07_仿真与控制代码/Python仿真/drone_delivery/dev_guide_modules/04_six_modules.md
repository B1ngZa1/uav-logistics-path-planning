# 模块4：6子模块实现详解

> 项目：基于深度强化学习的多无人机协同物流配送路径规划
> 模块：4 - 6子模块实现详解
> 版本：v2.0
> 日期：2026-07-11

---

## 4.1 模块概述 / Module Overview

本项目采用Kaiwu无限之谷的6子模块结构，每个Agent包含以下模块：

```
agent_<algo>/
├── agent.py          # Agent标准接口
├── algorithm/        # 算法实现
│   ├── __init__.py
│   └── algorithm.py
├── conf/             # 配置文件
│   ├── __init__.py
│   ├── conf.py
│   └── monitor_builder.py
├── feature/          # 特征工程
│   ├── __init__.py
│   ├── definition.py
│   └── preprocessor.py
├── model/            # 神经网络模型
│   ├── __init__.py
│   └── model.py
└── workflow/         # 训练流程
    └── __init__.py
```

---

## 4.2 agent.py: Agent标准接口 / Agent Standard Interface

### 4.2.1 接口定义

```python
class Agent:
    """Agent标准接口 / Agent Standard Interface"""

    def reset(self, env_obs):
        """
        回合开始时重置 / Reset at episode start

        Args:
            env_obs: 环境观测 / Environment observation

        Returns:
            None
        """
        pass

    def predict(self, list_obs_data):
        """
        训练时动作选择（随机） / Action selection during training (stochastic)

        Args:
            list_obs_data: 观测数据列表 / List of observation data

        Returns:
            动作数据列表 / List of action data
        """
        pass

    def exploit(self, env_obs):
        """
        评估时动作选择（确定性） / Action selection during evaluation (deterministic)

        Args:
            env_obs: 环境观测 / Environment observation

        Returns:
            动作数据 / Action data
        """
        pass

    def learn(self, list_sample_data):
        """
        模型训练 / Model training

        Args:
            list_sample_data: 样本数据列表 / List of sample data

        Returns:
            训练信息 / Training info
        """
        pass

    def save_model(self, path, id):
        """
        保存模型 / Save model

        Args:
            path: 保存路径 / Save path
            id: 模型ID / Model ID

        Returns:
            None
        """
        pass

    def load_model(self, path, id):
        """
        加载模型 / Load model

        Args:
            path: 加载路径 / Load path
            id: 模型ID / Model ID

        Returns:
            None
        """
        pass

    def observation_process(self, env_obs):
        """
        环境观测 → ObsData / Environment observation → ObsData

        Args:
            env_obs: 环境观测 / Environment observation

        Returns:
            ObsData对象 / ObsData object
        """
        pass

    def action_process(self, act_data):
        """
        ActData → 动作 / ActData → Action

        Args:
            act_data: 动作数据 / Action data

        Returns:
            环境动作 / Environment action
        """
        pass

    def reward_shaping(self, reward):
        """
        奖励塑形 / Reward shaping

        Args:
            reward: 原始奖励 / Original reward

        Returns:
            塑形后奖励 / Shaped reward
        """
        pass
```

### 4.2.2 实现示例

```python
"""
职责 / Responsibility: Q-learning Agent实现
参考 / Reference: data/reference/kaiwu_gorge_walk_v2/agent_q_learning/
领域适配 / Domain Adaptation: 从峡谷→无人机的关键映射
"""

from drone_delivery.agent_q_learning.algorithm.algorithm import QLearningAlgorithm
from drone_delivery.agent_q_learning.model.model import QTableModel
from drone_delivery.agent_q_learning.conf.conf import Config
from drone_delivery.agent_q_learning.feature.definition import ObsData, ActData


class Agent:
    """Q-learning Agent / Q-learning Agent"""

    def __init__(self):
        """初始化 / Initialize"""
        self.algorithm = QLearningAlgorithm()
        self.model = QTableModel()
        self.config = Config()

    def reset(self, env_obs):
        """
        回合开始时重置 / Reset at episode start

        Args:
            env_obs: 环境观测 / Environment observation
        """
        self.algorithm.reset()

    def predict(self, list_obs_data):
        """
        训练时动作选择（随机） / Action selection during training (stochastic)

        Args:
            list_obs_data: 观测数据列表 / List of observation data

        Returns:
            动作数据列表 / List of action data
        """
        list_act_data = []
        for obs_data in list_obs_data:
            # ε-greedy探索 / ε-greedy exploration
            act_data = self.algorithm.select_action(obs_data, training=True)
            list_act_data.append(act_data)
        return list_act_data

    def exploit(self, env_obs):
        """
        评估时动作选择（确定性） / Action selection during evaluation (deterministic)

        Args:
            env_obs: 环境观测 / Environment observation

        Returns:
            动作数据 / Action data
        """
        obs_data = self.observation_process(env_obs)
        act_data = self.algorithm.select_action(obs_data, training=False)
        return act_data

    def learn(self, list_sample_data):
        """
        模型训练 / Model training

        Args:
            list_sample_data: 样本数据列表 / List of sample data

        Returns:
            训练信息 / Training info
        """
        for sample_data in list_sample_data:
            self.algorithm.update(sample_data)
        return {}

    def save_model(self, path, id):
        """
        保存模型 / Save model

        Args:
            path: 保存路径 / Save path
            id: 模型ID / Model ID
        """
        self.model.save(path, id)

    def load_model(self, path, id):
        """
        加载模型 / Load model

        Args:
            path: 加载路径 / Load path
            id: 模型ID / Model ID
        """
        self.model.load(path, id)

    def observation_process(self, env_obs):
        """
        环境观测 → ObsData / Environment observation → ObsData

        Args:
            env_obs: 环境观测 / Environment observation

        Returns:
            ObsData对象 / ObsData object
        """
        # 特征处理 / Feature processing
        feature = self.feature_preprocessor.process(env_obs)
        legal_act = self.get_legal_actions(env_obs)
        return ObsData(feature=feature, legal_act=legal_act)

    def action_process(self, act_data):
        """
        ActData → 动作 / ActData → Action

        Args:
            act_data: 动作数据 / Action data

        Returns:
            环境动作 / Environment action
        """
        return act_data.act

    def reward_shaping(self, reward):
        """
        奖励塑形 / Reward shaping

        Args:
            reward: 原始奖励 / Original reward

        Returns:
            塑形后奖励 / Shaped reward
        """
        # 基于势能的奖励塑形 / Potential-based reward shaping
        return reward
```

---

## 4.3 algorithm/: 算法实现 / Algorithm Implementation

### 4.3.1 算法类定义

```python
"""
职责 / Responsibility: Q-learning算法实现
参考 / Reference: data/reference/kaiwu_gorge_walk_v2/agent_q_learning/algorithm/
领域适配 / Domain Adaptation: 从峡谷→无人机的关键映射
"""

import numpy as np
from typing import Dict, List, Tuple


class QLearningAlgorithm:
    """Q-learning算法 / Q-learning Algorithm"""

    def __init__(self, learning_rate=0.1, gamma=0.99, epsilon=1.0):
        """
        初始化 / Initialize

        Args:
            learning_rate: 学习率 / Learning rate
            gamma: 折扣因子 / Discount factor
            epsilon: 探索率 / Exploration rate
        """
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table: Dict[Tuple, float] = {}

    def reset(self):
        """
        重置算法状态 / Reset algorithm state
        """
        pass  # Q-learning无需重置

    def get_q_value(self, state: int, action: int) -> float:
        """
        获取Q值 / Get Q-value

        Args:
            state: 状态 / State
            action: 动作 / Action

        Returns:
            Q值 / Q-value
        """
        return self.q_table.get((state, action), 0.0)

    def select_action(self, obs_data, training: bool = True) -> 'ActData':
        """
        选择动作 / Select action

        Args:
            obs_data: 观测数据 / Observation data
            training: 是否训练 / Whether training

        Returns:
            动作数据 / Action data
        """
        state = obs_data.feature
        legal_act = obs_data.legal_act

        if training and np.random.random() < self.epsilon:
            # ε-greedy探索 / ε-greedy exploration
            legal_actions = [a for a in range(len(legal_act)) if legal_act[a]]
            action = np.random.choice(legal_actions)
        else:
            # 贪心选择 / Greedy selection
            q_values = [self.get_q_value(state, a) for a in range(len(legal_act))]
            legal_q_values = [(q, a) for q, a in zip(q_values, legal_act) if a]
            action = max(legal_q_values, key=lambda x: x[0])[1]

        return ActData(act=action)

    def update(self, sample_data):
        """
        更新Q值 / Update Q-value

        Args:
            sample_data: 样本数据 / Sample data
            包含: obs, act, rew, next_obs, done
        """
        state = sample_data.obs.feature
        action = sample_data.act.act
        reward = sample_data.rew
        next_state = sample_data._obs.feature
        done = sample_data.done

        # Q值更新 / Q-value update
        current_q = self.get_q_value(state, action)

        if done:
            target = reward
        else:
            next_q_values = [self.get_q_value(next_state, a) for a in range(len(sample_data._obs.legal_act))]
            max_next_q = max(next_q_values)
            target = reward + self.gamma * max_next_q

        # 更新Q值 / Update Q-value
        new_q = current_q + self.lr * (target - current_q)
        self.q_table[(state, action)] = new_q
```

### 4.3.2 使用示例

```python
# 创建算法实例 / Create algorithm instance
algorithm = QLearningAlgorithm(learning_rate=0.1, gamma=0.99, epsilon=1.0)

# 选择动作 / Select action
obs_data = ObsData(feature=state, legal_act=legal_actions)
act_data = algorithm.select_action(obs_data, training=True)

# 更新Q值 / Update Q-value
sample_data = SampleData(obs=obs_data, act=act_data, rew=reward, _obs=next_obs_data, done=done)
algorithm.update(sample_data)
```

---

## 4.4 conf/: 配置文件 / Configuration

### 4.4.1 配置类定义

```python
"""
职责 / Responsibility: Q-learning配置
参考 / Reference: data/reference/kaiwu_gorge_walk_v2/agent_q_learning/conf/
领域适配 / Domain Adaptation: 从峡谷→无人机的关键映射
"""


class Config:
    """Q-learning配置 / Q-learning Configuration"""

    # 算法超参数 / Algorithm hyperparameters
    LEARNING_RATE = 0.1      # 学习率 / Learning rate
    GAMMA = 0.99             # 折扣因子 / Discount factor
    EPSILON_START = 1.0      # 初始探索率 / Initial exploration rate
    EPSILON_END = 0.01       # 最终探索率 / Final exploration rate
    EPSILON_DECAY = 0.995    # 探索率衰减 / Exploration rate decay

    # 环境参数 / Environment parameters
    GRID_SIZE = 40           # 网格大小 / Grid size
    NUM_ORDERS = 5           # 订单数量 / Number of orders
    MAX_STEPS = 500          # 最大步数 / Maximum steps

    # 训练参数 / Training parameters
    NUM_EPISODES = 10000     # 训练回合数 / Number of training episodes
    EVAL_INTERVAL = 100      # 评估间隔 / Evaluation interval
    SAVE_INTERVAL = 500      # 保存间隔 / Save interval
    LOG_INTERVAL = 10        # 日志间隔 / Log interval

    # 随机种子 / Random seed
    SEED = 42
```

### 4.4.2 配置文件（TOML格式）

```toml
# agent_q_learning/conf/train_env_conf.toml

[env_conf]
# 起点坐标 / Start coordinates
# 数组，包含两个整数元素[x, z]，每个坐标取值范围为0~63。
start = [29, 9]

# 终点坐标 / End coordinates
# 数组，包含两个整数元素[x, z]，每个坐标取值范围为0~63。
end = [11, 55]

# 最大步数 / Maximum steps
# 整型，取值范围1~2000。
max_step = 2000

# 宝箱ID数组 / Treasure ID array
treasure_id = []

# 是否生成随机宝箱 / Whether to generate random treasures
treasure_random = false

# 随机宝箱数量 / Number of random treasures
treasure_count = 0
```

---

## 4.5 feature/: 特征工程 / Feature Engineering

### 4.5.1 特征定义

```python
"""
职责 / Responsibility: 特征定义
参考 / Reference: data/reference/kaiwu_gorge_walk_v2/agent_q_learning/feature/
领域适配 / Domain Adaptation: 从峡谷→无人机的关键映射
"""


def create_cls(name, **kwargs):
    """
    创建数据类 / Create data class

    Args:
        name: 类名 / Class name
        **kwargs: 属性和默认值 / Attributes and default values

    Returns:
        数据类 / Data class
    """
    # 简化实现 / Simplified implementation
    class DataClass:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    DataClass.__name__ = name
    return DataClass


# 数据协议 / Data protocol
ObsData = create_cls("ObsData", feature=None, legal_act=None)
ActData = create_cls("ActData", act=None, val=None)
SampleData = create_cls("SampleData", obs=None, _obs=None, act=None, rew=None, done=None, ret=None)
```

### 4.5.2 特征预处理器

```python
"""
职责 / Responsibility: 特征预处理
参考 / Reference: data/reference/kaiwu_gorge_walk_v2/agent_q_learning/feature/
领域适配 / Domain Adaptation: 从峡谷→无人机的关键映射
"""

import numpy as np


class FeaturePreprocessor:
    """特征预处理器 / Feature Preprocessor"""

    def __init__(self, grid_size=40):
        """
        初始化 / Initialize

        Args:
            grid_size: 网格大小 / Grid size
        """
        self.grid_size = grid_size

    def process(self, env_obs):
        """
        处理环境观测 / Process environment observation

        Args:
            env_obs: 环境观测 / Environment observation
            {
                "position": (x, y),
                "order_status": (5,),
                "carrying": int,
            }

        Returns:
            特征向量 / Feature vector
        """
        position = env_obs["position"]
        order_status = env_obs["order_status"]
        carrying = env_obs["carrying"]

        # 归一化位置 / Normalize position
        x_norm = position[0] / self.grid_size
        y_norm = position[1] / self.grid_size

        # 拼接特征 / Concatenate features
        feature = np.array([x_norm, y_norm, *order_status, carrying])

        return feature

    def get_legal_actions(self, env_obs):
        """
        获取合法动作 / Get legal actions

        Args:
            env_obs: 环境观测 / Environment observation

        Returns:
            合法动作掩码 / Legal action mask
            [上, 下, 左, 右]
        """
        position = env_obs["position"]
        legal_act = [True, True, True, True]

        # 边界检查 / Boundary check
        if position[1] <= 0:
            legal_act[0] = False  # 不能向上
        if position[1] >= self.grid_size - 1:
            legal_act[1] = False  # 不能向下
        if position[0] <= 0:
            legal_act[2] = False  # 不能向左
        if position[0] >= self.grid_size - 1:
            legal_act[3] = False  # 不能向右

        return legal_act
```

---

## 4.6 model/: 神经网络模型 / Neural Network Models

### 4.6.1 Q值表模型

```python
"""
职责 / Responsibility: Q值表模型
参考 / Reference: data/reference/kaiwu_gorge_walk_v2/agent_q_learning/model/
领域适配 / Domain Adaptation: 从峡谷→无人机的关键映射
"""

import numpy as np


class QTableModel:
    """Q值表模型 / Q-table Model"""

    def __init__(self):
        """初始化 / Initialize"""
        self.q_table = {}

    def get_q_value(self, state, action):
        """
        获取Q值 / Get Q-value

        Args:
            state: 状态 / State
            action: 动作 / Action

        Returns:
            Q值 / Q-value
        """
        return self.q_table.get((state, action), 0.0)

    def set_q_value(self, state, action, value):
        """
        设置Q值 / Set Q-value

        Args:
            state: 状态 / State
            action: 动作 / Action
            value: Q值 / Q-value
        """
        self.q_table[(state, action)] = value

    def save(self, path, id):
        """
        保存模型 / Save model

        Args:
            path: 保存路径 / Save path
            id: 模型ID / Model ID
        """
        import pickle
        with open(f"{path}/q_table_{id}.pkl", "wb") as f:
            pickle.dump(self.q_table, f)

    def load(self, path, id):
        """
        加载模型 / Load model

        Args:
            path: 加载路径 / Load path
            id: 模型ID / Model ID
        """
        import pickle
        with open(f"{path}/q_table_{id}.pkl", "rb") as f:
            self.q_table = pickle.load(f)
```

### 4.6.2 神经网络模型（DQN/PPO）

```python
"""
职责 / Responsibility: 神经网络模型
参考 / Reference: data/reference/kaiwu_gorge_walk_v2/agent_dqn/model/
领域适配 / Domain Adaptation: 从峡谷→无人机的关键映射
"""

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """Q网络 / Q Network"""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        """
        初始化 / Initialize

        Args:
            state_dim: 状态维度 / State dimension
            action_dim: 动作维度 / Action dimension
            hidden_dim: 隐藏层维度 / Hidden dimension
        """
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        """
        前向传播 / Forward pass

        Args:
            state: 状态 / State

        Returns:
            Q值 / Q-values
        """
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class ActorCritic(nn.Module):
    """Actor-Critic网络 / Actor-Critic Network"""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        """
        初始化 / Initialize

        Args:
            state_dim: 状态维度 / State dimension
            action_dim: 动作维度 / Action dimension
            hidden_dim: 隐藏层维度 / Hidden dimension
        """
        super().__init__()

        # Actor网络 / Actor network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

        # Critic网络 / Critic network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        """
        前向传播 / Forward pass

        Args:
            state: 状态 / State

        Returns:
            动作概率和状态价值 / Action probabilities and state value
        """
        action_probs = self.actor(state)
        state_value = self.critic(state)
        return action_probs, state_value
```

---

## 4.7 workflow/: 训练流程 / Training Workflow

### 4.7.1 训练流程定义

```python
"""
职责 / Responsibility: 训练流程
参考 / Reference: data/reference/kaiwu_gorge_walk_v2/agent_q_learning/workflow/
领域适配 / Domain Adaptation: 从峡谷→无人机的关键映射
"""

import numpy as np
from typing import Dict, List


class TrainingWorkflow:
    """训练流程 / Training Workflow"""

    def __init__(self, agent, env, config):
        """
        初始化 / Initialize

        Args:
            agent: Agent实例 / Agent instance
            env: 环境实例 / Environment instance
            config: 配置 / Configuration
        """
        self.agent = agent
        self.env = env
        self.config = config

    def train(self, num_episodes):
        """
        训练模型 / Train model

        Args:
            num_episodes: 训练回合数 / Number of training episodes

        Returns:
            训练日志 / Training log
        """
        training_log = []

        for episode in range(num_episodes):
            # 重置环境 / Reset environment
            env_obs, info = self.env.reset(seed=self.config.SEED + episode)
            self.agent.reset(env_obs)

            episode_reward = 0
            episode_steps = 0
            done = False

            while not done:
                # 处理观测 / Process observation
                obs_data = self.agent.observation_process(env_obs)

                # 选择动作 / Select action
                list_obs_data = [obs_data]
                list_act_data = self.agent.predict(list_obs_data)
                act_data = list_act_data[0]

                # 执行动作 / Execute action
                action = self.agent.action_process(act_data)
                next_env_obs, reward, terminated, truncated, info = self.env.step(action)

                # 奖励塑形 / Reward shaping
                shaped_reward = self.agent.reward_shaping(reward)

                # 处理下一观测 / Process next observation
                next_obs_data = self.agent.observation_process(next_env_obs)

                # 创建样本数据 / Create sample data
                from drone_delivery.agent_q_learning.feature.definition import SampleData
                sample_data = SampleData(
                    obs=obs_data,
                    _obs=next_obs_data,
                    act=act_data,
                    rew=shaped_reward,
                    done=terminated,
                    ret=0
                )

                # 更新模型 / Update model
                self.agent.learn([sample_data])

                # 更新状态 / Update state
                env_obs = next_env_obs
                episode_reward += reward
                episode_steps += 1
                done = terminated or truncated

            # 记录日志 / Record log
            training_log.append({
                "episode": episode,
                "reward": episode_reward,
                "steps": episode_steps,
            })

            # 打印进度 / Print progress
            if episode % self.config.LOG_INTERVAL == 0:
                avg_reward = np.mean([log["reward"] for log in training_log[-self.config.LOG_INTERVAL:]])
                print(f"Episode {episode}: Avg Reward = {avg_reward:.2f}")

            # 保存模型 / Save model
            if episode % self.config.SAVE_INTERVAL == 0:
                self.agent.save_model("models", episode)

        return training_log
```

---

## 4.8 下一步 / Next Steps

了解6子模块后，可以：

1. **阅读模块5**：学习数据协议规范
2. **阅读模块6**：了解配置文件格式
3. **查看代码**：参考`drone_delivery/agent_q_learning/`中的完整实现
4. **运行测试**：执行`train_test.py`中的冒烟测试

---

**最后更新**：2026-07-11
**版本**：v2.0
**风格**：模仿Kaiwu无限之谷开发指南
