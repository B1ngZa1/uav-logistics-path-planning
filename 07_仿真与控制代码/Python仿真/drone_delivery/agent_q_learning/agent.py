#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 2026 Drone Delivery RL Project. All Rights Reserved.
###########################################################################
"""
无人机配送 — Q-learning智能体
Drone Delivery — Q-learning Agent

职责 / Responsibility:
    Q-learning (异策略/离线策略) 智能体，用于无人机外卖配送任务。
    Q-learning (off-policy) agent for drone food delivery tasks.
    集成 SparseQTable 模型、Algorithm 算法、Preprocessor 预处理器。
    Integrates SparseQTable model, Algorithm, and Preprocessor.

参考 / Reference:
    kaiwu_gorge_walk_v2/agent_q_learning/agent.py
    agent_sarsa/agent.py (Agent structure pattern)

领域适配 / Domain Adaptation:
    Gorge Walk treasure collection → Drone pickup-delivery:
    - hero_pos → drone_pos (x,y grid)
    - treasure → delivery_order
    - treasure_state (10-bit binary) → order_status_mask (5-bit)
    - endpoint → depot at [30,10]
    - 4-direction (UP/DOWN/LEFT/RIGHT) → same
    - max_step=1000 → max_steps=500
    - treasure_score=100 → delivery_reward=+100
    - end_score=150 → safe_return_bonus=+500

Q-learning vs SARSA (与SARSA的区别):
    Q-learning 使用 max_a' Q(s', a') (异策略/离线策略) 来更新Q值，
    而 SARSA 使用 Q(s', a') (同策略/在线策略/实际采取的动作)。
    Q-learning uses max_a' Q(s', a') (off-policy) to update Q-values,
    while SARSA uses Q(s', a') (on-policy / actual action taken).
"""

import numpy as np
import os
from typing import Dict, Any, Optional

from agent_q_learning.conf.conf import Config
from agent_q_learning.model.model import SparseQTable
from agent_q_learning.algorithm.algorithm import Algorithm
from agent_q_learning.feature.preprocessor import Preprocessor
from agent_q_learning.feature.definition import (
    ObsData, ActData, SampleData, reward_shaping,
)


class Agent:
    """
    Q-learning 智能体 — 无人机外卖配送。
    Q-learning Agent — Drone Food Delivery.

    使用异策略Q-learning算法进行表格型强化学习。
    Uses off-policy Q-learning algorithm for tabular reinforcement learning.

    核心组件 / Core components:
        - model (SparseQTable): 稀疏Q表 (字典存储，仅已访问状态占用内存)
          Sparse Q-table (dict-based, only visited states consume memory)
        - algorithm (Algorithm): Q-learning算法 (迭代更新 + epsilon衰减)
          Q-learning algorithm (iterative update + epsilon decay)
        - preprocessor (Preprocessor): 特征预处理器 (状态编码 + 合法动作计算)
          Feature preprocessor (state encoding + legal action computation)

    Usage / 用法::

        agent = Agent()
        env_obs, _ = env.reset()
        agent.reset(env_obs)
        obs_data = agent.observation_process(env_obs)

        while not done:
            act_data = agent.predict([obs_data])
            act = agent.action_process(act_data)
            _, reward, terminated, truncated, env_obs = env.step(act)
            shaped_reward = agent.reward_shaping(reward)
            # ... store transition ...
            obs_data = next_obs_data

        agent.learn(train_data)
    """

    def __init__(self,
                 agent_type: str = "player",
                 device: Any = None,
                 logger: Any = None,
                 monitor: Any = None) -> None:
        """
        初始化Q-learning智能体。
        Initialize Q-learning agent.

        Args:
            agent_type: 智能体类型 / Agent type ("player" for single-agent)
            device: 计算设备 (表格型RL固定用CPU) / Compute device (CPU for tabular RL)
            logger: 日志记录器 / Logger instance
            monitor: 监控实例 / Monitor instance
        """
        self.agent_type = agent_type
        self.logger = logger
        self.monitor = monitor

        # ============================================================
        # 超参数引用 (Hyper-parameter references)
        # ============================================================
        self.gamma = Config.GAMMA
        self.learning_rate = Config.LEARNING_RATE
        self.action_size = Config.ACTION_SIZE

        # ============================================================
        # 模型: 稀疏Q表 / Model: Sparse Q-table
        # ============================================================
        self.model = SparseQTable(action_size=self.action_size, default_q=0.0)

        # ============================================================
        # 算法: Q-learning更新逻辑 / Algorithm: Q-learning update logic
        # ============================================================
        self.algorithm = Algorithm(
            gamma=self.gamma,
            learning_rate=self.learning_rate,
            action_size=self.action_size,
        )
        # 共享同一个Q表给算法 / Share the same Q-table with algorithm
        self.algorithm.Q = self.model

        # ============================================================
        # 预处理器: 状态编码 + 合法动作计算
        # Preprocessor: state encoding + legal action computation
        # ============================================================
        self.preprocessor = Preprocessor()

        # ============================================================
        # 探索策略 / Exploration strategy
        # ============================================================
        self.epsilon = Config.EPSILON_START
        self.epsilon_min = Config.EPSILON_MIN
        self.epsilon_decay = Config.EPSILON_DECAY

        # ============================================================
        # 训练状态 / Training state
        # ============================================================
        self.train_steps = 0          # 累计训练步数 / Cumulative training steps
        self.episodes = 0             # 已完成回合数 / Completed episodes

    # ============================================================
    # 日志辅助 (Logging helper)
    # ============================================================

    def _log(self, level: str, message: str) -> None:
        """
        安全日志输出。
        Safe logging output.

        Args:
            level: 日志级别 (info, warning, error) / Log level
            message: 日志消息 / Log message
        """
        if self.logger is None:
            return
        log_method = getattr(self.logger, level, None) or getattr(self.logger, "info", None)
        if log_method is not None:
            log_method(message)

    # ============================================================
    # 回合管理 (Episode management)
    # ============================================================

    def reset(self, env_obs: Optional[Dict[str, Any]] = None) -> None:
        """
        回合开始时调用，更新回合计数并衰减探索率。
        Called at episode start; increments episode counter and decays epsilon.

        Args:
            env_obs: 环境初始观测 (暂未使用) / Initial environment observation (unused for now)
        """
        self.episodes += 1
        # 衰减探索率 (在每回合开始时) / Decay epsilon at episode start
        self._decay_epsilon()

    def _decay_epsilon(self) -> None:
        """
        指数衰减探索率。
        Exponentially decay exploration rate.

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ============================================================
    # 动作选择 (Action selection)
    # ============================================================

    def predict(self, list_obs_data):
        """
        epsilon-greedy 动作选择 (用于训练)。
        Epsilon-greedy action selection (for training).

        以概率 epsilon 随机探索，以概率 (1-epsilon) 贪心利用。
        从合法动作中均匀随机探索，利用时选择最大Q值的合法动作。
        With probability epsilon: random exploration among legal actions.
        With probability (1-epsilon): greedy exploitation (max Q among legal actions).

        Args:
            list_obs_data: 观测数据列表 (使用第一个元素) / List of ObsData (uses first element)

        Returns:
            ActData: 包含选中动作的ActData对象 / ActData object with selected action
        """
        # 提取观测数据 / Extract observation data
        obs_data = list_obs_data[0]

        state_key = obs_data.feature
        # 合法动作列表 / Legal action list
        legal_actions = (
            obs_data.legal_actions
            if obs_data.legal_actions is not None
            else list(range(self.action_size))
        )

        # ε-greedy 动作选择 / ε-greedy action selection
        if np.random.random() < self.epsilon:
            # 探索: 从合法动作中随机选择 / Explore: random from legal actions
            action = int(np.random.choice(legal_actions))
        else:
            # 利用: 选择Q值最大的合法动作 (随机打破平局)
            # Exploit: select legal action with max Q-value (random tie-breaking)
            q_values = self.model[state_key]
            # 只在合法动作中查找最大值 / Only look at legal actions
            valid_q = {a: q_values[a] for a in legal_actions}
            max_q = max(valid_q.values())
            best_actions = [a for a, q in valid_q.items() if q == max_q]
            action = int(np.random.choice(best_actions))

        # 每步递增训练步数 / Increment training steps each step
        self.train_steps += 1

        # 每步衰减epsilon / Decay epsilon each step
        self._decay_epsilon()

        return ActData(act=action, val=None)

    def exploit(self, env_obs):
        """
        评估时动作选择 (确定性贪心策略)。
        Evaluation action selection (deterministic greedy policy).

        始终选择Q值最大的合法动作，不使用探索。
        Always selects action with max Q-value among legal actions, no exploration.

        Args:
            env_obs: 环境观测字典 / Environment observation dict

        Returns:
            int: 可执行的动作索引 (0-3) / Executable action index
        """
        obs_data = self.observation_process(env_obs)
        state_key = obs_data.feature
        legal_actions = (
            obs_data.legal_actions
            if obs_data.legal_actions is not None
            else list(range(self.action_size))
        )

        # 贪心选择: 最大Q值的合法动作 / Greedy: best Q among legal actions
        q_values = self.model[state_key]
        valid_q = {a: q_values[a] for a in legal_actions}
        max_q = max(valid_q.values())
        best_actions = [a for a, q in valid_q.items() if q == max_q]
        action = int(np.random.choice(best_actions))

        return self.action_process(ActData(act=action))

    # ============================================================
    # 学习 (Learning)
    # ============================================================

    def learn(self, list_sample_data):
        """
        使用Q-learning算法更新Q表 (委托给Algorithm)。
        Update Q-table using Q-learning algorithm (delegates to Algorithm).

        处理空输入的保护: 如果样本列表为空，返回空指标字典。
        Empty input guard: returns empty metrics dict if sample list is empty.

        Args:
            list_sample_data: 样本数据列表 / List of sample data
                - SampleData 对象 / SampleData objects
                - or dict with keys: state, action, reward, next_state, done

        Returns:
            dict: 学习指标 / Learning metrics:
                - td_error: 平均TD误差 / Average TD error
                - lr: 学习率 / Learning rate
                - epsilon: 当前探索率 / Current epsilon
                - train_steps: 累计更新次数 / Cumulative updates
                - q_table_size: Q表状态数 / Q-table state count
                - num_samples: 处理的样本数 / Number of samples processed
        """
        if list_sample_data is None or len(list_sample_data) == 0:
            return {}

        return self.algorithm.learn(list_sample_data)

    # ============================================================
    # 观测与动作处理 (Observation and action processing)
    # ============================================================

    def observation_process(self, env_obs):
        """
        处理环境观测，生成 ObsData。
        Process environment observation, produce ObsData.

        委托给 Preprocessor.observation_process()。
        Delegates to Preprocessor.observation_process().

        步骤 / Steps:
            1. 编码状态键 / Encode state key (position + order_status + carrying)
            2. 计算合法动作 / Compute legal actions (avoid obstacles/walls)
            3. 包装为 ObsData / Wrap into ObsData

        Args:
            env_obs: 环境观测字典 / Environment observation dict

        Returns:
            ObsData: 包含 feature (状态键) + legal_actions (合法动作列表)
        """
        return self.preprocessor.observation_process(env_obs)

    def action_process(self, act_data):
        """
        处理动作数据，提取可执行的动作索引。
        Process action data, extract executable action index.

        委托给 Preprocessor.action_process()，钳位到 [0, ACTION_SIZE-1]。
        Delegates to Preprocessor.action_process(); clamps to [0, ACTION_SIZE-1].

        Args:
            act_data: ActData 对象 / ActData object

        Returns:
            int: 可执行的动作索引 / Executable action index (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT)
        """
        return self.preprocessor.action_process(act_data)

    # ============================================================
    # 奖励塑形 (Reward shaping)
    # ============================================================

    def reward_shaping(self, reward, env_obs=None, prev_env_obs=None, action=None):
        """
        奖励塑形 — 直通原始奖励 (保留扩展入口)。
        Reward shaping — pass-through raw reward (with extension hook).

        委托给 definition.reward_shaping()。
        Delegates to definition.reward_shaping().

        Args:
            reward: 原始环境奖励 / Raw environment reward (float)
            env_obs: 当前环境观测 (可选) / Current observation (optional)
            prev_env_obs: 前一环境观测 (可选) / Previous observation (optional)
            action: 执行的动作 (可选) / Action taken (optional)

        Returns:
            float: 塑形后的奖励 / Shaped reward
        """
        return reward_shaping(reward, env_obs, prev_env_obs, action)

    # ============================================================
    # 模型持久化 (Model persistence)
    # ============================================================

    def save_model(self, path=None, id="latest"):
        """
        保存Q表到检查点文件。
        Save Q-table to a checkpoint file.

        保存格式为 .npy (numpy binary with allow_pickle=True)。
        Save format: .npy (numpy binary with allow_pickle=True).

        Args:
            path: 保存目录路径 / Save directory path (default: "ckpt/")
            id: 检查点标识符 / Checkpoint identifier (default: "latest")
        """
        if path is None:
            path = "outputs/models"

        os.makedirs(path, exist_ok=True)
        model_file_path = os.path.join(path, f"q_learning_{str(id)}.npy")

        # 保存Q表及训练元数据 / Save Q-table with training metadata
        payload = self.model.to_serializable()
        payload["epsilon"] = float(self.epsilon)
        payload["train_steps"] = int(self.train_steps)
        payload["episodes"] = int(self.episodes)

        np.save(model_file_path, payload, allow_pickle=True)
        self._log(
            "info",
            f"Model saved to {model_file_path} "
            f"(states={len(self.model)}, epsilon={self.epsilon:.4f})",
        )

    def load_model(self, path=None, id="latest"):
        """
        从检查点文件加载Q表。
        Load Q-table from a checkpoint file.

        如果文件不存在或格式不兼容，静默回退到空白Q表。
        If file not found or incompatible, silently falls back to fresh Q-table.

        Args:
            path: 加载目录路径 / Load directory path (default: "outputs/models/")
            id: 检查点标识符 / Checkpoint identifier (default: "latest")
        """
        if path is None:
            path = "outputs/models"

        model_file_path = os.path.join(path, f"q_learning_{str(id)}.npy")

        # 文件不存在 → 使用空白Q表 / File not found → use fresh Q-table
        if not os.path.exists(model_file_path):
            self._log("warning", f"Model file not found: {model_file_path}; using fresh Q table")
            return

        try:
            payload = np.load(model_file_path, allow_pickle=True)
            if isinstance(payload, np.ndarray) and payload.shape == ():
                payload = payload.item()

            if isinstance(payload, dict):
                # 恢复Q表 / Restore Q-table
                restored = SparseQTable.from_serializable(
                    payload, action_size=self.action_size
                )
                self.model = restored
                # 保持算法引用同步 / Keep algorithm reference in sync
                self.algorithm.Q = self.model

                # 恢复训练状态 / Restore training state
                self.epsilon = max(
                    self.epsilon_min,
                    min(Config.EPSILON_START, float(payload.get("epsilon", self.epsilon))),
                )
                self.train_steps = int(payload.get("train_steps", 0))
                self.episodes = int(payload.get("episodes", 0))

                self._log(
                    "info",
                    f"Model loaded from {model_file_path} "
                    f"(states={len(self.model)}, epsilon={self.epsilon:.4f}, "
                    f"episodes={self.episodes})",
                )
            else:
                self._log(
                    "warning",
                    f"Ignoring incompatible checkpoint {model_file_path}; using fresh Q table",
                )
        except (TypeError, ValueError, KeyError) as error:
            self._log(
                "warning",
                f"Ignoring incompatible checkpoint {model_file_path}: {error}; "
                f"using fresh Q table",
            )

    # ============================================================
    # 属性 (Properties)
    # ============================================================

    @property
    def q_table_size(self) -> int:
        """已访问状态数量 / Number of visited states in Q-table."""
        return len(self.model)

    def __repr__(self) -> str:
        return (f"Agent(epsilon={self.epsilon:.4f}, "
                f"Q_states={len(self.model)}, "
                f"episodes={self.episodes}, "
                f"train_steps={self.train_steps})")
