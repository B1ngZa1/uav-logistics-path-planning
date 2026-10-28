#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 2026 Drone Delivery RL Project. All Rights Reserved.
###########################################################################
"""
无人机配送 — SARSA智能体
Drone Delivery — SARSA Agent

职责 / Responsibility:
    SARSA在线策略智能体，用于无人机外卖配送任务。
    SARSA on-policy agent for drone food delivery tasks.

SARSA核心特性 / SARSA Core Characteristics:
    - 在线策略/同策略(On-Policy): 学习与探索使用同一策略。
      更新Q(s,a)时使用当前策略实际选择的a'，而非max Q(s',a')。
    - On-Policy: uses same policy for learning and exploration.
      Updates Q(s,a) using actual a' chosen by current policy, not max Q(s',a').
    - 对探索敏感: SARSA的Q值反映了epsilon-greedy探索的影响，
      因此学到的策略在随机探索下更保守安全。
    - Exploration-sensitive: SARSA's Q-values reflect epsilon-greedy's impact,
      leading to more conservative and safer policies during random exploration.

更新公式 / Update Formula:
    Q(s,a) += lr * [r + gamma * Q(s',a') * (1-done) - Q(s,a)]

参考 / Reference:
    kaiwu_gorge_walk_v2/agent_sarsa/agent.py



领域适配 / Domain Adaptation:
    峡谷行走(Gorge Walk) → 无人机配送(Drone Delivery)
    hero_pos            → drone_pos          (无人机网格位置)
    treasure            → delivery_order      (配送订单)
    organ status        → order_status_mask   (订单完成状态)
    endpoint            → depot               (配送中心)
    4-direction         → same 4-direction    (4方向动作)
    max_step=1000       → max_steps=500       (最大步数)
"""

import numpy as np
from typing import List, Dict, Any, Optional

from agent_sarsa.conf.conf import Config
from agent_sarsa.algorithm.algorithm import Algorithm
from agent_sarsa.feature.definition import (
    ObsData, ActData, SampleData, reward_shaping, SampleManager,
)
from agent_sarsa.feature.preprocessor import (
    encode_state_key, legal_actions_from_position,
    drone_position, is_walkable, ACTION_DELTAS,
)


class Agent:
    """
    SARSA智能体 — 无人机配送任务。
    SARSA agent — Drone Delivery task.

    使用SARSA在线策略时序差分学习，维护稀疏Q表，
    通过epsilon-greedy策略平衡探索与利用。
    Uses SARSA on-policy TD learning, maintains sparse Q-table,
    balances exploration and exploitation via epsilon-greedy.
    """

    def __init__(self,
                 agent_type: str = "player",
                 device: Optional[str] = None,
                 logger: Optional[Any] = None,
                 monitor: Optional[Any] = None) -> None:
        """
        初始化SARSA无人机配送智能体。
        Initialize SARSA drone delivery agent.

        Args:
            agent_type: 智能体类型 / Agent type ("player")
            device: 计算设备 / Computing device (unused for tabular SARSA)
            logger: 日志记录器 / Logger instance
            monitor: 监控实例 / Monitor instance
        """
        self.logger = logger
        self.monitor = monitor
        self.agent_type = agent_type

        # ============================================================
        # 超参数初始化 (从Config加载)
        # Hyperparameter Initialization (loaded from Config)
        # ============================================================
        self.state_size = Config.STATE_SIZE
        self.action_size = Config.ACTION_SIZE
        self.learning_rate = Config.LR
        self.gamma = Config.GAMMA

        # Epsilon探索率调度 / Epsilon exploration schedule
        self.epsilon = Config.EPSILON_START
        self.epsilon_min = Config.EPSILON_MIN
        self.epsilon_decay = Config.EPSILON_DECAY

        # ============================================================
        # 算法与模型初始化
        # Algorithm & Model Initialization
        # ============================================================
        self.algorithm = Algorithm(
            gamma=self.gamma,
            learning_rate=self.learning_rate,
            state_size=self.state_size,
            action_size=self.action_size,
        )

        # ============================================================
        # 回合级状态 (Episode-level State)
        # ============================================================
        # 已知被阻挡的动作: {(x,y): {blocked_action, ...}}
        # Known blocked actions: {(x,y): {blocked_action, ...}}
        self.blocked_actions_by_pos: Dict[tuple, set] = {}

        # 前一步观测(用于奖励塑形) / Previous step observation (for reward shaping)
        self._prev_env_obs: Optional[Dict[str, Any]] = None

    # ============================================================
    # 日志辅助 (Logging Helper)
    # ============================================================

    def _log(self, level: str, message: str) -> None:
        """
        安全的日志记录辅助方法。
        Safe logging helper method.
        """
        if self.logger is None:
            return
        log_method = getattr(self.logger, level, None) or getattr(self.logger, "info", None)
        if log_method is not None:
            log_method(message)

    # ============================================================
    # 回合生命周期 (Episode Lifecycle)
    # ============================================================

    def reset(self, env_obs: Optional[Dict[str, Any]] = None) -> None:
        """
        回合开始时的重置操作。
        Reset operations at episode start.

        清除阻塞动作记录和上一步观测。
        Clears blocked action records and previous observation.

        Args:
            env_obs: 初始环境观测(可选) / Initial environment observation (optional)
        """
        self.blocked_actions_by_pos.clear()
        self._prev_env_obs = env_obs

    def reset_episode(self) -> None:
        """
        清除回合级状态(防止回合间状态泄漏)。
        Clear episode-level state (prevent leakage between episodes).
        """
        self.blocked_actions_by_pos.clear()
        self._prev_env_obs = None

    # ============================================================
    # 探索率衰减 (Epsilon Decay)
    # ============================================================

    def decay_epsilon(self) -> None:
        """
        对探索率进行指数衰减。
        Apply exponential decay to exploration rate.

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ============================================================
    # 动作选择 (Action Selection)
    # ============================================================

    def predict(self, list_obs_data: List[Any]) -> List[Any]:
        """
        使用epsilon-greedy策略预测动作（训练模式）。
        Predict action using epsilon-greedy policy (training mode).

        SARSA在线策略: 使用epsilon-greedy选择动作。
        返回的动作将在下一步被用作"实际执行的a'"来进行Q值更新。
        SARSA on-policy: selects action via epsilon-greedy.
        The returned action will be used as the "actual a'" for Q-value update
        in the next step.

        Args:
            list_obs_data: 观测数据列表 / List of observation data

        Returns:
            动作数据列表 / List of action data
        """
        obs_data = list_obs_data[0]
        # 提取状态特征 / Extract state feature
        state = obs_data.feature if hasattr(obs_data, "feature") else obs_data
        # 提取合法动作 / Extract legal actions
        legal_actions = (
            obs_data.legal_actions if hasattr(obs_data, "legal_actions") and obs_data.legal_actions
            else list(range(self.action_size))
        )

        # epsilon-greedy 动作选择 / epsilon-greedy action selection
        action = self._epsilon_greedy(
            state=state,
            epsilon=self.epsilon,
            legal_actions=legal_actions,
        )
        return [ActData(act=action)]

    def exploit(self, env_obs: Dict[str, Any]) -> int:
        """
        利用当前策略进行评估（纯贪心，用于验证）。
        Exploit current policy for evaluation (pure greedy, for validation).

        SARSA评估: 使用学习到的Q表进行贪心决策，不进行随机探索。
        SARSA evaluation: greedy decision using learned Q-table, no random exploration.

        Args:
            env_obs: 环境观测 / Environment observation

        Returns:
            可执行的动作 / Executable action
        """
        obs_data = self.observation_process(env_obs)
        state = obs_data.feature if hasattr(obs_data, "feature") else obs_data
        legal_actions = (
            obs_data.legal_actions if hasattr(obs_data, "legal_actions") and obs_data.legal_actions
            else list(range(self.action_size))
        )
        action = self._best_action(state, legal_actions)
        act_data = ActData(act=action)
        return self.action_process(act_data)

    def _epsilon_greedy(self, state: int, epsilon: float,
                        legal_actions: List[int]) -> int:
        """
        Epsilon-greedy动作选择。
        Epsilon-greedy action selection.

        以epsilon概率随机探索，以(1-epsilon)概率贪心利用。
        With epsilon probability: random exploration.
        With (1-epsilon) probability: greedy exploitation.

        Args:
            state: 当前状态键 / Current state key
            epsilon: 探索率 / Exploration rate
            legal_actions: 合法动作列表 / List of legal actions

        Returns:
            选择的动作索引 / Selected action index
        """
        legal_actions = legal_actions or list(range(self.action_size))

        # 探索: 从合法动作中随机选择 / Exploration: random choice from legal actions
        if np.random.rand() <= epsilon:
            return int(np.random.choice(legal_actions))

        # 利用: 选择Q值最大的合法动作 / Exploitation: choose best legal action
        return self._best_action(state, legal_actions)

    def _best_action(self, state: int, legal_actions: List[int]) -> int:
        """
        从合法动作中选择Q值最大的动作，随机打破平局。
        Select the action with max Q-value from legal actions, break ties randomly.

        随机打破平局避免总是选择第一个动作（系统性偏差）。
        Random tie-breaking avoids always selecting the first action (systematic bias).

        Args:
            state: 当前状态键 / Current state key
            legal_actions: 合法动作列表 / List of legal actions

        Returns:
            最佳动作索引 / Best action index
        """
        if not legal_actions:
            return 0

        # 获取Q值并遮蔽非法动作(设为负无穷) / Get Q-values & mask illegal actions (-inf)
        q_values = self.algorithm.Q[state, :].copy()
        for illegal_action in set(range(self.action_size)) - set(legal_actions):
            q_values[illegal_action] = -1e9

        # 随机打破平局 / Random tie-breaking among max values
        legal_q = q_values[legal_actions]
        if np.allclose(legal_q, legal_q[0]):
            return int(np.random.choice(legal_actions))

        return int(np.argmax(q_values))

    # ============================================================
    # 学习 (Learning) — SARSA在线策略更新
    # ============================================================

    def learn(self, list_sample_data: List[Any]) -> Dict[str, float]:
        """
        使用SARSA算法更新Q表（在线策略/同策略）。
        Update Q-table using SARSA algorithm (on-policy).

        SARSA更新公式 / SARSA Update Formula:
            Q(s,a) += lr * [r + gamma * Q(s',a') * (1-done) - Q(s,a)]

        关键区别(对比Q-learning) / Key Difference (vs Q-learning):
        - SARSA: 使用sample["next_action"]作为a'（在线策略/同策略）
          SARSA: uses sample["next_action"] as a' (on-policy)
        - Q-learning: 使用max_a' Q(s',a')作为a'（离线策略/异策略）
          Q-learning: uses max_a' Q(s',a') as a' (off-policy)

        Args:
            list_sample_data: 样本列表 / List of sample dicts or SampleData objects

        Returns:
            学习结果字典, 包含td_error和lr / Learning result dict with td_error and lr
        """
        # 将SampleData转换为算法可用的字典格式
        # Convert SampleData to dict format usable by algorithm
        if hasattr(list_sample_data[0], "obs"):
            # SampleData对象列表 → 字典列表 / List of SampleData → list of dicts
            converted = []
            for s in list_sample_data:
                converted.append({
                    "state": s.obs,
                    "action": s.act,
                    "reward": s.rew,
                    "next_state": s._obs,
                    "next_action": s.next_act if s.next_act is not None else -1,
                    "done": s.done,
                })
            list_sample_data = converted

        # 委托给算法核心进行Q表更新 / Delegate to algorithm core for Q-table update
        return self.algorithm.learn(list_sample_data)

    # ============================================================
    # 观测与动作处理 (Observation & Action Processing)
    # ============================================================

    def observation_process(self, env_obs: Dict[str, Any]) -> ObsData:
        """
        将无人机配送环境观测处理为状态特征ObsData。
        Process drone delivery environment observation into ObsData.

        状态编码(使用env.get_state_key()兼容格式):
        State encoding (using env.get_state_key() compatible format):
            state_key = (pos_idx << 6) | (order_status_bits << 1) | carrying

        组件 / Components:
            - pos_idx: x * grid_size + y  → 0~1599 (11 bits)
            - order_status_bits: 5-bit  → 0~31 (5 bits)
            - carrying: 0/1 → 0~1 (1 bit)

        优先使用环境自带的get_state_key()方法，回退到手动编码。
        Prefers environment's get_state_key() method, falls back to manual encoding.

        Args:
            env_obs: 环境观测字典(simple_delivery_env返回的格式)
                     Environment observation dict (format from simple_delivery_env)

        Returns:
            ObsData对象, 包含feature(状态键)和legal_actions
            ObsData object with feature (state key) and legal_actions
        """
        # 优先使用环境的get_state_key方法 / Prefer environment's get_state_key
        if hasattr(env_obs, "get_state_key"):
            feature = env_obs.get_state_key()
        elif isinstance(env_obs, dict) and "get_state_key" in dir(env_obs):
            # 某些包装格式 / Some wrapper formats
            feature = env_obs.get_state_key()
        else:
            # 回退: 手动编码 / Fallback: manual encoding
            feature = encode_state_key(env_obs)

        # 计算合法动作 / Compute legal actions
        x, y = drone_position(env_obs)
        pos_key = (x, y)
        blocked = self.blocked_actions_by_pos.get(pos_key, set())
        legal_actions = legal_actions_from_position(x, y, blocked_actions=blocked)

        return ObsData(
            feature=feature,
            legal_actions=legal_actions,
            raw_obs=env_obs,
        )

    def action_process(self, act_data) -> int:
        """
        将动作数据处理为可执行的动作整数。
        Process action data into executable action integer.

        支持列表格式(predict返回)和单个ActData(exploit使用)。
        Supports list format (predict return) and single ActData (exploit use).

        Args:
            act_data: ActData对象, 整数, 或列表 / ActData object, int, or list

        Returns:
            可执行的动作索引(0~3) / Executable action index (0~3)
        """
        # 处理列表格式(predict返回值) / Handle list format (predict return value)
        if isinstance(act_data, list):
            act_data = act_data[0]

        # 从ActData对象中提取动作 / Extract action from ActData object
        if hasattr(act_data, "act"):
            action = act_data.act
        else:
            action = act_data

        # 裁剪到有效范围 [0, action_size-1] / Clamp to valid range
        return int(max(0, min(self.action_size - 1, action)))

    # ============================================================
    # 奖励塑形 (Reward Shaping)
    # ============================================================

    def reward_shaping(self, env_reward: float,
                       env_obs: Optional[Dict[str, Any]] = None,
                       action: Optional[int] = None) -> float:
        """
        对原始环境奖励进行增强塑形。
        Enhance raw environment reward with shaping.

        委托给 feature/definition.py 中的 reward_shaping 函数。
        Delegates to reward_shaping function in feature/definition.py.

        Args:
            env_reward: 环境原始奖励 / Raw environment reward
            env_obs: 当前环境观测(可选) / Current env observation (optional)
            action: 执行的动作(可选) / Action taken (optional)

        Returns:
            塑形后的奖励 / Shaped reward
        """
        shaped = reward_shaping(
            env_reward=env_reward,
            env_obs=env_obs,
            prev_env_obs=self._prev_env_obs,
            action=action,
        )
        # 更新前一步观测 / Update previous observation
        self._prev_env_obs = env_obs
        return shaped

    # ============================================================
    # 转移记录 (Transition Recording)
    # ============================================================

    def record_transition(self, prev_obs: Dict[str, Any],
                          action: int,
                          next_obs: Dict[str, Any]) -> None:
        """
        记录状态转移，用于追踪被阻挡的动作。
        Record transition for blocked-action tracking.

        若无人机执行动作后位置未改变且回合未终止，标记该动作为"被阻挡"。
        If drone position didn't change after action and episode didn't terminate,
        marks the action as "blocked".

        Args:
            prev_obs: 前一环境观测 / Previous environment observation
            action: 执行的动作 / Action taken
            next_obs: 下一环境观测 / Next environment observation
        """
        prev_pos = self._extract_position(prev_obs)
        next_pos = self._extract_position(next_obs)

        if prev_pos is None or next_pos is None:
            return

        terminated = bool(next_obs.get("terminated", False)) if isinstance(next_obs, dict) else False
        prev_key = (int(prev_pos[0]), int(prev_pos[1]))

        if prev_pos[0] == next_pos[0] and prev_pos[1] == next_pos[1] and not terminated:
            # 位置未改变 = 该方向被阻挡 / Position unchanged = direction blocked
            if prev_key not in self.blocked_actions_by_pos:
                self.blocked_actions_by_pos[prev_key] = set()
            self.blocked_actions_by_pos[prev_key].add(int(action))
        else:
            # 移动成功 → 清除该方向的阻塞标记 / Movement succeeded → clear block
            if prev_key in self.blocked_actions_by_pos:
                self.blocked_actions_by_pos[prev_key].discard(int(action))

    def _extract_position(self, env_obs: Dict[str, Any]):
        """
        从环境观测中提取位置。
        Extract position from environment observation.

        Args:
            env_obs: 环境观测 / Environment observation

        Returns:
            位置数组[x, y] 或 None / Position array [x, y] or None
        """
        if env_obs is None:
            return None
        obs = env_obs.get("observation", env_obs) if isinstance(env_obs, dict) else {}
        position = obs.get("position", None)
        if position is not None:
            return np.asarray(position, dtype=np.int32)
        return None

    # ============================================================
    # 模型持久化 (Model Persistence)
    # ============================================================

    def save_model(self, path: Optional[str] = None, id: str = "1") -> None:
        """
        将Q表保存到检查点文件。
        Save Q-table to checkpoint file.

        保存内容包括Q表数据和当前epsilon值。
        Save includes Q-table data and current epsilon value.

        Args:
            path: 保存目录路径 / Directory path for saving (default: "ckpt")
            id: 检查点标识符 / Checkpoint identifier
        """
        import os
        path = path or "ckpt"
        os.makedirs(path, exist_ok=True)
        model_file_path = f"{path}/model.ckpt-{str(id)}.npy"

        # 保存Q表和epsilon / Save Q-table and epsilon
        payload = {
            "q_table": self.algorithm.Q.to_serializable(),
            "epsilon": float(self.epsilon),
        }
        np.save(model_file_path, payload, allow_pickle=True)
        self._log("info", f"save model {model_file_path} successfully")

    def load_model(self, path: Optional[str] = None, id: str = "1") -> None:
        """
        从检查点文件加载Q表。
        Load Q-table from checkpoint file.

        Args:
            path: 加载目录路径 / Directory path for loading (default: "ckpt")
            id: 检查点标识符 / Checkpoint identifier
        """
        import os
        from agent_sarsa.model.model import SparseQTable as SQT

        path = path or "ckpt"
        model_file_path = f"{path}/model.ckpt-{str(id)}.npy"
        try:
            payload = np.load(model_file_path, allow_pickle=True)
            # 解压0维数组 / Unpack 0-d array
            if isinstance(payload, np.ndarray) and payload.shape == ():
                payload = payload.item()

            if isinstance(payload, dict):
                # 新格式: 包含q_table键 / New format: contains q_table key
                q_table = SQT.from_serializable(
                    payload.get("q_table", {}),
                    action_size=self.action_size,
                )
                self.algorithm.Q = q_table
                self.epsilon = max(
                    self.epsilon_min,
                    min(Config.EPSILON_START,
                        float(payload.get("epsilon", self.epsilon))),
                )
                self._log("info", f"load model {model_file_path} successfully")
            else:
                # 旧格式: 原始numpy Q表 / Legacy format: raw numpy Q-table
                self.algorithm.Q = np.load(model_file_path)
                self._log("info", f"load model (legacy) {model_file_path} successfully")
        except FileNotFoundError:
            self._log("warning",
                      f"File {model_file_path} not found; using fresh Q table")
            self.algorithm.Q = SQT(action_size=self.action_size)
        except (TypeError, ValueError) as error:
            self._log(
                "warning",
                f"Ignoring incompatible checkpoint {model_file_path}: {error}; "
                f"using fresh Q table",
            )
            self.algorithm.Q = SQT(action_size=self.action_size)
