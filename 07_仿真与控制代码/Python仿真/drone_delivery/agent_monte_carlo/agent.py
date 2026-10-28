#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 2026 Drone Delivery RL Project. All Rights Reserved.
###########################################################################
"""
Monte Carlo Agent for Drone Food Delivery.

蒙特卡洛智能体，用于无人机外卖配送任务。
参考实现：Kaiwu Gorge Walk v2 agent_monte_carlo/agent.py

Monte Carlo (MC) is a model-free, episode-based RL method.
Unlike TD methods (Q-learning/SARSA) which update after every step, MC waits
until the end of an episode and then updates Q-values using the full cumulative
discounted return G_t for each state-action pair encountered.

蒙特卡洛是一种无模型的、基于回合的强化学习方法。
与每步更新的TD方法（Q-learning/SARSA）不同，MC等到回合结束后，
使用每个状态-动作对遇到的完整累积折扣回报G_t来更新Q值。

Key differences from Q-learning/SARSA (与Q-learning/SARSA的关键区别):
    1. Update timing (更新时机):
       MC:  Only at episode end (回合结束时)
       TD:  After every step (每步之后)

    2. Update target (更新目标):
       MC:  G_t = sum_{k=t}^{T} gamma^{k-t} * r_k  (完整回报 / full return)
       TD:  r + gamma * V(s')  (单步自举 / one-step bootstrap)

    3. Variance (方差):
       MC:  HIGH variance (完整回报是多步随机变量之和)
            → needs lower learning rate (lr=0.01 vs TD's 0.1)
       TD:  LOW variance (单步奖励)

    4. Bias (偏差):
       MC:  UNBIASED (使用实际回报，无自举)
       TD:  BIASED (自举引入初始Q值偏差)

Domain mapping from Gorge Walk v2 to Drone Delivery:
    hero_pos       → drone_pos         (无人机网格位置)
    treasure       → delivery_order    (配送订单)
    treasure_state → order_status_mask (订单完成状态)
    endpoint       → depot             (配送中心)
    start          → depot             (配送中心)
    4-direction    → same 4-direction  (相同4方向)
    treasure_score=100 → delivery_reward=+100 (配送奖励)
"""


import numpy as np
from kaiwudrl.interface.agent import BaseAgent
from agent_monte_carlo.conf.conf import Config
from agent_monte_carlo.algorithm.algorithm import Algorithm
from agent_monte_carlo.feature.definition import ObsData, ActData, SampleData, SampleManager


class Agent(BaseAgent):
    """
    Monte Carlo Agent for Drone Food Delivery.

    使用蒙特卡洛算法进行无人机外卖配送的智能体。

    MC is episode-based (回合制):
        - Collects entire episode trajectory (收集整个回合轨迹)
        - Computes G_t for each step at episode end (回合结束计算每步G_t)
        - Updates Q-values using first-visit MC (使用首次访问MC更新Q值)
        - Decays epsilon after each episode (每回合后衰减epsilon)
    """

    def __init__(self, agent_type="player", device=None, logger=None, monitor=None) -> None:
        """
        Initialize Monte Carlo agent for drone delivery.

        初始化无人机配送蒙特卡洛智能体。

        Args:
            agent_type: Type of agent / 智能体类型
            device: Computing device / 计算设备
            logger: Logger instance / 日志记录器实例
            monitor: Monitor instance / 监控实例
        """
        self.logger = logger
        self.monitor = monitor

        # Initialize hyperparameters from Config
        # 从Config初始化超参数
        self.state_size = Config.STATE_SIZE
        self.action_size = Config.ACTION_SIZE
        self.learning_rate = Config.START_LR
        self.gamma = Config.GAMMA

        # Epsilon schedule for exploration decay (MC uses slower decay)
        # 探索率衰减调度（MC使用较慢的衰减）
        self.epsilon = Config.EPSILON_START
        self.epsilon_min = Config.EPSILON_MIN
        self.epsilon_decay = Config.EPSILON_DECAY
        self.episodes = Config.EPISODES

        # Initialize Monte Carlo algorithm
        # 初始化蒙特卡洛算法
        self.algorithm = Algorithm(
            gamma=self.gamma,
            learning_rate=self.learning_rate,
            state_size=self.state_size,
            action_size=self.action_size,
        )

        # Tracking for blocked actions (positions where movement failed)
        # 追踪被阻挡的动作（移动失败的位置）
        self.blocked_actions_by_pos = {}

        # Episode counter
        # 回合计数器
        self.episode_cnt = 0

        super().__init__(agent_type, device, logger, monitor)

    def _log(self, level: str, message: str) -> None:
        """
        Safe logging helper.
        安全的日志辅助方法。

        Args:
            level: Log level (info, warning, error, debug)
            message: Log message
        """
        if self.logger is None:
            return
        log_method = getattr(self.logger, level, None) or getattr(self.logger, "info", None)
        if log_method is not None:
            log_method(message)

    def reset_episode(self) -> None:
        """
        Clear episode-level state that must not leak between episodes.

        清除回合级别的状态，避免回合间状态泄漏。

        Called at the start of each new training episode.
        在每个新训练回合开始时调用。
        """
        self.blocked_actions_by_pos.clear()

    def decay_epsilon(self) -> None:
        """
        Apply exponential decay to the exploration rate.

        对探索率进行指数衰减。

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        MC uses EPSILON_DECAY=0.999 (slower than Q-learning's 0.9995)
        because MC updates less frequently (once per episode vs once per step).
        MC使用EPSILON_DECAY=0.999（比Q-learning的0.9995更慢），
        因为MC更新频率较低（每回合一次 vs 每步一次）。
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def predict(self, list_obs_data):
        """
        Predict action using epsilon-greedy policy (MC on-policy).

        使用epsilon-greedy策略预测动作（MC同策略）。

        MC prediction is identical to Q-learning's predict — both use
        epsilon-greedy over the current Q-table. The difference is only
        in how learn() processes the collected data.
        MC的预测与Q-learning的predict完全相同——两者都使用当前Q表上的
        epsilon-greedy。区别仅在于learn()如何处理收集到的数据。

        Args:
            list_obs_data: List of observation data / 观测数据列表

        Returns:
            List of action data / 动作数据列表
        """
        obs_data = list_obs_data[0]
        legal_actions = (
            getattr(obs_data, "legal_actions", None)
            or list(range(self.action_size))
        )
        action = self._epsilon_greedy(
            state=obs_data.feature,
            epsilon=self.epsilon,
            legal_actions=legal_actions,
        )
        return [ActData(act=action)]

    def exploit(self, env_obs):
        """
        Exploit current policy for evaluation with small exploration.

        利用当前策略进行评估（包含小幅探索以处理未知状态）。

        MC evaluation: uses the learned Q-table greedily with minimal
        exploration to handle unseen states.
        MC评估：使用已学习的Q表贪心选择，保持最低探索处理未见状态。

        Args:
            env_obs: Environment observation / 环境观测

        Returns:
            Executable action / 可执行动作
        """
        from agent_monte_carlo.feature.preprocessor import observation_process, action_process

        obs_data = observation_process(env_obs)
        legal_actions = (
            getattr(obs_data, "legal_actions", None)
            or list(range(self.action_size))
        )
        action = self._best_action(obs_data.feature, legal_actions)
        act_data = ActData(act=action)
        return action_process(act_data)

    def _epsilon_greedy(self, state, epsilon=0.1, legal_actions=None):
        """
        Epsilon-greedy algorithm for action selection.

        ε-贪心算法用于动作选择。

        Args:
            state: Current state key (integer) / 当前状态键（整数）
            epsilon: Exploration rate / 探索率
            legal_actions: List of valid actions / 有效动作列表

        Returns:
            Selected action index / 选择的动作索引
        """
        legal_actions = legal_actions or list(range(self.action_size))

        # Exploration: choose random action from legal actions
        # 探索：从合法动作中随机选择
        if np.random.rand() <= epsilon:
            action = int(np.random.choice(legal_actions))
            return action

        # Exploitation: choose best action from legal actions
        # 利用：从合法动作中选择最佳动作
        return self._best_action(state, legal_actions)

    def _best_action(self, state, legal_actions):
        """
        Select the best action from legal actions, breaking ties randomly.

        从合法动作中选择最佳动作，随机打破平局。

        确保不会总是选择第一个动作（避免系统性偏差）。
        Ensures the first action is not always selected (avoids systematic bias).

        Args:
            state: Current state key / 当前状态键
            legal_actions: List of valid actions / 有效动作列表

        Returns:
            Best action index / 最佳动作索引
        """
        if not legal_actions:
            return 0

        # Get Q-values from the algorithm
        # 从算法获取Q值
        q_values = self.algorithm.get_q_values(state)

        # Find max Q among legal actions
        # 在合法动作中找到最大Q值
        legal_q = [(a, q_values[a]) for a in legal_actions]
        max_q = max(v for _, v in legal_q)
        best_actions = [a for a, v in legal_q if v == max_q]

        # Random tie-breaking
        # 随机打破平局
        return int(np.random.choice(best_actions))

    def learn(self, list_sample_data):
        """
        Learn from a complete episode using Monte Carlo method.

        使用蒙特卡洛方法从完整回合中学习。

        This is the KEY difference from TD methods:
        - TD: learn() is called once per step with 1 sample
        - MC: learn() is called once per EPISODE with ALL samples

        这是与TD方法的关键区别：
        - TD: learn()每步调用一次，传入1个样本
        - MC: learn()每回合调用一次，传入所有样本

        MC work flow (MC工作流程):
            1. All episode samples are passed in as a list
               传入整个回合的样本列表
            2. Algorithm computes G_t for each step (backward pass)
               算法为每步计算G_t（反向计算）
            3. First-visit MC updates Q(s,a) for each unique (s,a) pair
               首次访问MC为每对唯一的(s,a)更新Q值
            4. Q(s,a) += lr * (G_t - Q(s,a))
               使用学习率更新Q值

        Args:
            list_sample_data: List of all samples from the episode.
                回合中所有样本的列表。Each sample is a dict or SampleData with keys:
                    state, action, reward, next_state, done
                每个样本是dict或SampleData，包含键：state, action, reward, next_state, done

        Returns:
            Learning result dict with total_error, num_updates, lr, mean_return
            包含total_error, num_updates, lr, mean_return的学习结果字典
        """
        # Convert SampleData objects to dicts if needed
        # 如果需要，将SampleData对象转换为字典
        from agent_monte_carlo.feature.definition import sample_process
        processed_samples = sample_process(list_sample_data)

        # Delegate to algorithm for first-visit MC update
        # 委托给算法进行首次访问MC更新
        result = self.algorithm.learn(processed_samples)

        # Log learning statistics
        # 记录学习统计信息
        if self.logger and self.episode_cnt % 100 == 0:
            self._log(
                "info",
                f"[MC] episode={self.episode_cnt}, "
                f"updates={result['num_updates']}, "
                f"total_error={result['total_error']:.2f}, "
                f"mean_return={result['mean_return']:.2f}, "
                f"epsilon={self.epsilon:.4f}, "
                f"visited_states={self.algorithm.num_visited_states()}",
            )

        return result

    def observation_process(self, env_obs):
        """
        Process drone delivery environment observation into state feature.

        将无人机配送环境观测处理为状态特征。

        Uses the feature/preprocessor module for state encoding.
        使用feature/preprocessor模块进行状态编码。

        State encoding (状态编码):
            state_key = (pos_idx << 6) | (order_status_bits << 1) | carrying

        Args:
            env_obs: Environment observation dict / 环境观测字典

        Returns:
            ObsData with feature (state key) and legal_actions
            包含feature（状态键）和legal_actions的ObsData
        """
        from agent_monte_carlo.feature.preprocessor import observation_process as obs_proc
        return obs_proc(env_obs)

    def action_process(self, act_data):
        """
        Process action data into an executable action integer.

        将动作数据处理为可执行的整型动作。

        Args:
            act_data: Action data object / 动作数据对象

        Returns:
            Executable action index / 可执行的动作索引
        """
        from agent_monte_carlo.feature.preprocessor import action_process as act_proc
        return act_proc(act_data)

    def record_transition(self, prev_obs, action, next_obs):
        """
        Record transition for blocked-action tracking.

        记录状态转移用于追踪被阻挡的动作。

        If the drone position did not change after taking an action and the
        episode did not terminate, the action is considered blocked.
        MC uses this to avoid wasting episode steps on impossible actions.
        如果无人机执行动作后位置未变化且回合未终止，则认为该动作被阻挡。
        MC使用此功能避免在不可能的动作上浪费回合步数。

        Args:
            prev_obs: Previous environment observation / 上一个环境观测
            action: Action that was taken / 执行的动作
            next_obs: Next environment observation / 下一个环境观测
        """
        prev_pos = self._extract_position(prev_obs)
        next_pos = self._extract_position(next_obs)

        if prev_pos is None or next_pos is None:
            return

        terminated = bool(next_obs.get("terminated", False)) if isinstance(next_obs, dict) else False
        prev_key = (int(prev_pos[0]), int(prev_pos[1]))

        if prev_pos[0] == next_pos[0] and prev_pos[1] == next_pos[1] and not terminated:
            self.blocked_actions_by_pos.setdefault(prev_key, set()).add(int(action))
        else:
            self.blocked_actions_by_pos.get(prev_key, set()).discard(int(action))

    def _extract_position(self, env_obs):
        """
        Extract position from environment observation.

        从环境观测中提取位置。

        Args:
            env_obs: Environment observation / 环境观测

        Returns:
            Position array [x, y] or None / 位置数组 [x, y] 或 None
        """
        obs = env_obs.get("observation", env_obs) if isinstance(env_obs, dict) else {}
        position = obs.get("position", None)
        if position is not None:
            return np.asarray(position, dtype=np.int32)
        return None

    def save_model(self, path=None, id="1") -> None:
        """
        Save the Q-table to a checkpoint file.

        将Q表保存到检查点文件。

        MC saves the full Q-table (sparse dictionary) along with
        training metadata (epsilon, episode count, visit counts).
        MC保存完整的Q表（稀疏字典）及训练元数据（epsilon、回合计数、访问计数）。

        Args:
            path: Directory path for saving / 保存目录路径
            id: Checkpoint identifier / 检查点标识符
        """
        import os

        path = path or "ckpt"
        os.makedirs(path, exist_ok=True)
        model_file_path = f"{path}/model.ckpt-{str(id)}.npy"

        # Save Q-table with metadata for version compatibility
        # 保存Q表及元数据用于版本兼容性
        payload = {
            "algorithm": self.algorithm.to_serializable(),
            "epsilon": float(self.epsilon),
            "episode_cnt": self.episode_cnt,
        }
        np.save(model_file_path, payload, allow_pickle=True)
        self._log("info", f"[MC] save model {model_file_path} successfully")

    def load_model(self, path=None, id="1") -> None:
        """
        Load the Q-table from a checkpoint file.

        从检查点文件加载Q表。

        Args:
            path: Directory path for loading / 加载目录路径
            id: Checkpoint identifier / 检查点标识符
        """
        import os

        path = path or "ckpt"
        model_file_path = f"{path}/model.ckpt-{str(id)}.npy"

        try:
            payload = np.load(model_file_path, allow_pickle=True)
            if isinstance(payload, np.ndarray) and payload.shape == ():
                payload = payload.item()

            if isinstance(payload, dict) and "algorithm" in payload:
                # New format with full algorithm serialization
                # 新格式：完整算法序列化
                self.algorithm = Algorithm.from_serializable(
                    payload["algorithm"],
                    gamma=self.gamma,
                    learning_rate=self.learning_rate,
                )
                self.epsilon = max(
                    self.epsilon_min,
                    min(Config.EPSILON_START, float(payload.get("epsilon", self.epsilon))),
                )
                self.episode_cnt = int(payload.get("episode_cnt", 0))
                self._log(
                    "info",
                    f"[MC] load model {model_file_path} successfully, "
                    f"states={self.algorithm.num_visited_states()}",
                )
            elif isinstance(payload, dict):
                # Legacy format with q_table
                # 旧格式：q_table
                from agent_monte_carlo.algorithm.algorithm import Algorithm as Algo
                algo = Algo(
                    gamma=self.gamma,
                    learning_rate=self.learning_rate,
                    state_size=self.state_size,
                    action_size=self.action_size,
                )
                restored = Algo.from_serializable(
                    {"q_table": payload.get("q_table", {}),
                     "action_size": self.action_size},
                    gamma=self.gamma,
                    learning_rate=self.learning_rate,
                )
                self.algorithm = restored
                self.epsilon = max(
                    self.epsilon_min,
                    min(Config.EPSILON_START, float(payload.get("epsilon", self.epsilon))),
                )
                self._log("info", f"[MC] load model (legacy) {model_file_path} successfully")
            else:
                self._log("warning", f"[MC] unrecognized checkpoint format: {model_file_path}")
        except FileNotFoundError:
            self._log("warning", f"[MC] File {model_file_path} not found; using fresh Q table")
            self.algorithm = Algorithm(
                gamma=self.gamma,
                learning_rate=self.learning_rate,
                state_size=self.state_size,
                action_size=self.action_size,
            )
        except (TypeError, ValueError, KeyError) as error:
            self._log(
                "warning",
                f"[MC] Ignoring incompatible checkpoint {model_file_path}: {error}; "
                f"using fresh Q table",
            )
            self.algorithm = Algorithm(
                gamma=self.gamma,
                learning_rate=self.learning_rate,
                state_size=self.state_size,
                action_size=self.action_size,
            )
