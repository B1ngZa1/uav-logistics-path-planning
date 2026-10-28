# -*- coding: utf-8 -*-
"""
================================================================================
agent.py — DQN 智能体 / DQN Agent
================================================================================
职责 / Responsibility:
    将模型(DuelingDQN)、算法(Algorithm)、预处理器(Preprocessor)组合
    为完整的强化学习 DQN Agent, 遵循 KaiWu 标准接口:
    - predict: epsilon-贪心动作选择 (训练用 / for training)
    - exploit: 纯贪心动作选择 (评估用 / for evaluation)
    - learn: 委托 Algorithm 处理样本并执行学习
    - save / load: 模型持久化 / model persistence
    Combines model, algorithm, preprocessor into a complete DQN Agent
    following KaiWu standard interface.

参考 / Reference:
    - KaiWu DRL 框架的 Agent 设计模式
    - agent_q_learning/agent.py (KaiWu标准接口参考)
    - Gorge Walk DIY agent 的 Agent 封装方式

领域适配 / Domain Adaptation:
    - 使用 50 维向量观测 + 8 方向动作的 Dueling DQN
    - 配送任务: legal_act 掩码限制非法方向选择
    - Dueling DQN with 50-dim vector obs + 8-direction actions
    - Delivery task: legal_act mask constrains illegal direction selection

DQN 训练特点 / DQN Training Characteristics:
    - 异策略 (off-policy): 使用 replay buffer 存储历史经验
    - 每步学习: 每次环境交互后采样 minibatch 进行梯度更新
    - Double DQN: 使用在线网络选动作, 目标网络评估值, 减少过估计
    - Epsilon 衰减: 每步指数衰减, 平衡探索与利用
================================================================================
"""

import os
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import torch

from conf.conf import config
from feature.definition import (
    ObsData,
    ActData,
    SampleData,
    reward_shaping as _reward_shaping,
    SampleManager,
)
from feature.preprocessor import Preprocessor
from model.model import DuelingDQN
from algorithm.algorithm import Algorithm


# =============================================================================
# Agent — DQN 智能体封装 / Agent — DQN Agent Wrapper
# =============================================================================

class Agent:
    """
    DQN 智能体 / DQN Agent

    封装了感知 (preprocessor) → 决策 (model) → 学习 (algorithm) 的完整闭环。
    Encapsulates the full perception → decision → learning loop.

    Usage / 用法::

        agent = Agent()
        env_obs, _ = env.reset()
        agent.reset(env_obs)
        obs_data = agent.observation_process(env_obs)

        while not done:
            act_data = agent.predict([obs_data])        # epsilon-greedy
            act = agent.action_process(act_data)
            env_obs, reward, terminated, truncated, _ = env.step(act)
            shaped_reward = agent.reward_shaping(reward)
            next_obs_data = agent.observation_process(env_obs)
            done = terminated or truncated

            # 构造样本并学习
            # Construct sample and learn
            sample = SampleData(
                obs=obs_data.feature, act=act, rew=shaped_reward,
                next_obs=next_obs_data.feature, done=done,
                legal_act=obs_data.legal_act,
            )
            agent.learn([sample])
            obs_data = next_obs_data
    """

    def __init__(
        self,
        device: str = config.DEVICE,
        grid_size: int = 10,
        logger: Any = None,
        monitor: Any = None,
    ) -> None:
        """
        Initialize the DQN agent.

        Args / 参数:
            device (str): 计算设备 / Computation device ("cpu" or "cuda")
            grid_size (int): 网格尺寸 / Grid size (default: 10x10)
            logger: 日志记录器 / Logger instance
            monitor: 监控实例 / Monitor instance
        """
        self.logger = logger
        self.monitor = monitor
        self._device: torch.device = torch.device(device)
        self._grid_size: int = grid_size

        # -----------------------------------------------------------------
        # 子组件 / Sub-components
        # -----------------------------------------------------------------
        self._preprocessor: Preprocessor = Preprocessor(grid_size=grid_size)
        """特征预处理器 / Feature preprocessor — 原始观测 → ObsData"""

        self._sample_manager: SampleManager = SampleManager()
        """样本管理器 / Sample manager — 构造 SampleData"""

        self._algorithm: Algorithm = Algorithm(device=device)
        """DQN 训练算法 / DQN training algorithm — Double Dueling DQN"""

    # =========================================================================
    # 日志辅助 / Logging Helper
    # =========================================================================

    def _log(self, level: str, message: str) -> None:
        """
        安全日志输出 / Safe logging output.

        Args / 参数:
            level (str): 日志级别 (info, warning, error) / Log level
            message (str): 日志消息 / Log message
        """
        if self.logger is None:
            return
        log_method = getattr(self.logger, level, None) or getattr(self.logger, "info", None)
        if log_method is not None:
            log_method(message)

    # =========================================================================
    # reset — 回合重置 / Episode Reset
    # =========================================================================

    def reset(self, env_obs: Optional[Dict[str, Any]] = None) -> None:
        """
        回合开始时调用 — 重置回合状态。
        Called at episode start — resets episode state.

        在 DQN 中, reset 主要用于递增全局步计数和清除 episod 缓存。
        不需要在这里衰减 epsilon (epsilon 在 algorithm.learn() 中每步衰减)。
        In DQN, reset mainly increments step counter and clears episode cache.
        Epsilon is decayed per-step inside algorithm.learn().

        Args / 参数:
            env_obs: 环境初始观测 (可选) / Initial environment observation (optional)
        """
        # 清除上一步缓存 / Clear previous step cache
        self._last_obs_data = None
        self._last_act = None

    # =========================================================================
    # predict — Epsilon-Greedy 动作选择 (训练用)
    # =========================================================================

    def predict(self, list_obs_data: List[ObsData]) -> ActData:
        """
        训练模式动作选择: epsilon-贪心策略, 8 个方向动作。
        Training mode action selection: epsilon-greedy with 8-direction actions.

        以概率 epsilon 随机选择合法动作 (探索),
        否则选择 Q 值最高的合法动作 (利用)。
        每次调用后递增全局步计数 (algorithm.step_tick)。
        With probability epsilon, randomly choose a legal action (explore),
        otherwise choose the legal action with highest Q-value (exploit).
        Increments global step counter on each call (algorithm.step_tick).

        Args / 参数:
            list_obs_data (List[ObsData]): 观测数据列表 (使用第一个元素)
                                            List of ObsData (uses first element)

        Returns / 返回:
            ActData: 包含选中动作的 ActData 对象
                     ActData object with selected action and its Q-value
        """
        obs_data = list_obs_data[0]

        # 缓存当前观测 (供 learn() 构造样本使用)
        # Cache current observation for sample construction in learn()
        self._last_obs_data = obs_data

        # ---- Epsilon-贪心 / Epsilon-greedy ----
        if np.random.random() < self._algorithm.epsilon:
            # 探索: 随机选择合法动作 / Explore: random legal action
            legal_indices = np.where(obs_data.legal_act > 0.5)[0]
            if len(legal_indices) == 0:
                # 极端情况: 无合法动作, 选动作0
                # Edge case: no legal action available, pick action 0
                act = 0
            else:
                act = int(np.random.choice(legal_indices))
            val = 0.0
        else:
            # 利用: 选Q值最高的合法动作 / Exploit: highest Q among legal actions
            act, val = self._greedy_action(obs_data)

        # 缓存选中的动作 / Cache selected action
        self._last_act = act

        # 递增全局步计数 / Increment global step counter
        self._algorithm.step_tick()

        return self._preprocessor.action_process(act=act, val=val)

    # =========================================================================
    # exploit — 纯贪心动作选择 (评估用)
    # =========================================================================

    def exploit(self, env_obs: Union[dict, np.ndarray]) -> ActData:
        """
        评估模式动作选择: 纯贪心 argmax, 无探索。
        Evaluation mode action selection: pure greedy argmax, no exploration.

        始终选择 Q 值最高的合法动作, 不使用 epsilon 随机。
        Always selects the legal action with highest Q-value, no epsilon randomness.

        Args / 参数:
            env_obs: 环境观测字典 / Environment observation dict
                     {"position": (2,), "vector": (50,)}

        Returns / 返回:
            ActData: 包含最优动作的 ActData 对象
                     ActData object with the best action and its Q-value
        """
        # 预处理观测 (不缓存, 评估时不需要构造样本)
        # Preprocess observation (no caching; no sample construction needed during eval)
        obs_data: ObsData = self._preprocessor.observation_process(env_obs)

        # 纯贪心选择 / Pure greedy selection
        act, val = self._greedy_action(obs_data)

        return self._preprocessor.action_process(act=act, val=val)

    def _greedy_action(self, obs_data: ObsData) -> Tuple[int, float]:
        """
        贪心动作选择: 在所有合法动作中选 Q 值最高的。
        Greedy action selection: choose the legal action with highest Q-value.

        对非法动作施加极大负值, 确保不会被选中。
        Apply large negative penalty to illegal actions to exclude them.

        Args / 参数:
            obs_data (ObsData): 预处理后的观测 / Preprocessed observation

        Returns / 返回:
            Tuple[int, float]: (action_index, q_value)
        """
        # 使用在线网络计算 Q 值
        # Compute Q-values using online network
        feature_tensor = torch.tensor(
            obs_data.feature, dtype=torch.float32
        ).unsqueeze(0).to(self._device)

        with torch.no_grad():
            q_tensor: torch.Tensor = self._algorithm.q_online(feature_tensor)

        q_numpy: np.ndarray = q_tensor.cpu().numpy().flatten()

        # 对非法动作施加极大负值惩罚 / Apply large negative to illegal actions
        mask = obs_data.legal_act.astype(bool)
        masked_q = np.where(mask, q_numpy, -1e9)

        act = int(np.argmax(masked_q))
        val = float(q_numpy[act])
        return act, val

    # =========================================================================
    # learn — 学习入口 / Learning Entry Point
    # =========================================================================

    def learn(self, list_sample_data: List[SampleData]) -> dict:
        """
        从经验样本中学习 — 委托给 Algorithm.learn()。
        Learn from experience samples — delegates to Algorithm.learn().

        DQN 特点: 每步学习 (off-policy)。
        样本被推入 replay buffer, 从 buffer 中采样 minibatch 计算 Double DQN loss。
        DQN feature: learns per step (off-policy).
        Samples are pushed into replay buffer; minibatches are sampled for
        Double DQN loss computation.

        Args / 参数:
            list_sample_data (List[SampleData]): 单步转移样本列表
                                                 List of single-step transition samples

        Returns / 返回:
            dict: 学习统计数据 / Learning statistics
                {
                    "loss": float,          # MSE loss
                    "q_mean": float,        # Q值均值 / mean Q-value
                    "epsilon": float,       # 当前探索率 / current epsilon
                    "learn_count": int,     # 学习迭代数 / learn iteration
                    "buffer_size": int,     # 缓冲区大小 / buffer size
                }
        """
        if list_sample_data is None or len(list_sample_data) == 0:
            return {
                "loss": 0.0,
                "q_mean": 0.0,
                "epsilon": self._algorithm.epsilon,
                "learn_count": self._algorithm.learn_count,
                "buffer_size": self._algorithm.buffer_size,
            }

        # 委托 Algorithm 学习 / Delegate to Algorithm for learning
        stats = self._algorithm.learn(list_sample_data)

        return stats

    # =========================================================================
    # save_model / load_model — 模型持久化 / Model Persistence
    # =========================================================================

    def save_model(self, path: Optional[str] = None, id: str = "latest") -> None:
        """
        保存模型到检查点文件 (torch.save)。
        Save model to checkpoint file (torch.save).

        保存内容包括: 在线网络权重、目标网络权重、优化器状态、
        epsilon、训练步数等元数据。
        Saves: online network weights, target network weights, optimizer state,
        epsilon, training steps, and other metadata.

        Args / 参数:
            path (Optional[str]): 保存目录路径 (默认 "outputs/models")
                                   Save directory path (default "outputs/models")
            id (str): 检查点标识符 / Checkpoint identifier (default "latest")
        """
        if path is None:
            path = "outputs/models"

        os.makedirs(path, exist_ok=True)
        filepath = os.path.join(path, f"dqn_{str(id)}.pth")
        self._algorithm.save_model(filepath)
        self._log("info", f"Model saved to {filepath} (steps={self._algorithm.total_steps})")

    def load_model(self, path: Optional[str] = None, id: str = "latest") -> None:
        """
        从检查点文件加载模型 (torch.load)。
        Load model from checkpoint file (torch.load).

        加载内容包括: 在线网络权重、目标网络权重、优化器状态、
        epsilon、训练步数等元数据。
        Loads: online network weights, target network weights, optimizer state,
        epsilon, training steps, and other metadata.

        Args / 参数:
            path (Optional[str]): 加载目录路径 (默认 "outputs/models")
                                   Load directory path (default "outputs/models")
            id (str): 检查点标识符 / Checkpoint identifier (default "latest")
        """
        if path is None:
            path = "outputs/models"

        filepath = os.path.join(path, f"dqn_{str(id)}.pth")
        if not os.path.exists(filepath):
            self._log("warning", f"Model file not found: {filepath}; using fresh model")
            return

        try:
            self._algorithm.load_model(filepath)
            self._log(
                "info",
                f"Model loaded from {filepath} "
                f"(steps={self._algorithm.total_steps}, epsilon={self._algorithm.epsilon:.4f})",
            )
        except Exception as e:
            self._log("warning", f"Failed to load model from {filepath}: {e}; using fresh model")

    # =========================================================================
    # observation_process — 观测处理 / Observation Processing
    # =========================================================================

    def observation_process(self, env_obs: Union[dict, np.ndarray]) -> ObsData:
        """
        处理环境观测, 生成 ObsData — 委托给 Preprocessor.observation_process()。
        Process env obs → ObsData — delegates to Preprocessor.observation_process().

        Args / 参数:
            env_obs: 环境观测 / Environment observation dict

        Returns / 返回:
            ObsData: 包含 feature (50,) 和 legal_act (8,) 的结构化数据
        """
        return self._preprocessor.observation_process(env_obs)

    # =========================================================================
    # action_process — 动作处理 / Action Processing
    # =========================================================================

    def action_process(self, act_data: ActData) -> int:
        """
        处理动作数据, 提取可执行的动作索引 — 委托给 Preprocessor.action_process()。
        Process action data → executable action index — delegates to Preprocessor.

        Args / 参数:
            act_data (ActData): 结构化动作数据 / Structured action data

        Returns / 返回:
            int: 可执行的动作索引 (0-7) / Executable action index
        """
        return self._preprocessor.action_process(act_data)

    # =========================================================================
    # reward_shaping — 奖励塑形 / Reward Shaping
    # =========================================================================

    def reward_shaping(
        self,
        reward: float,
        env_obs: Optional[Dict[str, Any]] = None,
        prev_env_obs: Optional[Dict[str, Any]] = None,
        action: Optional[int] = None,
    ) -> float:
        """
        奖励塑形 — 委托给 definition.reward_shaping()。
        Reward shaping — delegates to definition.reward_shaping().

        Args / 参数:
            reward (float): 原始环境奖励 / Raw environment reward
            env_obs: 当前环境观测 (可选) / Current observation (optional)
            prev_env_obs: 前一环境观测 (可选) / Previous observation (optional)
            action: 执行的动作 (可选) / Action taken (optional)

        Returns / 返回:
            float: 塑形后的奖励 / Shaped reward
        """
        return _reward_shaping(reward)

    # =========================================================================
    # 属性代理 / Property Delegates
    # =========================================================================

    @property
    def epsilon(self) -> float:
        """当前探索率 / Current exploration rate"""
        return self._algorithm.epsilon

    @property
    def total_steps(self) -> int:
        """全局训练步数 / Global training steps"""
        return self._algorithm.total_steps

    @property
    def buffer_size(self) -> int:
        """回放缓冲区当前大小 / Current replay buffer size"""
        return self._algorithm.buffer_size

    @property
    def learn_count(self) -> int:
        """学习迭代计数 / Learning iteration count"""
        return self._algorithm.learn_count

    @property
    def last_loss(self) -> float:
        """最近一次 loss 值 / Most recent loss value"""
        return self._algorithm.last_loss

    def __repr__(self) -> str:
        return (
            f"DQNAgent(epsilon={self.epsilon:.4f}, "
            f"steps={self.total_steps}, "
            f"buffer={self.buffer_size}, "
            f"learn_count={self.learn_count})"
        )
