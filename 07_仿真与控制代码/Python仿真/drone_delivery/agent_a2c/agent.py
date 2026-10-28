#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
================================================================================
agent.py -- Advantage Actor-Critic (A2C) Agent / A2C 智能体
================================================================================

Responsibility / 职责:
    Integrates Actor-Critic model, A2C algorithm, and preprocessor into a
    complete RL agent for drone food delivery (Level 2).
    将 Actor-Critic 模型、A2C 算法、预处理器组合为完整的无人机配送 RL Agent。

    - predict(): Sample action from Categorical distribution (training)
    - exploit(): Argmax greedy action selection (evaluation)
    - learn(): Delegate to Algorithm for batch on-policy learning
    - observation_process(): Env observation -> ObsData
    - action_process(): ActData -> action index
    - reward_shaping(): Reward shaping (pass-through)
    - save_model() / load_model(): Model persistence via torch.save/load

Reference / 参考:
    - kaiwu_infinity_valley/agent_a2c/agent.py (A2C Agent pattern)
    - agent_q_learning/agent.py (drone delivery Agent interface)
    - Mnih et al., "Asynchronous Methods for Deep RL", ICML 2016

Domain Adaptation / 领域适配:
    - Infinity Valley 8-dir + 153-dim -> Drone Delivery 8-dir + 50-dim
    - Kaiwu BaseAgent -> Standalone Agent class (smoke test compatible)
    - Level 2: 2D random merchant delivery / 层次二: 2D随机商家配送
================================================================================
"""

import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from agent_a2c.conf.conf import Config
from agent_a2c.model.model import ActorCritic
from agent_a2c.feature.definition import (
    ObsData, ActData, SampleData, reward_shaping,
)
from agent_a2c.feature.preprocessor import Preprocessor
from agent_a2c.algorithm.algorithm import Algorithm


class Agent:
    """
    A2C Agent -- Drone Food Delivery (Level 2).
    A2C 智能体 -- 无人机外卖配送 (层次二).

    Uses Advantage Actor-Critic for deep reinforcement learning.
    Integrates ActorCritic network, Algorithm, and Preprocessor.

    Core Components / 核心组件:
        - model (ActorCritic): Dual-head Actor-Critic network
        - algorithm (Algorithm): A2C learning algorithm
        - preprocessor (Preprocessor): Feature preprocessor
        - optimizer (Adam): PyTorch Adam optimizer

    Smoke Test Usage / 烟雾测试用法::

        agent = Agent()
        env_obs, _ = env.reset()
        agent.reset(env_obs)
        obs_data = agent.observation_process(env_obs)
        while not done:
            act_data = agent.predict([obs_data])
            act = agent.action_process(act_data)
            obs, reward, terminated, truncated, _ = env.step(act)
            shaped_reward = agent.reward_shaping(reward)
            next_obs_data = agent.observation_process(obs)
            collector.add(obs=obs_data, _obs=next_obs_data, act=act,
                         rew=shaped_reward, done=(terminated or truncated))
            obs_data = next_obs_data
        collector.process_last_frame()
        agent.learn(collector.get_game_data())
    """

    def __init__(self,
                 agent_type: str = "player",
                 device: Any = None,
                 logger: Any = None,
                 monitor: Any = None) -> None:
        """Initialize A2C agent. / 初始化A2C智能体。"""
        self.agent_type = agent_type
        self.logger = logger
        self.monitor = monitor

        if device is None:
            device = Config.DEVICE
        self.device = torch.device(device)
        self._log("info", f"A2C Agent on device: {self.device}")

        self.model = ActorCritic(
            state_dim=Config.STATE_DIM,
            hidden_dim=Config.HIDDEN_DIM,
            action_size=Config.ACTION_SIZE,
        ).to(self.device)

        self.preprocessor = Preprocessor(grid_size=40)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=Config.LR)

        self.algorithm = Algorithm(
            model=self.model, device=str(self.device),
            lr=Config.LR, gamma=Config.GAMMA,
            entropy_coef=Config.ENTROPY_COEF,
            value_coef=Config.VALUE_COEF,
            max_grad_norm=Config.MAX_GRAD_NORM,
        )
        self.optimizer = self.algorithm._optimizer

        self.train_steps: int = 0
        self.episodes: int = 0
        self.predict_count: int = 0

    def _log(self, level: str, message: str) -> None:
        """Safe logging. / 安全日志。"""
        if self.logger is None:
            return
        m = getattr(self.logger, level, None) or getattr(self.logger, "info", None)
        if m:
            m(message)

    def reset(self, env_obs: Any = None) -> None:
        """Reset at episode start. / 回合开始时重置。"""
        self.episodes += 1
        self.preprocessor = Preprocessor(grid_size=40)

    def predict(self, list_obs_data: List[ObsData]) -> ActData:
        """
        Sample action from Categorical distribution (training).
        从Categorical分布中随机采样动作 (训练模式)。

        Args:
            list_obs_data: List of ObsData (uses first element)
        Returns:
            ActData with action index, state value, and log probability
        """
        if not list_obs_data:
            return ActData(act=0, val=0.0, log_prob=0.0)

        obs_data = list_obs_data[0]
        obs_tensor = torch.tensor(
            obs_data.feature, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        legal_tensor: Optional[torch.Tensor] = None
        if obs_data.legal_act is not None:
            legal_tensor = torch.tensor(
                obs_data.legal_act, dtype=torch.float32, device=self.device
            ).unsqueeze(0)

        self.model.eval()
        with torch.no_grad():
            action_int, log_prob_t, value_t = self.model.get_action(
                obs=obs_tensor, legal_mask=legal_tensor,
            )

        self.predict_count += 1
        return ActData(
            act=action_int,
            val=float(value_t.cpu().item()),
            log_prob=float(log_prob_t.cpu().item()),
        )

    def exploit(self, env_obs: Any) -> int:
        """
        Greedy action selection (evaluation). / 贪心动作选择 (评估)。

        Returns:
            int: Executable action index (0-7)
        """
        obs_data = self.observation_process(env_obs)
        obs_tensor = torch.tensor(
            obs_data.feature, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        legal_tensor: Optional[torch.Tensor] = None
        if obs_data.legal_act is not None:
            legal_tensor = torch.tensor(
                obs_data.legal_act, dtype=torch.float32, device=self.device
            ).unsqueeze(0)

        self.model.eval()
        with torch.no_grad():
            logits, _ = self.model(obs_tensor)
        if legal_tensor is not None:
            logits = logits.clone()
            logits[legal_tensor < 0.5] = -1e10
        return int(torch.softmax(logits, dim=-1).argmax(dim=-1).cpu().item())

    def learn(self, list_sample_data: List[SampleData]) -> Dict[str, float]:
        """
        Batch on-policy A2C learning. / 批量同策略A2C学习。

        Returns:
            dict: learning metrics (loss, policy_loss, value_loss, entropy, grad_norm)
        """
        if not list_sample_data:
            return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0,
                    "entropy": 0.0, "grad_norm": 0.0}
        metrics = self.algorithm.learn(list_sample_data)
        self.train_steps += 1
        return metrics

    def observation_process(self, env_obs: Any) -> ObsData:
        """
        Process env observation -> ObsData. / 环境观测 -> ObsData。
        Delegates to Preprocessor.observation_process().
        """
        return self.preprocessor.observation_process(env_obs)

    def action_process(self, act_data: ActData) -> int:
        """
        Extract action index from ActData. / 从ActData提取动作索引。
        """
        a = act_data.act
        return int(max(0, min(Config.ACTION_SIZE - 1, a))) if a is not None else 0

    def reward_shaping(self, reward: float, env_obs: Any = None) -> float:
        """
        Reward shaping (pass-through). / 奖励塑形 (直通)。
        Delegates to definition.reward_shaping().
        """
        return reward_shaping(reward, env_obs)

    def save_model(self, path: Optional[str] = None, id: str = "latest") -> None:
        """Save model to checkpoint. / 保存模型到检查点。"""
        if path is None:
            path = "outputs/models"
        os.makedirs(path, exist_ok=True)
        fp = os.path.join(path, f"model.ckpt-{str(id)}.pkl")
        torch.save({k: v.clone().cpu() for k, v in self.model.state_dict().items()}, fp)
        self._log("info", f"Model saved to {fp}")

    def load_model(self, path: Optional[str] = None, id: str = "latest") -> None:
        """Load model from checkpoint. / 从检查点加载模型。"""
        if path is None:
            path = "outputs/models"
        fp = os.path.join(path, f"model.ckpt-{str(id)}.pkl")
        if not os.path.exists(fp):
            self._log("warning", f"Model file not found: {fp}")
            return
        try:
            self.model.load_state_dict(torch.load(fp, map_location=self.device))
            self._log("info", f"Model loaded from {fp}")
        except Exception as e:
            self._log("warning", f"Failed to load model: {e}")

    def __repr__(self) -> str:
        return f"Agent(episodes={self.episodes}, train_steps={self.train_steps})"
