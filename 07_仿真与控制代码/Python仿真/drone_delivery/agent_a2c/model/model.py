# -*- coding: utf-8 -*-
"""
================================================================================
model/model.py — Actor-Critic 网络 (PyTorch) / Actor-Critic Network (PyTorch)
================================================================================
职责 / Responsibility:
    实现共享特征提取层的 Actor-Critic 架构:
    - 共享主干: Linear(50,256) → Tanh → Linear(256,128) → Tanh
    - Actor 头: Linear(128,8) 输出动作 logits
    - Critic 头: Linear(128,1) 输出状态价值
    Implements Actor-Critic with shared feature backbone.

    注意 / Note:
        共享层使用Tanh激活函数（参数一致性规范要求Critic使用Tanh）。
        Shared layers use Tanh activation (param consistency spec requires Tanh for Critic).

参考 / Reference:
    - kaiwu_infinity_valley agent_a2c 的 PolicyValueModel
    - agent_dqn 的 DuelingDQN 模型
领域适配 / Domain Adaptation:
    - 纯向量观测 (50维), 无图像输入, 无CNN分支
    - Pure vector observation (50-dim), no image input, no CNN branch
================================================================================
"""

from typing import Tuple, Optional

import torch
import torch.nn as nn
from torch.distributions import Categorical

from agent_a2c.conf.conf import Config as config


class ActorCritic(nn.Module):
    """
    Actor-Critic 网络 / Actor-Critic Network

    网络结构 / Architecture::

        Input(50) → Shared(Linear→Tanh→Linear→Tanh)
                         ├── Actor(Linear→8 action logits)
                         └── Critic(Linear→1 state value)

    Usage / 用法::

        model = ActorCritic()
        logits, value = model(obs_tensor)          # forward pass / 前向传播
        action, log_prob, value = model.get_action(obs_tensor)  # sample action / 采样动作
    """

    def __init__(
        self,
        state_dim: int = config.STATE_DIM,
        hidden_dim: int = config.HIDDEN_DIM,
        action_size: int = config.ACTION_SIZE,
    ) -> None:
        """
        Initialize the Actor-Critic network.

        Args / 参数:
            state_dim (int): 状态维度 / State dimension (50)
            hidden_dim (int): 隐藏层维度 / Hidden layer dimension (256)
            action_size (int): 动作空间大小 / Action space size (8)
        """
        super().__init__()

        # ---- 共享特征提取层 / Shared feature extractor ----
        # 使用Tanh激活（参数一致性规范：Critic必须用Tanh，避免价值估计爆炸）
        # Using Tanh (param spec: Critic must use Tanh to avoid value estimation explosion)
        self.shared: nn.Sequential = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 128),
            nn.Tanh(),
        )

        # ---- Actor 头: 输出动作 logits / Actor head: action logits ----
        self.actor: nn.Linear = nn.Linear(128, action_size)

        # ---- Critic 头: 输出状态价值 / Critic head: state value ----
        self.critic: nn.Linear = nn.Linear(128, 1)

        # 权重初始化 / Weight initialization
        self._init_weights()

    def _init_weights(self) -> None:
        """正交初始化权重（适配Tanh激活） / Orthogonal weight init (for Tanh activation)"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播: 输出动作 logits 和状态价值。
        Forward pass: output action logits and state value.

        Args / 参数:
            obs (Tensor): 观测向量 (B, 50) / Observation vector

        Returns / 返回:
            Tuple[Tensor, Tensor]:
                - logits: 动作 logits (B, 8) / Action logits
                - value: 状态价值 (B, 1) / State value
        """
        features: torch.Tensor = self.shared(obs)
        logits: torch.Tensor = self.actor(features)
        value: torch.Tensor = self.critic(features)
        return logits, value

    def get_action(
        self,
        obs: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        从策略分布中采样动作, 返回 (action_int, log_prob, value)。
        Sample action from policy distribution, return (action_int, log_prob, value).

        Args / 参数:
            obs (Tensor): 观测向量 (1, 50) / Observation vector
            legal_mask (Optional[Tensor]): 合法动作掩码 (1, 8), 1=合法 / Legal action mask

        Returns / 返回:
            Tuple[int, Tensor, Tensor]:
                - action: 采样的动作索引 (Python int) / Sampled action index
                - log_prob: 动作的对数概率 (标量 Tensor) / Log probability of action
                - value: 状态价值 (标量 Tensor) / State value
        """
        logits, value = self.forward(obs)

        # 掩码非法动作 / Mask illegal actions
        if legal_mask is not None:
            logits = logits.clone()
            logits[legal_mask < 0.5] = -1e10

        dist: Categorical = Categorical(logits=logits)
        action: torch.Tensor = dist.sample()
        log_prob: torch.Tensor = dist.log_prob(action)

        action_int: int = int(action.cpu().item())
        return action_int, log_prob, value
