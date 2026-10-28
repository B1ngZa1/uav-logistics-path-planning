# -*- coding: utf-8 -*-
"""
================================================================================
model/model.py — Dueling DQN 网络模型 / Dueling DQN Network Model
================================================================================
职责 / Responsibility:
    实现 Dueling DQN 网络架构:
    - 特征提取层: Linear(50, HIDDEN_DIM) → ReLU → Linear(HIDDEN_DIM, 128) → ReLU
    - 价值流 V(s):   Linear(128, 64) → ReLU → Linear(64, 1)
    - 优势流 A(s,a): Linear(128, 64) → ReLU → Linear(64, 8)
    - 融合: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
    Implements Dueling DQN architecture with orthogonal initialization.

参考 / Reference:
    - Wang et al., "Dueling Network Architectures for Deep RL", ICML 2016
    - Gorge Walk DIY agent 的 DuelingDQN 实现
    - PyTorch orthogonal initialization best practices
领域适配 / Domain Adaptation:
    - 输入: 50维观测向量 → 输出: 8维Q值 (对应8个方向动作)
    - Input: 50-dim obs → Output: 8-dim Q-values (for 8 direction actions)
================================================================================
"""

from typing import Tuple

import torch
import torch.nn as nn

from conf.conf import config


# =============================================================================
# 正交初始化工具 / Orthogonal Initialization Utilities
# =============================================================================

def orthogonal_init(layer: nn.Linear, gain: float = 1.0) -> None:
    """
    对 Linear 层执行正交初始化。
    Apply orthogonal initialization to a Linear layer.

    使用正交矩阵初始化权重 (PyTorch 默认使用 kaiming_uniform),
    有助于提高Dueling DQN训练稳定性。
    Using orthogonal init (instead of default kaiming_uniform)
    improves training stability for Dueling DQN.

    Args / 参数:
        layer (nn.Linear): 目标线性层 / Target linear layer
        gain (float): 正交初始化增益因子 / Gain factor for orthogonal init
    """
    if hasattr(layer, "weight") and layer.weight.ndim >= 2:
        nn.init.orthogonal_(layer.weight, gain=gain)
    if hasattr(layer, "bias") and layer.bias is not None:
        nn.init.constant_(layer.bias, 0.0)


# =============================================================================
# Dueling DQN 网络 / Dueling DQN Network
# =============================================================================

class DuelingDQN(nn.Module):
    """
    Dueling DQN 网络 — 将状态值函数与动作优势函数分离学习。
    Dueling DQN — learns state-value and action-advantage separately.

    架构 / Architecture::

        input (50) ──┐
                      ├─► feature_layer ──► shared_rep (128)
                                           │
                         ┌─────────────────┴────────────────┐
                         ▼                                   ▼
                    value_head                          advantage_head
                    V(s) = (1,)                         A(s,a) = (8,)
                         │                                   │
                         └────────────┬──────────────────────┘
                                      ▼
                            Q(s,a) = V(s) + A(s,a) - mean(A)

    分离表示使网络能够学习哪些状态有价值, 而不必学习每个动作的效果。
    Separated representation allows learning state value without needing
    to learn the effect of every action.

    Usage / 用法::

        model = DuelingDQN()
        q_values = model(obs_tensor)         # shape: (batch, 8)
        v_value, advantage = model.forward_decompose(obs_tensor)  # decomposed
    """

    def __init__(
        self,
        state_dim: int = config.STATE_DIM,
        action_size: int = config.ACTION_SIZE,
        hidden_dim: int = config.HIDDEN_DIM,
    ) -> None:
        """
        Initialize Dueling DQN.

        Args / 参数:
            state_dim (int): 观测向量维度 / Observation dim (default: 50)
            action_size (int): 动作数 / Number of actions (default: 8)
            hidden_dim (int): 隐藏层维度 / Hidden dim (default: 256)
        """
        super(DuelingDQN, self).__init__()

        self._state_dim: int = state_dim
        self._action_size: int = action_size
        self._hidden_dim: int = hidden_dim

        # -----------------------------------------------------------------
        # 特征提取层 / Feature Extraction Layer
        # input(50) → Linear → ReLU → Linear(256→128) → ReLU → rep(128)
        # -----------------------------------------------------------------
        self.feature_layer: nn.Sequential = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(inplace=True),
        )

        # -----------------------------------------------------------------
        # 价值头 V(s) — 输出标量 / Value Head — outputs scalar
        # rep(128) → Linear(128,64) → ReLU → Linear(64,1) → V(s)
        # -----------------------------------------------------------------
        self.value_head: nn.Sequential = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

        # -----------------------------------------------------------------
        # 优势头 A(s,a) — 输出8维 / Advantage Head — outputs (8,)
        # rep(128) → Linear(128,64) → ReLU → Linear(64,8) → A(s,a)
        # -----------------------------------------------------------------
        self.advantage_head: nn.Sequential = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, action_size),
        )

        # 正交初始化 / Apply orthogonal initialization
        self._init_weights()

    def _init_weights(self) -> None:
        """对所有 Linear 层执行正交初始化 / Orthogonal init all Linear layers"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                orthogonal_init(module, gain=1.0)

    # -------------------------------------------------------------------------
    # 前向传播 / Forward
    # -------------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播: 返回 Q(s,a) = V(s) + A(s,a) - mean(A)。
        Forward: returns Q(s,a) = V(s) + A(s,a) - mean(A).

        Args / 参数:
            x (Tensor): 输入观测 (batch, 50) / Input observation

        Returns / 返回:
            Tensor: Q值 (batch, 8) / Q-values
        """
        features: torch.Tensor = self.feature_layer(x)

        value: torch.Tensor = self.value_head(features)          # (batch, 1)
        advantage: torch.Tensor = self.advantage_head(features)  # (batch, 8)

        # Q(s,a) = V + A - mean(A)
        # 减去均值确保 identifiability: 给定Q值时可唯一确定V和A
        # Subtracting mean ensures identifiability
        q_values: torch.Tensor = value + advantage - advantage.mean(dim=1, keepdim=True)

        return q_values

    # -------------------------------------------------------------------------
    # 分解前向（调试用） / Decomposed Forward (for Debugging)
    # -------------------------------------------------------------------------

    def forward_decompose(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回分开的 V(s) 和 A(s,a), 用于分析/调试。
        Returns separate V(s) and A(s,a) for analysis/debugging.

        Args / 参数:
            x (Tensor): 输入观测 (batch, 50) / Input observation

        Returns / 返回:
            Tuple[Tensor, Tensor]: (value, advantage)
                value     — V(s)  (batch, 1)
                advantage — A(s,a) (batch, 8)
        """
        features: torch.Tensor = self.feature_layer(x)
        value: torch.Tensor = self.value_head(features)
        advantage: torch.Tensor = self.advantage_head(features)
        return value, advantage

    # -------------------------------------------------------------------------
    # 属性 / Properties
    # -------------------------------------------------------------------------

    @property
    def state_dim(self) -> int:
        """观测维度 / Observation dimension"""
        return self._state_dim

    @property
    def action_size(self) -> int:
        """动作数 / Action count"""
        return self._action_size
