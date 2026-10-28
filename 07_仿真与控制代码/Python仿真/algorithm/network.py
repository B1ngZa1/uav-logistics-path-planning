"""
神经网络定义
项目：基于深度强化学习的多无人机协同物流配送路径规划
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorNetwork(nn.Module):
    """Actor策略网络"""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU()
        )
        self.mean = nn.Linear(hidden_dim // 4, action_dim)
        self.log_std = nn.Linear(hidden_dim // 4, action_dim)

    def forward(self, state):
        x = self.shared(state)
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(-20, 2)
        return mean, log_std


class CriticNetwork(nn.Module):
    """Critic价值网络"""

    def __init__(self, state_dim, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, 1)
        )

    def forward(self, state):
        return self.network(state)


class CentralizedCritic(nn.Module):
    """集中式Critic（CTDE训练时使用）"""

    def __init__(self, global_state_dim, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(global_state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, global_state):
        return self.network(global_state)


def init_weights(module):
    """网络权重初始化"""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=0.01)
        nn.init.constant_(module.bias, 0)
