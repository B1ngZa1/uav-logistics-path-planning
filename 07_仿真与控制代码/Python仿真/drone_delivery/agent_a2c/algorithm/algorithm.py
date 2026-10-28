# -*- coding: utf-8 -*-
"""
================================================================================
algorithm/algorithm.py -- A2C 算法实现 / A2C Algorithm Implementation
================================================================================
职责 / Responsibility:
    实现 Advantage Actor-Critic 学习算法:
    - 计算 n-step 折扣回报与优势函数 / Compute n-step returns and advantages
    - 计算 actor loss: -log_pi(a|s) * advantage / Compute actor loss
    - 计算 critic loss: MSE(V(s), return) / Compute critic loss
    - 熵正则化 + 梯度反向传播 + 梯度裁剪 / Entropy bonus + backward + clip grad

参考 / Reference:
    - kaiwu_infinity_valley agent_a2c 的 Algorithm 实现
    - A2C paper: "Asynchronous Methods for Deep Reinforcement Learning" (Mnih et al., 2016)
    - agent_dqn 的 Algorithm 学习模式
领域适配 / Domain Adaptation:
    - 纯全连接网络, 无 CNN 分支 / Pure MLP network, no CNN branch
    - 完整 episode 批量学习 (on-policy) / Full episode batch learning (on-policy)
================================================================================
"""

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from agent_a2c.conf.conf import Config as config


class Algorithm:
    """
    A2C 算法 / A2C Algorithm

    每轮收集完整 episode 样本后调用 learn() 进行批量更新。
    Calls learn() with full episode samples collected each round.

    Usage / 用法::

        algo = Algorithm(model, device="cpu")
        stats = algo.learn(samples)  # list of SampleData / 样本列表
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = config.DEVICE,
        lr: float = config.LR,
        gamma: float = config.GAMMA,
        entropy_coef: float = config.ENTROPY_COEF,
        value_coef: float = config.VALUE_COEF,
        max_grad_norm: float = config.MAX_GRAD_NORM,
    ) -> None:
        """
        Initialize the A2C algorithm.

        Args / 参数:
            model (nn.Module): Actor-Critic 网络 / Actor-Critic network
            device (str): 计算设备 / Computation device
            lr (float): 学习率 / Learning rate
            gamma (float): 折扣因子 / Discount factor
            entropy_coef (float): 熵系数 / Entropy coefficient
            value_coef (float): 价值损失权重 / Value loss weight
            max_grad_norm (float): 梯度裁剪阈值 / Gradient clipping threshold
        """
        self._device: torch.device = torch.device(device)
        self._model: nn.Module = model.to(self._device)
        self._gamma: float = gamma
        self._entropy_coef: float = entropy_coef
        self._value_coef: float = value_coef
        self._max_grad_norm: float = max_grad_norm

        self._optimizer: torch.optim.Adam = torch.optim.Adam(
            self._model.parameters(), lr=lr
        )

        self._train_step: int = 0
        """全局训练步数 / Global training step count"""

    # =========================================================================
    # learn -- A2C 批量学习 / A2C Batch Learning
    # =========================================================================

    def learn(self, list_sample_data: List) -> Dict[str, float]:
        """
        从完整 episode 的样本中学习 / Learn from full episode samples.

        A2C 流程 / A2C Process:
            1. 提取所有样本字段 / Extract all sample fields
            2. 计算折扣回报 (n-step returns) / Compute discounted returns
            3. 前向传播获取 logits 和 value / Forward pass for logits and values
            4. 计算优势 = return - V(s) / Compute advantage = return - V(s)
            5. 计算 actor loss = -log_pi(a|s) * advantage / Compute actor loss
            6. 计算 critic loss = MSE(V(s), return) / Compute critic loss
            7. 添加熵 bonus / Add entropy bonus
            8. 反向传播 + 梯度裁剪 + 优化器步进 / Backward + clip grad + optimizer step

        Args / 参数:
            list_sample_data (List[SampleData]): 完整 episode 的样本列表 / Full episode samples
                Each sample has: .state (50,), .action (int), .reward (float),
                .log_prob (float), .value (float), .done (bool)

        Returns / 返回:
            Dict[str, float]: 学习统计 / Learning statistics
                {"loss", "policy_loss", "value_loss", "entropy", "grad_norm"}
        """
        if len(list_sample_data) == 0:
            return {
                "loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0,
                "entropy": 0.0, "grad_norm": 0.0,
            }

        # ---- Step 1: 提取样本字段 / Extract sample fields ----
        obs_np: np.ndarray = np.stack([s.state for s in list_sample_data])
        actions_np: np.ndarray = np.array(
            [s.action for s in list_sample_data], dtype=np.int64
        )
        rewards_np: np.ndarray = np.array(
            [s.reward for s in list_sample_data], dtype=np.float32
        )
        dones_np: np.ndarray = np.array(
            [s.done for s in list_sample_data], dtype=np.float32
        )

        # 转为 tensor / Convert to tensor
        obs: torch.Tensor = torch.tensor(obs_np, dtype=torch.float32).to(self._device)
        actions: torch.Tensor = torch.tensor(
            actions_np, dtype=torch.long
        ).to(self._device)
        rewards: torch.Tensor = torch.tensor(
            rewards_np, dtype=torch.float32
        ).to(self._device)

        # ---- Step 2: 计算折扣回报 (n-step returns) / Compute discounted returns ----
        returns: List[float] = self._compute_returns(rewards_np, dones_np)
        returns_tensor: torch.Tensor = torch.tensor(
            np.array(returns, dtype=np.float32)
        ).to(self._device)

        # ---- Step 3: 前向传播 / Forward pass ----
        self._model.train()
        logits, values = self._model(obs)
        values = values.squeeze(-1)  # (B,) / 压缩维度

        # ---- Step 4: 计算动作分布与对数概率 / Compute action distribution & log prob ----
        dist = torch.distributions.Categorical(logits=logits)
        log_probs: torch.Tensor = dist.log_prob(actions)  # (B,) / 所选动作的对数概率
        entropy: torch.Tensor = dist.entropy().mean()     # 标量 / scalar

        # ---- Step 5: 计算优势 / Compute advantages ----
        advantages: torch.Tensor = returns_tensor - values.detach()

        # 优势标准化 / Advantage normalization
        adv_mean: torch.Tensor = advantages.mean()
        adv_std: torch.Tensor = advantages.std() + 1e-8
        advantages: torch.Tensor = (advantages - adv_mean) / adv_std

        # ---- Step 6: 计算损失 / Compute losses ----
        # Actor loss: -log_pi * advantage (策略梯度) / Policy gradient
        policy_loss: torch.Tensor = (-log_probs * advantages.detach()).mean()

        # Critic loss: MSE(V(s), return) / Value loss
        value_loss: torch.Tensor = F.mse_loss(values, returns_tensor)

        # 总损失: policy_loss + value_coef * value_loss - entropy_coef * entropy
        # Total loss with entropy bonus (encourage exploration / 鼓励探索)
        loss: torch.Tensor = (
            policy_loss
            + self._value_coef * value_loss
            - self._entropy_coef * entropy
        )

        # ---- Step 7: 反向传播 + 梯度裁剪 / Backpropagation + Gradient clipping ----
        self._optimizer.zero_grad()
        loss.backward()
        grad_norm_val = nn.utils.clip_grad_norm_(
            self._model.parameters(), self._max_grad_norm
        )
        self._optimizer.step()

        self._train_step += 1

        # ---- Step 8: 返回统计 / Return statistics ----
        return {
            "loss": float(loss.detach().cpu().item()),
            "policy_loss": float(policy_loss.detach().cpu().item()),
            "value_loss": float(value_loss.detach().cpu().item()),
            "entropy": float(entropy.detach().cpu().item()),
            "grad_norm": float(grad_norm_val.detach().cpu().item())
            if isinstance(grad_norm_val, torch.Tensor)
            else float(grad_norm_val),
        }

    # =========================================================================
    # 折扣回报计算 / Discounted Return Computation
    # =========================================================================

    def _compute_returns(
        self,
        rewards: np.ndarray,
        dones: np.ndarray,
    ) -> List[float]:
        """
        向后递推计算 n-step 折扣回报。
        Compute n-step discounted returns backwards.

        Returns = r_t + gamma * (1 - done_t) * Returns_{t+1}

        When done_t = True, stop propagating future rewards.
        当 done_t = True 时, 不传播未来奖励。

        Args / 参数:
            rewards (np.ndarray): 奖励数组 (T,) / Reward array
            dones (np.ndarray): 终止标记 (T,) / Done flags (1 = terminal)

        Returns / 返回:
            List[float]: 折扣回报列表 (T,) / Discounted returns list
        """
        T: int = len(rewards)
        rets: List[float] = [0.0] * T
        R: float = 0.0

        for t in reversed(range(T)):
            if dones[t] > 0.5:
                R = 0.0  # 重置回报 / Reset return at terminal
            R = float(rewards[t]) + self._gamma * R
            rets[t] = R

        return rets

    # =========================================================================
    # 模型持久化 / Model Persistence
    # =========================================================================

    def save_model(self, filepath: str) -> None:
        """
        保存模型参数 / Save model parameters.

        Args / 参数:
            filepath (str): 保存路径 / Save path
        """
        torch.save(self._model.state_dict(), filepath)

    def load_model(self, filepath: str) -> None:
        """
        加载模型参数 / Load model parameters.

        Args / 参数:
            filepath (str): 模型文件路径 / Model file path
        """
        self._model.load_state_dict(
            torch.load(filepath, map_location=self._device)
        )

    # =========================================================================
    # 属性 / Properties
    # =========================================================================

    @property
    def train_step(self) -> int:
        """全局训练步数 / Global training step count"""
        return self._train_step

    @property
    def model(self) -> nn.Module:
        """当前模型 / Current model"""
        return self._model

    def train(self) -> None:
        """设置模型为训练模式 / Set model to training mode"""
        self._model.train()

    def eval(self) -> None:
        """设置模型为评估模式 / Set model to evaluation mode"""
        self._model.eval()
