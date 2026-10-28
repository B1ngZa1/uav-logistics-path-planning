# -*- coding: utf-8 -*-
"""
================================================================================
algorithm/algorithm.py — Double Dueling DQN 算法 / Double Dueling DQN Algorithm
================================================================================
职责 / Responsibility:
    实现 Double DQN + Dueling 架构的训练算法:
    - 维护经验回放缓冲区 (deque, maxlen=100000)
    - 从缓冲区采样 minibatch 计算 Double DQN loss
    - 使用 MSE: (r + gamma * Q_target(s', argmax Q_online(s',·)) - Q_online(s,a))^2
    - 周期性地将在线网络权重硬拷贝到目标网络 (TARGET_UPDATE=100)
    - epsilon 每步衰减
    Implement Double DQN with Dueling architecture training algorithm.

参考 / Reference:
    - van Hasselt et al., "Deep RL with Double Q-learning", AAAI 2016
    - Wang et al., "Dueling Network Architectures", ICML 2016
    - Gorge Walk DIY agent 的 Algorithm 实现
领域适配 / Domain Adaptation:
    - 50维观测 → DuelingDQN → 8个Q值
    - 配送环境: 奖励稀疏, 需要 replay buffer 稳定训练
    - 50-dim obs → DuelingDQN → 8 Q-values
    - Delivery env: sparse rewards, replay buffer for stable training
================================================================================
"""

from typing import List, Optional, Tuple
from collections import deque
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from conf.conf import config
from feature.definition import SampleData
from model.model import DuelingDQN


# =============================================================================
# Replay Buffer — 经验回放缓冲区
# =============================================================================

class ReplayBuffer:
    """
    经验回放缓冲区 — 固定容量 FIFO 队列 / Fixed-capacity FIFO replay buffer.

    使用 deque 实现, 当缓冲区满时自动丢弃最旧样本。
    Implemented with deque; oldest samples auto-dropped when full.
    """

    def __init__(self, capacity: int = config.BUFFER_SIZE) -> None:
        """
        Initialize replay buffer.

        Args / 参数:
            capacity (int): 最大样本数 / Maximum number of samples
        """
        self._capacity: int = capacity
        self._buffer: deque = deque(maxlen=capacity)

    def push(self, sample: SampleData) -> None:
        """
        添加一个样本到缓冲区 / Push a single sample into buffer.

        Args / 参数:
            sample (SampleData): 转移样本 / Transition sample
        """
        self._buffer.append(sample)

    def sample(self, batch_size: int) -> List[SampleData]:
        """
        随机采样一批样本 / Randomly sample a batch.

        Args / 参数:
            batch_size (int): 批大小 / Batch size

        Returns / 返回:
            List[SampleData]: 采样的样本列表 / Sampled transitions
        """
        return random.sample(list(self._buffer), min(batch_size, len(self._buffer)))

    def __len__(self) -> int:
        """返回当前缓冲区大小 / Return current buffer size"""
        return len(self._buffer)

    @property
    def is_ready(self) -> bool:
        """
        缓冲区是否已准备好用于学习 (样本数 >= 预热阈值)。
        Whether buffer is ready for learning (samples >= warmup threshold).
        """
        return len(self._buffer) >= config.MIN_BUFFER_SIZE_FOR_LEARN


# =============================================================================
# Double Dueling DQN Algorithm
# =============================================================================

class Algorithm:
    """
    Double Dueling DQN 训练算法 / Double Dueling DQN Training Algorithm

    核心逻辑 / Core logic:
        1. 维护在线网络 Q_online 和目标网络 Q_target
        2. 每步收集样本到 replay buffer
        3. buffer 预热后, 每步采样 minibatch 计算 Double DQN loss
        4. 使用 Double DQN 目标: r + gamma * Q_target(s', argmax Q_online(s',·))
        5. 每 TARGET_UPDATE 步将 Q_online 权重拷贝到 Q_target (hard update)

    Usage / 用法::

        algo = Algorithm()
        algo.learn([sample1, sample2, ...])
        # epsilon is decayed internally per step
        print(algo.epsilon)
    """

    def __init__(
        self,
        device: str = config.DEVICE,
        gamma: float = config.GAMMA,
        learning_rate: float = config.LEARNING_RATE,
        batch_size: int = config.BATCH_SIZE,
        target_update: int = config.TARGET_UPDATE,
    ) -> None:
        """
        Initialize the DQN algorithm.

        Args / 参数:
            device (str): 计算设备 / Computation device
            gamma (float): 折扣因子 / Discount factor
            learning_rate (float): 学习率 / Learning rate
            batch_size (int): 训练批大小 / Training batch size
            target_update (int): 目标网络更新间隔 / Target network update interval
        """
        # -----------------------------------------------------------------
        # 网络 / Networks
        # -----------------------------------------------------------------
        self._device: torch.device = torch.device(device)

        self._q_online: DuelingDQN = DuelingDQN().to(self._device)
        """在线网络 / Online network — 用于动作选择和梯度更新"""

        self._q_target: DuelingDQN = DuelingDQN().to(self._device)
        """目标网络 / Target network — 用于计算 TD target, 定期硬更新"""

        # 初始时目标网络与在线网络权重一致
        # Initially sync target from online
        self._hard_update()

        # -----------------------------------------------------------------
        # 优化器 / Optimizer
        # -----------------------------------------------------------------
        self._optimizer: optim.Adam = optim.Adam(
            self._q_online.parameters(), lr=learning_rate
        )
        self._loss_fn: nn.MSELoss = nn.MSELoss()

        # -----------------------------------------------------------------
        # 超参数 / Hyperparameters
        # -----------------------------------------------------------------
        self._gamma: float = gamma
        self._batch_size: int = batch_size
        self._target_update: int = target_update

        # -----------------------------------------------------------------
        # 缓冲区 / Replay Buffer
        # -----------------------------------------------------------------
        self._replay_buffer: ReplayBuffer = ReplayBuffer()

        # -----------------------------------------------------------------
        # 训练状态 / Training State
        # -----------------------------------------------------------------
        self._epsilon: float = config.EPSILON_START
        """当前探索率 / Current exploration rate"""

        self._total_steps: int = 0
        """全局训练步计数 / Global training step count"""

        self._learn_count: int = 0
        """学习迭代计数 / Learning iteration count"""

        # -----------------------------------------------------------------
        # 训练记录 / Training Logs
        # -----------------------------------------------------------------
        self._last_loss: float = 0.0
        """最近一次 loss 值 / Most recent loss value"""

        self._last_q_mean: float = 0.0
        """最近一次 Q 值均值 / Most recent mean Q-value"""

    # =========================================================================
    # learn — 学习入口 / Learning Entry Point
    # =========================================================================

    def learn(self, sample_datas: List[SampleData]) -> dict:
        """
        主学习循环: 存入样本 → 采样 minibatch → 计算 loss → 优化。
        Main learning loop: push samples → sample minibatch → compute loss → optimize.

        Args / 参数:
            sample_datas (List[SampleData]): 单步转移样本列表
                                             List of single-step transition samples

        Returns / 返回:
            dict: 学习统计数据 / Learning statistics
                {
                    "loss": float,         # MSE loss
                    "q_mean": float,       # Q值均值 / mean Q-value
                    "epsilon": float,      # 当前探索率 / current epsilon
                    "learn_count": int,    # 学习迭代数 / learn iteration
                    "buffer_size": int,    # 缓冲区大小 / buffer size
                }
        """
        # Step 1: 将样本存入回放缓冲区 / Push samples to replay buffer
        for sample in sample_datas:
            self._replay_buffer.push(sample)

        stats = {
            "loss": 0.0,
            "q_mean": 0.0,
            "epsilon": self._epsilon,
            "learn_count": self._learn_count,
            "buffer_size": len(self._replay_buffer),
        }

        # Step 2: 缓冲区预热检查 / Check if buffer is warm enough
        if not self._replay_buffer.is_ready:
            return stats

        # Step 3: 采样 minibatch / Sample minibatch
        batch = self._replay_buffer.sample(self._batch_size)

        # Step 4: 计算 Double DQN loss 并优化 / Compute Double DQN loss & optimize
        loss, q_mean = self._train_step(batch)
        self._last_loss = loss
        self._last_q_mean = q_mean

        stats["loss"] = loss
        stats["q_mean"] = q_mean

        # Step 5: 递增学习计数 / Increment learn count
        self._learn_count += 1

        # Step 6: 定期同步目标网络 / Periodic target network sync
        if self._learn_count % self._target_update == 0:
            self._hard_update()

        # Step 7: 衰减 epsilon / Decay epsilon
        self._decay_epsilon()

        return stats

    # =========================================================================
    # _train_step — 单次训练迭代 / Single Training Iteration
    # =========================================================================

    def _train_step(self, batch: List[SampleData]) -> Tuple[float, float]:
        """
        执行一次 minibatch 梯度下降。
        Perform a single minibatch gradient descent step.

        Double DQN 目标公式 / Double DQN target formula::

            target = r_s,a + gamma * Q_target(s', argmax_{a'} Q_online(s', a'))

        Args / 参数:
            batch (List[SampleData]): 采样的转移样本 / Sampled transitions

        Returns / 返回:
            Tuple[float, float]: (loss_value, q_mean)
        """
        # 构建张量 / Build tensors
        obs_batch = torch.tensor(
            np.stack([s.obs for s in batch]), dtype=torch.float32
        ).to(self._device)                  # (B, 50)

        act_batch = torch.tensor(
            [s.act for s in batch], dtype=torch.long
        ).to(self._device)                  # (B,)

        rew_batch = torch.tensor(
            [s.rew for s in batch], dtype=torch.float32
        ).to(self._device)                  # (B,)

        next_obs_batch = torch.tensor(
            np.stack([s.next_obs for s in batch]), dtype=torch.float32
        ).to(self._device)                  # (B, 50)

        done_batch = torch.tensor(
            [s.done for s in batch], dtype=torch.float32
        ).to(self._device)                  # (B,)

        # ---- 计算 Q_online(s, a) ----
        q_online_all: torch.Tensor = self._q_online(obs_batch)  # (B, 8)
        q_online_selected: torch.Tensor = q_online_all.gather(
            dim=1, index=act_batch.unsqueeze(1)
        ).squeeze(1)                                             # (B,)

        # ---- Double DQN 目标 / Double DQN target ----
        with torch.no_grad():
            # 用在线网络选动作: argmax Q_online(s', ·)
            q_online_next: torch.Tensor = self._q_online(next_obs_batch)  # (B, 8)
            next_actions: torch.Tensor = q_online_next.argmax(dim=1)       # (B,)

            # 用目标网络评价值: Q_target(s', chosen_action)
            q_target_next: torch.Tensor = self._q_target(next_obs_batch)   # (B, 8)
            q_target_selected: torch.Tensor = q_target_next.gather(
                dim=1, index=next_actions.unsqueeze(1)
            ).squeeze(1)                                                    # (B,)

            # TD target: r + gamma * Q_target(s', argmax Q_online(s',·)) * (1 - done)
            td_target: torch.Tensor = (
                rew_batch + self._gamma * q_target_selected * (1.0 - done_batch)
            )

        # ---- MSE Loss ----
        loss: torch.Tensor = self._loss_fn(q_online_selected, td_target)

        # ---- 反向传播 & 优化 / Backprop & Optimize ----
        self._optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪以稳定训练 / Gradient clipping for training stability
        torch.nn.utils.clip_grad_norm_(self._q_online.parameters(), max_norm=10.0)
        self._optimizer.step()

        q_mean = float(q_online_selected.mean().item())
        return float(loss.item()), q_mean

    # =========================================================================
    # 目标网络硬更新 / Target Network Hard Update
    # =========================================================================

    def _hard_update(self) -> None:
        """
        硬更新: 将在线网络权重完整拷贝到目标网络。
        Hard update: Copy all weights from online network to target network.
        """
        self._q_target.load_state_dict(self._q_online.state_dict())

    # =========================================================================
    # Epsilon 衰减 / Epsilon Decay
    # =========================================================================

    def _decay_epsilon(self) -> None:
        """
        按指数衰减探索率 / Exponentially decay epsilon.
        epsilon = max(epsilon * EPSILON_DECAY, EPSILON_MIN)
        """
        self._epsilon = max(
            self._epsilon * config.EPSILON_DECAY,
            config.EPSILON_MIN,
        )

    def reset_epsilon(self, value: Optional[float] = None) -> None:
        """
        重置探索率为指定值(默认为初始值) / Reset epsilon to given value (default: start).

        Args / 参数:
            value (Optional[float]): 目标 epsilon 值 / Target epsilon value
        """
        if value is None:
            self._epsilon = config.EPSILON_START
        else:
            self._epsilon = float(value)

    # =========================================================================
    # 模型保存 & 加载 / Model Save & Load
    # =========================================================================

    def save_model(self, filepath: str) -> None:
        """
        保存在线网络权重到文件 / Save online network weights to file.

        Args / 参数:
            filepath (str): 保存路径 / Save path
        """
        torch.save(
            {
                "q_online_state_dict": self._q_online.state_dict(),
                "q_target_state_dict": self._q_target.state_dict(),
                "optimizer_state_dict": self._optimizer.state_dict(),
                "epsilon": self._epsilon,
                "total_steps": self._total_steps,
                "learn_count": self._learn_count,
            },
            filepath,
        )

    def load_model(self, filepath: str) -> None:
        """
        从文件加载网络权重 / Load network weights from file.

        Args / 参数:
            filepath (str): 模型文件路径 / Model file path
        """
        checkpoint = torch.load(filepath, map_location=self._device)
        self._q_online.load_state_dict(checkpoint["q_online_state_dict"])
        self._q_target.load_state_dict(checkpoint["q_target_state_dict"])
        self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self._epsilon = checkpoint.get("epsilon", config.EPSILON_START)
        self._total_steps = checkpoint.get("total_steps", 0)
        self._learn_count = checkpoint.get("learn_count", 0)

    # =========================================================================
    # 属性 / Properties
    # =========================================================================

    @property
    def q_online(self) -> DuelingDQN:
        """在线网络 / Online network"""
        return self._q_online

    @property
    def q_target(self) -> DuelingDQN:
        """目标网络 / Target network"""
        return self._q_target

    @property
    def epsilon(self) -> float:
        """当前探索率 / Current exploration rate"""
        return self._epsilon

    @property
    def total_steps(self) -> int:
        """全局步数 / Global step count"""
        return self._total_steps

    @property
    def learn_count(self) -> int:
        """学习迭代数 / Learning iteration count"""
        return self._learn_count

    @property
    def buffer_size(self) -> int:
        """缓冲区当前大小 / Current buffer size"""
        return len(self._replay_buffer)

    @property
    def last_loss(self) -> float:
        """最近一次 loss / Most recent loss"""
        return self._last_loss

    @property
    def last_q_mean(self) -> float:
        """最近一次 Q 均值 / Most recent mean Q-value"""
        return self._last_q_mean

    def step_tick(self) -> None:
        """
        递增全局步计数 / Increment global step counter.
        在每步环境交互后调用, 用于跟踪训练进度。
        Called after each env step to track training progress.
        """
        self._total_steps += 1
