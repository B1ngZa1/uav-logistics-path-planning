# -*- coding: utf-8 -*-
"""
================================================================================
model/model.py — 学生DIY神经网络模型 / Student DIY Neural Network Model
================================================================================
职责 / Responsibility:
    学生自行实现的神经网络模型骨架。
    学生的任务是设计合适的网络架构来完成无人机配送任务。
    Skeleton for a neural network model to be implemented by students.
    Students' task is to design a suitable network architecture for the
    drone delivery task.

参考 / Reference:
    - agent_dqn/model/model.py (Dueling DQN 完整实现 / full DuelingDQN reference)
    - agent_q_learning/model/model.py (SparseQTable 表格型参考)

领域适配 / Domain Adaptation:
    - 输入: 50维观测向量 / Input: 50-dim observation vector
    - 输出: 取决于学生设计 (可能是8维Q值, 策略概率, 价值估计等)
    - Output: depends on student design (could be 8-dim Q-values, policy probs,
              value estimate, etc.)

提示 / Tips:
    - 如果是DQN类方法: 输入50维, 输出8维Q值
       For DQN-style: input 50-dim, output 8-dim Q-values
    - 如果是策略梯度: 输入50维, 输出8维动作概率 + 1维价值估计
       For policy gradient: input 50-dim, output 8 action probs + 1 value estimate
    - 如果是Actor-Critic: 需要Actor网络 + Critic网络 (可共享特征层)
       For Actor-Critic: need Actor network + Critic network (can share feature layers)
================================================================================
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """
    学生自行实现的神经网络模型 / Neural network model to be implemented by students

    提示 / Tips:
        - 继承 torch.nn.Module / Inherit torch.nn.Module
        - 输入形状: (batch_size, 50)  — 50维观测向量
        - 输出形状: 取决于算法选择
        - Input shape: (batch_size, 50)  — 50-dim observation vector
        - Output shape: depends on algorithm choice

    Usage / 用法::

        model = Model()
        output = model.forward(obs_tensor)   # obs_tensor shape: (batch, 50)
    """

    def __init__(self):
        """
        初始化模型 — 在此定义网络层。
        Initialize model — define network layers here.

        TODO: 在此定义网络层 / Define network layers here
        建议 / Suggestions:
            - 定义 self.feature_layer 作为特征提取器
            - 定义 self.output_head 作为输出头
            - 可使用 nn.Linear, nn.ReLU, nn.Sequential 等
            - Define self.feature_layer as feature extractor
            - Define self.output_head as output head
            - May use nn.Linear, nn.ReLU, nn.Sequential, etc.
        """
        super().__init__()
        # TODO: 在此定义网络层 / Define network layers here
        # 参考示例 / Example reference:
        # self.feature_layer = nn.Sequential(
        #     nn.Linear(50, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 128),
        #     nn.ReLU(),
        # )
        # self.output_head = nn.Linear(128, 8)  # DQN: Q-values for 8 actions
        pass

    def forward(self, x):
        """
        前向传播 — 在此实现从输入到输出的计算图。
        Forward pass — implement the computation graph from input to output.

        TODO: 在此实现前向传播 / Implement forward pass here
        建议 / Suggestions:
            - 接收 x (batch, 50) 张量
            - 经过特征提取层 → 输出头 → 返回结果
            - Accept x (batch, 50) tensor
            - Feature extractor → output head → return result

        Args / 参数:
            x (torch.Tensor): 输入张量 (batch_size, 50) / Input tensor

        Returns / 返回:
            torch.Tensor: 输出张量, 形状取决于设计 / Output tensor, shape depends on design
        """
        # TODO: 在此实现前向传播 / Implement forward pass here
        # 参考示例 / Example reference:
        # features = self.feature_layer(x)
        # output = self.output_head(features)
        # return output
        pass
