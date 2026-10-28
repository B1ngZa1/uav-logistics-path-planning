#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import os
import torch
from torch import nn


class Model(nn.Module):
    """
    Neural network model for framework compliance.

    This is a tabular Q-learning agent whose actual decision-making uses
    SparseQTable in algorithm.py.  This nn.Module exists so the KaiwuDRL
    framework can serialize/deserialize a valid PyTorch model.  The network
    is fully functional and can be used for inference, but the primary
    training loop operates on the sparse Q-table.

    Architecture:
        Linear(state_shape, 128) → ReLU → Linear(128, 64) → ReLU → Linear(64, action_shape)

    Weights are initialised with Kaiming uniform (ReLU nonlinearity) and
    biases are set to zero.
    """

    def __init__(self, state_shape, action_shape):
        super().__init__()

        # User-defined network
        # 用户自定义网络
        self.net = nn.Sequential(
            nn.Linear(state_shape, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, action_shape),
        )
        self._init_weights()

    def _init_weights(self):
        """Kaiming uniform init for Linear layers, zero bias."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Forward pass returning Q-values.

        Args:
            x: Tensor of shape (batch, state_shape), or 1-D tensor / numpy
               array that will be unsqueezed to (1, state_shape).

        Returns:
            Q-values tensor of shape (batch, action_shape).
        """
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.float32)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.net(x.float())

    def save_model(self, path=None, id="1"):
        """
        Save model state-dict in PyTorch .pkl format.

        Conforms to KaiwuDRL checkpoint naming: model.ckpt-{id}.pkl
        """
        path = path or "ckpt"
        os.makedirs(path, exist_ok=True)
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        torch.save(self.state_dict(), model_file_path)
        return model_file_path

    def load_model(self, path=None, id="1"):
        """
        Load model state-dict from PyTorch .pkl format.

        Conforms to KaiwuDRL checkpoint naming: model.ckpt-{id}.pkl
        """
        path = path or "ckpt"
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        self.load_state_dict(torch.load(model_file_path, weights_only=True))
        return model_file_path
