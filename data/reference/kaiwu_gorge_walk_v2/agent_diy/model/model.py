#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import torch
from torch import nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()

        # Dueling DQN: shared feature trunk + value/advantage heads.
        self.feature = nn.Sequential(
            nn.Linear(state_shape, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
        )
        self.value = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
        self.advantage = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, action_shape),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.float32)
        if x.dim() == 1:
            x = x.unsqueeze(0)

        hidden = self.feature(x.float())
        value = self.value(hidden)
        advantage = self.advantage(hidden)
        return value + advantage - advantage.mean(dim=1, keepdim=True)
