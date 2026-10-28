#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


"""
Infinite Valley Policy Network (for Policy Gradient) - Pure PyTorch Implementation

Instantiation:
    model = PolicyModel()

Input format:
    model(vec=V, img=I)
        V represents vector features, as a torch Tensor, with shape (B, *VECTOR_SHAPE);
        I represents image features, as a torch Tensor, with shape (B, *IMAGE_SHAPE).

Output format:
    Action probability distribution, as a torch Tensor, with shape (B, *OUTPUT_SHAPE);

无限之谷策略网络(用于Policy Gradient)——纯pytorch实现

实例化方法：
    model = PolicyModel()

实例输入形式：
    model(vec=V, img=I)
        V表示向量特征, 为torch Tensor, 形状为(B, *VECTOR_SHAPE);
        I表示图像特征, 为torch Tensor, 形状为(B, *IMAGE_SHAPE)。

实例输出形式：
        动作概率分布, 为torch Tensor, 形状为(B, *OUTPUT_SHAPE);

"""

from numpy import prod

import torch
from torch import nn

from agent_policy_gradient.conf.conf import Config


class PolicyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.vector_process = nn.Sequential(
            nn.Linear(Config.FEATURE_VECTOR_SHAPE[0], 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )

        cnn_layer1 = [
            nn.Conv2d(4, 16, kernel_size=3, stride=2, padding=2),
            nn.ReLU(inplace=True),
        ]
        cnn_layer2 = [
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=2),
            nn.ReLU(inplace=True),
        ]
        cnn_layer3 = [
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        ]
        cnn_layer4 = [
            nn.Conv2d(64, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        ]
        cnn_flatten = [nn.Flatten(), nn.Linear(512, 128), nn.ReLU(inplace=True)]

        self.cnn_layer = cnn_layer1 + cnn_layer2 + cnn_layer3 + cnn_layer4 + cnn_flatten
        self.image_process = nn.Sequential(*self.cnn_layer)
        self.use_cnn = Config.USE_CNN

        if self.use_cnn:
            stem_input_shape = prod(self.vector_process(torch.zeros(1, *Config.FEATURE_VECTOR_SHAPE)).shape[1:]) + prod(
                nn.Sequential(*self.image_process)(torch.zeros(1, *Config.FEATURE_IMAGE_SHAPE)).shape[1:]
            )
        else:
            stem_input_shape = prod(self.vector_process(torch.zeros(1, *Config.FEATURE_VECTOR_SHAPE)).shape[1:])

        self.stem = nn.Sequential(
            nn.ReLU(),
            nn.Linear(stem_input_shape, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.act_head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, prod(Config.ACTION_SHAPE)))

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, vec, img):
        vec_out = self.vector_process(vec)
        if self.use_cnn:
            img_out = self.image_process(img)
            out = self.stem(torch.cat((vec_out, img_out), dim=1))
        else:
            out = self.stem(vec_out)
        act = self.act_head(out)
        return act
