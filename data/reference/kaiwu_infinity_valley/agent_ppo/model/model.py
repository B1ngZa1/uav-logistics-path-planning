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
import torch.nn as nn
from torch.nn import ModuleDict
import torch.nn.functional as F

import numpy as np
from math import ceil, floor
from collections import OrderedDict
from typing import Dict, List, Tuple
from agent_ppo.conf.conf import Config


class Model(nn.Module):
    def __init__(self, device=None):
        super(Model, self).__init__()
        self.model_name = "infinity_valley"

        self.device = device
        self.label_size_list = Config.LABEL_SIZE_LIST

        self.unit_size = 128
        # build network
        # 构建网络
        # main module
        # 网络主模块
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

        all_dims = [stem_input_shape, 512, 256, 128]
        self.main_mlp = MLP(all_dims, "main_mlp")

        # output label
        # 输出标签
        self.label_mlp = ModuleDict(
            {
                "label{0}_mlp".format(label_index): MLP(
                    [self.unit_size, self.label_size_list[label_index]],
                    "label{0}_mlp".format(label_index),
                )
                for label_index in range(len(self.label_size_list))
            }
        )

        # output value
        # 输出value
        self.value_mlp = MLP([self.unit_size, 64, 1], "value_mlp")

    def forward(self, s, inference=False):

        feature_split_list = s.split(
            [
                np.prod(Config.FEATURE_VECTOR_SHAPE),
                np.prod(Config.FEATURE_IMAGE_SHAPE),
            ],
            dim=1,
        )

        vec = feature_split_list[0].to(torch.float32)
        img = feature_split_list[1].to(torch.float32).view(-1, *Config.FEATURE_IMAGE_SHAPE)

        vec_out = self.vector_process(vec)
        if self.use_cnn:
            img_out = self.image_process(img)
            main_nn = self.main_mlp(torch.cat((vec_out, img_out), dim=1))
        else:
            main_nn = self.main_mlp(vec_out)

        result_list = []
        # output label
        # 输出标签
        for label_index, label_dim in enumerate(self.label_size_list[:]):
            label_mlp_out = self.label_mlp["label{0}_mlp".format(label_index)](main_nn)
            result_list.append(label_mlp_out)

        # output value
        # 输出value
        value_result = self.value_mlp(main_nn)
        result_list.append(value_result)

        # prepare for infer graph
        # 为推理图做准备
        logits = torch.flatten(torch.cat(result_list[:-1], 1), start_dim=1)
        value = result_list[-1]
        if inference:
            return [logits, value]
        else:
            return result_list

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()


#######################
## Utility functions ##
#######################
def make_fc_layer(in_features: int, out_features: int, use_bias=True):
    """
    Wrapper function to create and initialize a linear layer
    Args:
        in_features (int): ``in_features``
        out_features (int): ``out_features``
    Returns:
        nn.Linear: the initialized linear layer
    """
    """
    用于创建和初始化线性层的包装函数
    参数：
        in_features (int): ``in_features``
        out_features (int): ``out_features``
    返回值：
        nn.Linear：已初始化的线性层
    """
    fc_layer = nn.Linear(in_features, out_features, bias=use_bias)

    # initialize weight and bias
    # 初始化权重和偏置
    nn.init.orthogonal(fc_layer.weight)
    if use_bias:
        nn.init.zeros_(fc_layer.bias)

    return fc_layer


############################
## Building-block classes ##
############################
class MLP(nn.Module):
    """
    A simple multi-layer perceptron
    """

    """
    一个简单的多层感知器
    """

    def __init__(
        self,
        fc_feat_dim_list: List[int],
        name: str,
        non_linearity: nn.Module = nn.ReLU,
        non_linearity_last: bool = False,
    ):
        super(MLP, self).__init__()
        self.fc_layers = nn.Sequential()
        for i in range(len(fc_feat_dim_list) - 1):
            fc_layer = make_fc_layer(fc_feat_dim_list[i], fc_feat_dim_list[i + 1])
            self.fc_layers.add_module("{0}_fc{1}".format(name, i + 1), fc_layer)
            if i + 1 < len(fc_feat_dim_list) - 1 or non_linearity_last:
                self.fc_layers.add_module("{0}_non_linear{1}".format(name, i + 1), non_linearity())

    def forward(self, data):
        return self.fc_layers(data)
