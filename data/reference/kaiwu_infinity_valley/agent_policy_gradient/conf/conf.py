#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import numpy as np


# Configuration with dimension and algorithm parameter settings
# 包含维度和算法参数设置的配置
class Config:

    # Whether to use CNN networks
    # 是否使用CNN网络
    USE_CNN = False
    VIEW_SIZE = 50 if USE_CNN else 0

    FEATURE_VECTOR_SHAPE = (153,)
    FEATURE_IMAGE_SHAPE = (4, VIEW_SIZE + 1, VIEW_SIZE + 1)

    ACTION_SHAPE = (8,)
    VALUE_SHAPE = (1,)

    # Discount factor GAMMA in RL
    # RL中的回报折扣GAMMA
    GAMMA = 0.95

    # Initial learning rate
    # 初始的学习率
    START_LR = 1e-4
    TARGET_LR = 5e-5
    TARGET_STEP = 2000

    VALUE_LOSS_COEFF = 0.5
    ENTROPY_LOSS_COEFF = 0.025
