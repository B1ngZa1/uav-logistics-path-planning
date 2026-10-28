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

    NUMB_HEAD = 1

    FEATURE_VECTOR_SHAPE = (153,)
    FEATURE_IMAGE_SHAPE = (4, VIEW_SIZE + 1, VIEW_SIZE + 1)
    DIM_OF_OBSERVATION = np.prod(FEATURE_VECTOR_SHAPE) + np.prod(FEATURE_IMAGE_SHAPE)
    DIM_OF_ACTION = 8

    INIT_LEARNING_RATE_START = 0.0001
    BETA_START = 0.025
    LOG_EPSILON = 1e-6

    RMSPROP_DECAY = 0.9
    RMSPROP_MOMENTUM = 0.0
    RMSPROP_EPSILON = 0.01
    CLIP_PARAM = 0.2

    MIN_POLICY = 0.00001

    LABEL_SIZE_LIST = [
        DIM_OF_ACTION,
    ]
    LEGAL_ACTION_SIZE_LIST = LABEL_SIZE_LIST.copy()
    # means each task whether need reinforce
    # 标识要进行强化学习的头
    IS_REINFORCE_TASK_LIST = [
        True,
    ] * NUMB_HEAD

    EVAL_FREQ = 5
    GAMMA = 0.995
    LAMDA = 0.95

    USE_GRAD_CLIP = True
    GRAD_CLIP_RANGE = 0.5
    VALUE_COEF = 0.001
