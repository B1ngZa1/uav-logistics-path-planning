#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from common_python.utils.common_func import create_cls
import numpy as np
from agent_diy.conf.conf import Config

# The create_cls function is used to dynamically create a class. The first parameter of the function is the type name,
# and the remaining parameters are the attributes of the class, which should have a default value of None.
# create_cls函数用于动态创建一个类，函数第一个参数为类型名称，剩余参数为类的属性，属性默认值应设为None
ObsData = create_cls(
    "ObsData",
    feature=None,
    legal_act=None,
)


ActData = create_cls(
    "ActData",
    act=None,
)


# Sample data structure (to be defined based on your algorithm)
# 样本数据结构（根据你的算法定义）
# SampleData = create_cls(
#     "SampleData",
#     obs=...,
#     action=...,
#     reward=...,
#     ...
# )


def reward_shaping(frame_no, score, terminated, truncated, remain_info, _remain_info, obs, _obs):
    """
    Reward shaping function (to be implemented by user)
    奖励塑形函数（由用户实现）
    """
    pass


def sample_process(list_game_data):
    """
    Sample processing function (to be implemented by user)
    样本处理函数（由用户实现）
    """
    pass
