#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


class Algorithm:
    def __init__(self, model, optimizer, scheduler, device=None, logger=None, monitor=None):
        """
        Algorithm initialization (to be implemented by user)
        算法初始化（由用户实现）
        """
        pass

    def learn(self, list_sample_data):
        """
        Learning algorithm implementation (to be implemented by user)
        学习算法实现（由用户实现）

        Args:
            list_sample_data: List of SampleData objects
            list_sample_data: SampleData对象列表
        """
        pass
