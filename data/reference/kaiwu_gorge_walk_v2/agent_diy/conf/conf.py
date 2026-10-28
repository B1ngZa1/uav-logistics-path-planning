#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


class Config:
    # Observation/action dimensions.
    OBSERVATION_SHAPE = 55
    ACTION_SIZE = 4

    # DQN hyperparameters.
    GAMMA = 0.98
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 128
    REPLAY_BUFFER_SIZE = 20000
    WARMUP_SIZE = 1000
    TARGET_UPDATE_FREQ = 200
    GRAD_CLIP_NORM = 10.0

    # Exploration schedule.
    EPSILON_START = 1.0
    EPSILON_END = 0.05
    EPSILON_DECAY_STEPS = 50000

    # Standalone workflow settings.
    EPISODES = 10000
    SAVE_MODEL_INTERVAL = 300
