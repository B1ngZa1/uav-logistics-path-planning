#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


class Config:

    # Official state space: position (64*64) * active treasure bitmask (2^10)
    STATE_SIZE = 64 * 64 * 1024
    ACTION_SIZE = 4

    # Q-Learning hyperparameters
    LEARNING_RATE = 0.15         # Lower initial rate to reduce TD-error oscillation
    GAMMA = 0.96                 # Increased from 0.9 for better long-term credit assignment
    # A 0.9993 decay still leaves epsilon near 0.88 after 190 episodes. Since
    # the fixed map already has a route planner, that spends almost the whole
    # 30-minute training window taking random actions and timing out.
    EPSILON_START = 0.35
    EPSILON_MIN = 0.02
    EPSILON_DECAY = 0.985        # Reaches the minimum after about 190 episodes
    EPISODES = 10000

    # Experience replay
    REPLAY_CAPACITY = 200000     # Replay buffer size
    BATCH_SIZE = 128             # Mini-batch size for training

    # Double Q-Learning target network sync
    TARGET_UPDATE_FREQ = 300     # Steps between Q_target ← Q sync

    # Learning rate scheduling
    LR_MIN = 0.003               # Keep a small adaptation rate late in training
    LR_DECAY_STEPS = 150000      # Avoid collapsing the rate in the first few minutes

    # Dimension of observation
    OBSERVATION_SHAPE = 55

    # Soft-update (Polyak averaging) parameters
    TARGET_SOFT_TAU = 0.005      # Soft-update coefficient for Polyak averaging
    USE_SOFT_UPDATE = False       # Whether to use Polyak averaging instead of hard sync
    USE_DNN = False               # Neural network mode toggle (True = DNN, False = Q-table)
