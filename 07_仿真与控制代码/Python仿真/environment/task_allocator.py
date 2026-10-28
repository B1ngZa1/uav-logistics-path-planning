"""
任务分配模块
项目：基于深度强化学习的多无人机协同物流配送路径规划
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


class TaskAllocator:
    """任务分配器"""

    def __init__(self, config):
        self.alpha = config.get('alpha', 0.5)  # 时间权重
        self.beta = config.get('beta', 0.3)   # 能耗权重
        self.gamma = config.get('gamma', 0.2)  # 负载均衡权重

    def greedy_allocate(self, uav_positions, order_positions, uav_capacities, order_weights):
        """贪心分配"""
        N = len(uav_positions)
        M = len(order_positions)
        allocation = np.zeros((N, M), dtype=int)
        remaining = uav_capacities.copy()

        for j in range(M):
            distances = np.linalg.norm(uav_positions - order_positions[j], axis=1)
            distances[remaining < order_weights[j]] = np.inf
            best = np.argmin(distances)
            allocation[best][j] = 1
            remaining[best] -= order_weights[j]

        return allocation

    def hungarian_allocate(self, uav_positions, order_positions):
        """匈牙利算法分配"""
        N = len(uav_positions)
        M = len(order_positions)

        # 构建代价矩阵
        cost = np.zeros((max(N, M), max(N, M)))
        for i in range(N):
            for j in range(M):
                cost[i][j] = np.linalg.norm(uav_positions[i] - order_positions[j])

        row_ind, col_ind = linear_sum_assignment(cost)

        allocation = np.zeros((N, M), dtype=int)
        for i, j in zip(row_ind, col_ind):
            if i < N and j < M:
                allocation[i][j] = 1

        return allocation

    def dynamic_reallocate(self, allocation, uav_socs, threshold=0.2):
        """动态重分配（电量不足时）"""
        new_alloc = allocation.copy()

        for i in range(len(uav_socs)):
            if uav_socs[i] < threshold:
                orders = np.where(new_alloc[i] == 1)[0]
                if len(orders) > 0:
                    loads = np.sum(new_alloc, axis=1)
                    target = np.argmin(loads)
                    for o in orders:
                        new_alloc[i][o] = 0
                        new_alloc[target][o] = 1

        return new_alloc
