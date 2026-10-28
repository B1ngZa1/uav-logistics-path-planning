#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import itertools
import math
import numpy as np
from agent_policy_gradient.conf.conf import Config
from collections import deque, defaultdict


def norm(pos):
    norm_pos = (
        (pos["x"] + 64250) / 128000,
        (pos["z"] + 64250) / 128000,
    )
    return norm_pos


def get_direction(pos_1, pos_2):
    """
    Calculate the bearing between two points
    输入两个点计算方位

    Args:
        x1 (int): x1
        z1 (int): z1
        x2 (int): x2
        z2 (int): z2

    Returns:
        int: Direction encoding, refer to the protocol
        int: 方向编码，参考协议
    """
    x1, z1 = pos_1
    x2, z2 = pos_2

    x = x2 - x1
    z = z2 - z1

    # Calculate the angle
    # 计算角度
    theta = math.atan2(z, x)
    if theta < 0:
        # range of atan2 is -pi to pi
        theta = 2 * math.pi + theta

    # Encode the direction based on the angle; refer to the protocol for the encoding method
    # 根据角度对方向进行编码，编码方式见协议
    r_direction = round(theta / (math.pi / 4)) + 1
    if r_direction == 9:
        r_direction = 1
    return r_direction


def get_relative_grid_pos(hero_grid_pos, target_grid_pos, n=50):
    # Calculate the relative position of target_grid_pos on a local grid centered at hero_grid_pos.
    # 计算 target_grid_pos 在以 hero_grid_pos 为中心的局部网格上的相对坐标。
    hero_x, hero_y = hero_grid_pos
    target_x, target_y = target_grid_pos

    # Calculate the boundaries of the local grid
    # 计算局部网格的边界
    min_x = hero_x - n
    max_x = hero_x + n
    min_y = hero_y - n
    max_y = hero_y + n

    # Check if the target point is within the local grid
    # 检查目标点是否在局部网格内
    if min_x <= target_x <= max_x and min_y <= target_y <= max_y:
        # Calculate the relative coordinates
        # 计算相对坐标
        relative_x = target_x - min_x
        relative_y = target_y - min_y
        return (relative_x, relative_y)
    else:
        # The target point is not within the local grid
        # 目标点不在局部网格内
        return None


def bump(a1, b1, a2, b2):
    """
    This function is used to determine if a collision with a wall has occurred.
        - There is no bump on the first frame.
        - Starting from the second frame, if the movement distance is less than 500, it is considered a collision with a wall

    该函数用于判断是否撞墙
        - 第一帧不会bump
        - 第二帧开始, 如果移动距离小于500则视为撞墙
    """
    if a2 == -1 and b2 == -1:
        return False
    if a1 == -1 and b1 == -1:
        return False

    dist = ((a1 - a2) ** 2 + (b1 - b2) ** 2) ** (0.5)

    return dist <= 500


def ln_distance(pos_1, pos_2, n):
    """

    Calculate the l_n distance
    计算 l_n距离

    Args:
        a1 (float): x1
        b1 (float): z1
        a2 (float): x2
        b2 (float): z2
        n (int): n

    Returns:
        _type_: ln distance
    """
    a1, b1 = pos_1
    a2, b2 = pos_2

    a = abs(a1 - a2)
    b = abs(b1 - b2)

    return (a**n + b**n) ** (1 / n)


def convert_pos_to_grid_pos(x, z):
    """

    Convert the position 'pos' into grid-based coordinates
    将pos转换为珊格化后坐标

    Args:
        x (float): x
        z (float): z

    Returns:
        _type_: tuple
    """
    x = (x + 64250) // 500
    z = (z + 64250) // 500

    # This step is necessary in order to be aligned with the order of json files
    # 这一步是为了与json文件的顺序保持一致
    x, z = z, x

    return x, z


def bfs_from_center_to_goal(map, goal):
    """
    Perform BFS from the center of the map to the goal.
    从地图中心出发到达目标点的BFS。

    Args:
        map (np.array): 地图, 0表示不可通过, 1表示可通过。The map, 0 for non-passable, 1 for passable.
        goal (tuple): 目标位置。The goal position.

    Returns:
        int: 从中心点到目标位置的距离。The distance from the center to the goal position.
    """
    center_x = map.shape[0] // 2
    center_y = map.shape[1] // 2
    start = (center_x, center_y)

    out = np.full((map.shape[0], map.shape[1]), -1)
    visited = {start}
    queue = deque([(start, 0)])

    while queue:
        loc, dist = queue.popleft()
        out[loc] = dist

        if loc == goal:
            return dist

        x, y = loc
        for dx, dy in [(-1, 0), (-1, -1), (1, 0), (1, -1), (0, -1), (-1, 1), (0, 1), (1, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < map.shape[0] and 0 <= ny < map.shape[1] and map[nx][ny] == 1 and (nx, ny) not in visited:
                queue.append(((nx, ny), dist + 1))
                visited.add((nx, ny))

    return -1


class Preprocessor:
    def __init__(self):
        # The map is rasterized into 256 * 256 grids, creating a memory map of the corresponding size
        # 地图被栅格化为256 * 256个网格, 创建对应大小的记忆地图
        self.memory_map = np.zeros((256, 256))

        self.recent_position_map = defaultdict(int)
        self.arrival_position_map = defaultdict(int)
        self.recent_position_max = 200
        self.recent_positions = deque(maxlen=self.recent_position_max)
        self.last_pos = None

    def update_position(self, hero_grid_pos):
        # If the queue is full, reduce the count of the oldest position
        # 如果队列满了，减少最旧位置的计数
        if len(self.recent_positions) == self.recent_position_max:
            oldest_pos = self.recent_positions.popleft()
            self.recent_position_map[oldest_pos] -= 1
            if self.recent_position_map[oldest_pos] == 0:
                del self.recent_position_map[oldest_pos]

        # Add the new position to the queue and dictionary
        # 将新位置添加到队列和字典中
        self.recent_positions.append(hero_grid_pos)
        self.recent_position_map[hero_grid_pos] += 1
        self.arrival_position_map[hero_grid_pos] += 1

    def get_legal_actions(self, grid):
        center_x = grid.shape[0] // 2
        center_z = grid.shape[1] // 2
        hero_grid_pos = (center_x, center_z)

        # The displacement caused by each action on the grid
        # 创建自定义形状的legal_action
        grid_actions = [
            (0, 1),
            (1, 1),
            (1, 0),
            (1, -1),
            (0, -1),
            (-1, -1),
            (-1, 0),
            (-1, 1),
        ]
        actions_len = len(grid_actions)
        legal_actions = [1] * actions_len
        legal_actions_feature = []

        for action_index in range(actions_len):
            action = grid_actions[action_index]
            new_x = hero_grid_pos[0] + action[0]
            new_z = hero_grid_pos[1] + action[1]

            # Create legal_action_feature based on the terrain near the agent
            # 根据智能体附近地形创建legal_action_feature
            legal_action_feature = [1] * 4
            for i in range(3):
                legal_point = (new_x + i * action[0], new_z + i * action[1])
                if grid[legal_point[0]][legal_point[1]] == 0:
                    legal_action_feature[i] = 0
            legal_action_feature[3] = min(self.recent_position_map.get((new_x, new_z), 0) / 25, 1)
            legal_actions_feature.extend(legal_action_feature)

            if 0 <= new_x < len(grid) and 0 <= new_z < len(grid[0]):
                # If the new target point has obstacles or monsters, it is considered an illegal action
                # 如果新目标点有障碍或有怪物, 视为非法动作
                if grid[new_x][new_z] == 0:
                    legal_actions[action_index] = 0
            else:
                legal_actions[action_index] = 0

        # Avoid situations where all actions are illegal
        # 避免出现所有动作都非法的情况
        if not any(legal_actions):
            legal_actions = [1] * 8

        return legal_actions, legal_actions_feature

    def process(self, obs):
        """
        Extract feature information

        Args:
            obs: Local observation information from the environment
        Returns:
            hero_info: Hero information
            treasure_infos: Treasure information
            monster_infos: Monster information
            obstacle_map: Surrounding obstacle map information
            memory_map: Surrounding memory map information
            treasure_map: Surrounding treasure map information
            monster_map: Surrounding monster map information
            legal_actions: Legal actions
            legal_actions_feature: Legal action features
            recent_position_map: Recent position information

        提取特征信息

        Args:
            obs: 环境局部观测信息
        Returns:
            hero_info: 英雄信息
            treasure_infos: 宝箱信息
            monster_infos: 怪物信息
            obstacle_map: 周边障碍物地图信息
            memory_map: 周边记忆地图信息
            treasure_map: 周边宝箱地图信息
            monster_map: 周边怪物地图信息
            legal_actions: 合法动作
            legal_actions_feature: 合法动作特征
            recent_position_map: 最近通过位置信息
        """

        # Get the map in the local view
        # 获取局部视野下的地图
        grid = []
        for map_info in obs["map_info"]:
            grid.append(map_info)
        grid = np.array(grid)

        # Hero information
        # 英雄信息
        hero_info = {}
        hero_pos = obs["frame_state"]["heroes"][0]["pos"]
        hero_grid_pos = convert_pos_to_grid_pos(hero_pos["x"], hero_pos["z"])
        hero_grid_pos_x, hero_grid_pos_z = hero_grid_pos

        hero_info["pos"] = hero_pos
        hero_info["grid_pos"] = hero_grid_pos
        hero_info["norm_pos"] = norm(hero_pos)

        self.update_position(hero_grid_pos)

        # Treasure information
        # 宝箱信息
        treasures = obs["frame_state"]["organs"]
        treasure_infos = {}
        treasure_grids = set()
        for treasure in treasures:
            single_treasure = {}
            treasure_grid_pos = convert_pos_to_grid_pos(treasure["pos"]["x"], treasure["pos"]["z"])
            grid_distance = bfs_from_center_to_goal(
                grid,
                get_relative_grid_pos(hero_grid_pos, treasure_grid_pos),
            )
            single_treasure["pos"] = treasure["pos"]
            single_treasure["status"] = treasure["status"]
            single_treasure["grid_pos"] = treasure_grid_pos
            single_treasure["norm_pos"] = norm(treasure["pos"])
            single_treasure["direction"] = get_direction(hero_grid_pos, treasure_grid_pos)
            single_treasure["l2_distance"] = ln_distance(hero_grid_pos, treasure_grid_pos, 2)
            single_treasure["grid_distance"] = grid_distance
            treasure_infos[treasure["config_id"]] = single_treasure
            treasure_grids.add(treasure_grid_pos)

        # monster information
        # 怪物信息
        monsters = obs["frame_state"]["monsters"]
        monster_infos = {}
        monsters_grids = set()
        for monster in monsters:
            single_monster = {}
            monster_grid_pos = convert_pos_to_grid_pos(monster["pos"]["x"], monster["pos"]["z"])
            grid_distance = bfs_from_center_to_goal(
                grid,
                get_relative_grid_pos(hero_grid_pos, monster_grid_pos),
            )
            single_monster["pos"] = monster["pos"]
            single_monster["grid_pos"] = monster_grid_pos
            single_monster["norm_pos"] = norm(monster["pos"])
            single_monster["direction"] = get_direction(hero_grid_pos, monster_grid_pos)
            single_monster["l2_distance"] = ln_distance(hero_grid_pos, monster_grid_pos, 2)
            single_monster["grid_distance"] = grid_distance
            monster_infos[monster["monster_id"]] = single_monster
            monsters_grids.add(monster_grid_pos)

            inaccessible_radius = 3
            monster_relative_grid_pos_x, monster_relative_grid_pos_z = get_relative_grid_pos(
                hero_grid_pos, monster_grid_pos
            )
            for i in range(
                monster_relative_grid_pos_x - inaccessible_radius, monster_relative_grid_pos_x + inaccessible_radius + 1
            ):
                for j in range(
                    monster_relative_grid_pos_z - inaccessible_radius,
                    monster_relative_grid_pos_z + inaccessible_radius + 1,
                ):
                    if 0 <= i < len(grid) and 0 <= j < len(grid[0]):  # Check if indices are within grid boundaries
                        if (
                            abs(i - monster_relative_grid_pos_x) + abs(j - monster_relative_grid_pos_z)
                            <= inaccessible_radius
                        ):
                            grid[i][j] = 0

        # legal actions information
        # 合法动作信息
        legal_actions, legal_actions_feature = self.get_legal_actions(grid)

        # image feature related
        # 图像特征相关
        view = Config.VIEW_SIZE
        obstacle_map = np.zeros((view * 2 + 1, view * 2 + 1))
        local_memory_map = np.zeros((view * 2 + 1, view * 2 + 1))
        treasure_map = np.zeros((view * 2 + 1, view * 2 + 1))
        monster_map = np.zeros((view * 2 + 1, view * 2 + 1))

        obstacle_big_map = np.zeros((view + 1, view + 1))
        local_memory_big_map = np.zeros((view + 1, view + 1))
        treasure_big_map = np.zeros((view + 1, view + 1))
        monster_big_map = np.zeros((view + 1, view + 1))

        # If not using a CNN network, skip generating image features
        # 如果不使用CNN网络, 跳过生成图像特征
        if Config.USE_CNN:

            # Dynamically update the local_memory_map, end_map, treasure_map,
            # and obstacle_map features, padding the map boundaries with 0
            # 动态更新local_memory_map, end_map, treasure_map, obstacle_map 四个特征，地图边界处用0进行padding
            for i, j in itertools.product(range(-view, view + 1), range(-view, view + 1)):
                local_i = view + i
                local_j = view + j

                if 0 <= local_i < len(grid) and 0 <= local_j < len(grid[0]):
                    obstacle_map[local_i][local_j] = grid[local_i][local_j]
                    if (
                        0 <= hero_grid_pos_x + i < self.memory_map.shape[0]
                        and 0 <= hero_grid_pos_z + j < self.memory_map.shape[1]
                    ):
                        local_memory_map[local_i][local_j] = self.memory_map[hero_grid_pos_x + i, hero_grid_pos_z + j]
                    if (hero_grid_pos_x + i, hero_grid_pos_z + j) in treasure_grids:
                        treasure_map[local_i][local_j] = 1
                    if (hero_grid_pos_x + i, hero_grid_pos_z + j) in monsters_grids:
                        monster_map[local_i][local_j] = 1

            # The memory map needs to be updated after the local_memory_map is initialized
            # Memory map 需要在local_memory_map初始化后更新
            self.memory_map[hero_grid_pos_x, hero_grid_pos_z] = min(
                1, 0.2 + self.memory_map[hero_grid_pos_x, hero_grid_pos_z]
            )

            # Iterate through the original matrix, aggregating each 2x2 block into the new matrix
            # 遍历原始矩阵，将每个 2x2 的块聚合到新的矩阵中
            for original_matrix, reduced_matrix in zip(
                (obstacle_map, local_memory_map, treasure_map, monster_map),
                (obstacle_big_map, local_memory_big_map, treasure_big_map, monster_big_map),
            ):
                original_matrix[:] = np.flipud(original_matrix)

                for i in range(view + 1):
                    for j in range(view + 1):
                        if i < view and j < view:
                            # Calculate the starting position of the 2x2 block
                            # 计算 2x2 块的起始位置
                            block = original_matrix[i * 2 : (i * 2) + 2, j * 2 : (j * 2) + 2]
                            # Take the maximum value of the block
                            # 取块的平均值
                            reduced_matrix[i, j] = np.max(block)
                        elif i == view and j < view:
                            # Handle the last row, take a 1x2 block
                            # 处理最后一行，取 1x2 块
                            block = original_matrix[i * 2, j * 2 : (j * 2) + 2]
                            reduced_matrix[i, j] = np.max(block)
                        elif j == view and i < view:
                            # Handle the last column, take a 2x1 block
                            # 处理最后一列，取 2x1 块
                            block = original_matrix[i * 2 : (i * 2) + 2, j * 2]
                            reduced_matrix[i, j] = np.max(block)
                        elif i == view / 2 and j == view / 2:
                            # Handle the middle cell, directly copy
                            # 处理中间的格子，直接复制
                            reduced_matrix[i, j] = original_matrix[i * 2, j * 2]

        return (
            hero_info,
            treasure_infos,
            monster_infos,
            obstacle_big_map.flatten().astype(int).tolist(),
            local_memory_big_map.flatten().tolist(),
            treasure_big_map.flatten().astype(int).tolist(),
            monster_big_map.flatten().astype(int).tolist(),
            legal_actions,
            legal_actions_feature,
            self.recent_position_map.copy(),
        )
