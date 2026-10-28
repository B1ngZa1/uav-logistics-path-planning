"""
改进的奖励函数（基于HCA-MAPPO文献）
项目：基于深度强化学习的多无人机协同物流配送路径规划

参考文献：林振希 - HCA-MAPPO 9项复合奖励函数
"""

import numpy as np


class ImprovedRewardFunction:
    """改进的奖励函数（9项分量）"""

    def __init__(self, config=None):
        # 奖励权重配置
        self.config = config or {}

        # 到达奖励
        self.R_goal = self.config.get('R_goal', 200.0)

        # 距离逼近奖励权重
        self.k_progress = self.config.get('k_progress', 10.0)

        # 非线性斥力场参数
        self.k_repulsive = self.config.get('k_repulsive', 50.0)
        self.d_safe = self.config.get('d_safe', 10.0)

        # 碰撞惩罚
        self.R_collision = self.config.get('R_collision', -50.0)

        # cos相似度引导权重
        self.k_guide = self.config.get('k_guide', 5.0)

        # 动作平滑惩罚权重
        self.k_smooth = self.config.get('k_smooth', 0.1)

        # 时间同步惩罚权重
        self.k_sync = self.config.get('k_sync', 1.0)

        # 安全间距惩罚权重
        self.k_safety = self.config.get('k_safety', 2.0)

        # 掉队惩罚权重
        self.k_straggle = self.config.get('k_straggle', 1.0)

        # 软到达奖励参数
        self.k_soft_arrival = self.config.get('k_soft_arrival', 50.0)
        self.arrival_radius = self.config.get('arrival_radius', 100.0)

        # 时间惩罚（仅当无进展时）
        self.k_time = self.config.get('k_time', 0.5)

        # 配送完成奖励
        self.R_delivery = self.config.get('R_delivery', 200.0)

        # 历史记录（用于计算进度）
        self.prev_distances = {}

    def compute(self, uav_id, uav_pos, uav_vel, target_pos,
                obstacles, other_uavs, delivered, collisions,
                prev_action=None, curr_action=None, step=0):
        """
        计算奖励

        参数:
            uav_id: 无人机ID
            uav_pos: 无人机位置 [x, y, z]
            uav_vel: 无人机速度 [vx, vy, vz]
            target_pos: 目标位置 [x, y, z]
            obstacles: 障碍物列表 [{'center': [x,y,z], 'radius': r}, ...]
            other_uavs: 其他无人机位置列表
            delivered: 是否完成配送
            collisions: 碰撞信息
            prev_action: 上一步动作
            curr_action: 当前动作
            step: 当前步数

        返回:
            total_reward: 总奖励
            reward_info: 各分量详情
        """
        reward_info = {}
        total_reward = 0.0

        # 计算距离
        uav_pos = np.array(uav_pos)
        target_pos = np.array(target_pos)
        uav_vel = np.array(uav_vel)

        current_distance = np.linalg.norm(uav_pos - target_pos)

        # ==================== R1: 到达奖励 ====================
        if delivered:
            r1 = self.R_goal
            reward_info['R1_goal'] = r1
            total_reward += r1

        # ==================== R2: 密集进度奖励 ====================
        prev_distance = self.prev_distances.get(uav_id, current_distance)
        progress = (prev_distance - current_distance) / max(prev_distance, 1e-6)
        r2 = self.k_progress * progress
        reward_info['R2_progress'] = r2
        total_reward += r2
        self.prev_distances[uav_id] = current_distance

        # ==================== R3: 非线性斥力场避障 ====================
        r3 = 0.0
        min_obstacle_dist = float('inf')
        for obs in obstacles:
            obs_center = np.array(obs['center'])
            obs_radius = obs.get('radius', 5.0)
            dist = np.linalg.norm(uav_pos - obs_center) - obs_radius
            min_obstacle_dist = min(min_obstacle_dist, dist)

            if dist < self.d_safe:
                # 非线性斥力场：距离越近惩罚越大
                repulsive = self.k_repulsive * (self.d_safe / max(dist, 0.1) - 1)
                r3 -= repulsive

        reward_info['R3_repulsive'] = r3
        total_reward += r3

        # ==================== R4: 碰撞惩罚 ====================
        r4 = 0.0
        if collisions.get('obstacle_collision', False):
            r4 += self.R_collision
        if collisions.get('uav_collision', False):
            r4 += self.R_collision
        reward_info['R4_collision'] = r4
        total_reward += r4

        # ==================== R5: cos相似度智能引导 ====================
        r5 = 0.0
        goal_direction = target_pos - uav_pos
        goal_direction_norm = np.linalg.norm(goal_direction)
        if goal_direction_norm > 1e-6:
            goal_direction = goal_direction / goal_direction_norm

            # 计算速度方向与目标方向的cos相似度
            vel_norm = np.linalg.norm(uav_vel)
            if vel_norm > 1e-6:
                vel_direction = uav_vel / vel_norm
                cos_sim = np.dot(vel_direction, goal_direction)
                r5 = self.k_guide * cos_sim

        reward_info['R5_guide'] = r5
        total_reward += r5

        # ==================== R6: 动作平滑惩罚 ====================
        r6 = 0.0
        if prev_action is not None and curr_action is not None:
            action_diff = np.array(curr_action) - np.array(prev_action)
            r6 = -self.k_smooth * np.linalg.norm(action_diff) ** 2

        reward_info['R6_smooth'] = r6
        total_reward += r6

        # ==================== R7: 时间同步惩罚 ====================
        r7 = 0.0
        # 简化：如果有多个UAV，惩罚到达时间差异
        if len(other_uavs) > 0:
            # 计算到目标的平均距离
            avg_distance = current_distance
            for other_pos in other_uavs:
                other_dist = np.linalg.norm(np.array(other_pos) - target_pos)
                avg_distance += other_dist
            avg_distance /= (len(other_uavs) + 1)

            # 惩罚偏离平均距离的程度
            distance_deviation = abs(current_distance - avg_distance)
            r7 = -self.k_sync * distance_deviation / 100.0

        reward_info['R7_sync'] = r7
        total_reward += r7

        # ==================== R8: 安全间距惩罚 ====================
        r8 = 0.0
        for other_pos in other_uavs:
            other_dist = np.linalg.norm(uav_pos - np.array(other_pos))
            if other_dist < self.d_safe:
                r8 -= self.k_safety * (self.d_safe - other_dist)

        reward_info['R8_safety'] = r8
        total_reward += r8

        # ==================== R9: 掉队惩罚 ====================
        r9 = 0.0
        if len(other_uavs) > 0:
            # 计算到其他UAV的平均距离
            avg_uav_dist = 0.0
            for other_pos in other_uavs:
                avg_uav_dist += np.linalg.norm(uav_pos - np.array(other_pos))
            avg_uav_dist /= len(other_uavs)

            # 如果偏离太远，施加惩罚
            if avg_uav_dist > 50.0:  # 50米阈值
                r9 = -self.k_straggle * (avg_uav_dist - 50.0) / 50.0

        reward_info['R9_straggle'] = r9
        total_reward += r9

        # ==================== 软到达奖励 ====================
        if current_distance < self.arrival_radius:
            soft_arrival = self.k_soft_arrival * (1 - current_distance / self.arrival_radius)
            reward_info['soft_arrival'] = soft_arrival
            total_reward += soft_arrival

        # ==================== 时间惩罚（仅当无进展时）====================
        if progress <= 0:
            r_time = -self.k_time
            reward_info['time_penalty'] = r_time
            total_reward += r_time

        # ==================== 配送完成奖励 ====================
        if delivered:
            r_delivery = self.R_delivery
            reward_info['delivery_bonus'] = r_delivery
            total_reward += r_delivery

        return total_reward, reward_info

    def reset(self):
        """重置历史记录"""
        self.prev_distances = {}


# ==================== 使用示例 ====================

if __name__ == '__main__':
    # 创建奖励函数
    reward_func = ImprovedRewardFunction()

    # 模拟数据
    uav_pos = [10, 10, 15]
    uav_vel = [5, 3, 0]
    target_pos = [50, 50, 10]
    obstacles = [
        {'center': [30, 30, 15], 'radius': 5.0},
        {'center': [40, 20, 15], 'radius': 5.0},
    ]
    other_uavs = [[15, 10, 15], [10, 15, 15]]

    # 计算奖励
    total_reward, info = reward_func.compute(
        uav_id=0,
        uav_pos=uav_pos,
        uav_vel=uav_vel,
        target_pos=target_pos,
        obstacles=obstacles,
        other_uavs=other_uavs,
        delivered=False,
        collisions={'obstacle_collision': False, 'uav_collision': False},
        prev_action=[0, 0, 0],
        curr_action=[1, 0.5, 0],
        step=10
    )

    print(f"总奖励: {total_reward:.2f}")
    print("各分量:")
    for key, value in info.items():
        print(f"  {key}: {value:.2f}")
