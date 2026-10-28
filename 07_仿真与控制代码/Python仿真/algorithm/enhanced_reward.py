"""
增强版奖励函数（针对完成率优化）
项目：基于深度强化学习的多无人机协同物流配送路径规划

核心改进：
1. 更强的密集进度奖励
2. 方向引导奖励
3. 接近目标渐进奖励
4. 降低惩罚幅度
"""

import numpy as np


class EnhancedRewardFunction:
    """增强版奖励函数"""

    def __init__(self, config=None):
        config = config or {}

        # 进度奖励（大幅增加）
        self.k_progress = config.get('k_progress', 50.0)  # 从10增加到50

        # 方向引导奖励
        self.k_direction = config.get('k_direction', 20.0)

        # 接近目标渐进奖励
        self.k_approach = config.get('k_approach', 100.0)
        self.approach_radius = config.get('approach_radius', 50.0)  # 50米内开始获得奖励

        # 到达奖励
        self.R_arrival = config.get('R_arrival', 500.0)  # 从200增加到500
        self.arrival_threshold = config.get('arrival_threshold', 5.0)

        # 配送完成奖励
        self.R_delivery = config.get('R_delivery', 1000.0)  # 从200增加到1000

        # 碰撞惩罚（降低）
        self.R_collision = config.get('R_collision', -30.0)  # 从-50降低到-30

        # 时间惩罚（仅当无进展时）
        self.k_time = config.get('k_time', 0.1)  # 从0.5降低到0.1

        # 避障惩罚（降低）
        self.k_obstacle = config.get('k_obstacle', 10.0)

        # 安全距离
        self.d_safe = config.get('d_safe', 10.0)

        # 历史记录
        self.prev_distances = {}

    def compute(self, uav_id, uav_pos, uav_vel, target_pos,
                obstacles, delivered, collisions, step=0):
        """
        计算增强奖励
        """
        reward_info = {}
        total_reward = 0.0

        uav_pos = np.array(uav_pos)
        target_pos = np.array(target_pos)
        uav_vel = np.array(uav_vel)

        # 当前距离
        current_distance = np.linalg.norm(uav_pos - target_pos)

        # ==================== 1. 密集进度奖励 ====================
        prev_distance = self.prev_distances.get(uav_id, current_distance)
        progress = (prev_distance - current_distance) / max(prev_distance, 1e-6)
        r_progress = self.k_progress * progress
        reward_info['progress'] = r_progress
        total_reward += r_progress
        self.prev_distances[uav_id] = current_distance

        # ==================== 2. 方向引导奖励 ====================
        goal_direction = target_pos - uav_pos
        goal_dist = np.linalg.norm(goal_direction)

        if goal_dist > 1e-6:
            goal_dir_norm = goal_direction / goal_dist
            vel_norm = np.linalg.norm(uav_vel)

            if vel_norm > 0.1:  # 只有在移动时才计算方向
                vel_dir = uav_vel / vel_norm
                cos_sim = np.dot(vel_dir, goal_dir_norm)
                r_direction = self.k_direction * max(cos_sim, 0)  # 只奖励正向
                reward_info['direction'] = r_direction
                total_reward += r_direction

        # ==================== 3. 接近目标渐进奖励 ====================
        if current_distance < self.approach_radius:
            approach_factor = 1.0 - (current_distance / self.approach_radius)
            r_approach = self.k_approach * approach_factor ** 2  # 平方增长
            reward_info['approach'] = r_approach
            total_reward += r_approach

        # ==================== 4. 到达奖励 ====================
        if current_distance < self.arrival_threshold:
            r_arrival = self.R_arrival
            reward_info['arrival'] = r_arrival
            total_reward += r_arrival

        # ==================== 5. 配送完成奖励 ====================
        if delivered:
            r_delivery = self.R_delivery
            reward_info['delivery'] = r_delivery
            total_reward += r_delivery

        # ==================== 6. 碰撞惩罚 ====================
        if collisions.get('obstacle_collision', False):
            reward_info['collision'] = self.R_collision
            total_reward += self.R_collision

        if collisions.get('uav_collision', False):
            reward_info['uav_collision'] = self.R_collision * 0.5
            total_reward += self.R_collision * 0.5

        # ==================== 7. 避障惩罚（非线性）====================
        min_obs_dist = float('inf')
        for obs in obstacles:
            obs_center = np.array(obs['center'])
            obs_radius = obs.get('radius', 5.0)
            dist = np.linalg.norm(uav_pos - obs_center) - obs_radius
            min_obs_dist = min(min_obs_dist, dist)

        if min_obs_dist < self.d_safe and min_obs_dist > 0:
            r_obstacle = -self.k_obstacle * (self.d_safe / min_obs_dist - 1)
            reward_info['obstacle'] = r_obstacle
            total_reward += r_obstacle

        # ==================== 8. 时间惩罚（仅当无进展时）====================
        if progress <= 0 and not delivered:
            r_time = -self.k_time
            reward_info['time'] = r_time
            total_reward += r_time

        # ==================== 9. 速度奖励（鼓励移动）====================
        speed = np.linalg.norm(uav_vel)
        if speed > 0.1 and current_distance > self.arrival_threshold:
            r_speed = 0.5  # 小额移动奖励
            reward_info['speed'] = r_speed
            total_reward += r_speed

        return total_reward, reward_info

    def reset(self):
        """重置历史"""
        self.prev_distances = {}


# 测试
if __name__ == '__main__':
    reward_func = EnhancedRewardFunction()

    # 模拟场景
    uav_pos = [10, 10, 15]
    uav_vel = [5, 3, 0]
    target_pos = [50, 50, 10]
    obstacles = [{'center': [30, 30, 15], 'radius': 5.0}]

    total, info = reward_func.compute(
        uav_id=0,
        uav_pos=uav_pos,
        uav_vel=uav_vel,
        target_pos=target_pos,
        obstacles=obstacles,
        delivered=False,
        collisions={'obstacle_collision': False, 'uav_collision': False},
        step=10
    )

    print(f"总奖励: {total:.2f}")
    for k, v in info.items():
        print(f"  {k}: {v:.2f}")
