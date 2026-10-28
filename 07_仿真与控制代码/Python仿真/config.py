"""
配置文件
项目：基于深度强化学习的多无人机协同物流配送路径规划
"""

# 环境配置
# 统一依据：参数一致性规范 v1.0 (2026-07-01)
ENV_CONFIG = {
    "space_size": (100, 100, 50),   # 空域大小 (x, y, z)
    "num_uavs": 3,                  # 无人机数量
    "num_targets": 5,               # 配送目标数量
    "num_obstacles": 10,            # 障碍物数量
    "dt": 0.1,                      # 时间步长 (s) — 统一值：代码0.5/文档0.1 → 0.1
    "max_steps": 1000,              # 最大步数 — 统一值：代码500/文档1000 → 1000
    "v_max": 15.0,                  # 最大速度 (m/s)
    "a_max": 5.0,                   # 最大加速度 (m/s²) — 统一值：代码3.0/文档5.0 → 5.0
    "z_min": 10.0,                  # 最低飞行高度 (m)
    "z_max": 50.0,                  # 最高飞行高度 (m) — 统一值：代码40/文档120 → 50
    "d_safe_uav": 10.0,             # UAV间安全距离 (m) — 区分场景：UAV间10m
    "d_safe_obstacle": 5.0,         # 障碍物安全距离 (m) — 区分场景：障碍物5m
}

# 无人机配置
UAV_CONFIG = {
    "max_speed": 10.0,          # 最大速度 (m/s)
    "max_acceleration": 5.0,    # 最大加速度 (m/s²)
    "battery_capacity": 10000,  # 电池容量 (mAh)
    "mass": 1.5,                # 质量 (kg)
    "min_safe_distance": 2.0,   # 最小安全距离 (m)
}

# PPO算法配置
# 统一依据：参数一致性规范 v1.0 (2026-07-01)
PPO_CONFIG = {
    "learning_rate": 3e-4,      # 学习率
    "min_learning_rate": 1e-5,  # 最小学习率 (余弦退火下限)
    "gamma": 0.99,              # 折扣因子
    "gae_lambda": 0.95,         # GAE参数
    "clip_epsilon": 0.2,        # 裁剪参数
    "entropy_coef": 0.01,       # 熵系数
    "value_coef": 0.5,          # 价值系数
    "max_grad_norm": 0.5,       # 梯度裁剪
    "batch_size": 256,          # 批量大小
    "hidden_dim": 256,          # 隐藏层维度
    "update_epochs": 4,         # 每epoch更新次数
    "buffer_size": 10000,       # 经验池大小
}

# 网络结构配置
# 统一依据：参数一致性规范 v1.0 (2026-07-01)
NETWORK_CONFIG = {
    "actor_hidden_dims": [256, 128, 64],    # Actor隐藏层维度
    "actor_activation": "ReLU",             # Actor激活函数
    "critic_hidden_dims": [256, 128, 64],   # Critic隐藏层维度
    "critic_activation": "Tanh",            # Critic激活函数 — 统一值：必须Tanh，禁止ReLU
    "actor_init_gain": 0.01,                # Actor正交初始化增益
    "critic_init_gain": 1.0,                # Critic正交初始化增益
    "log_std_init": -0.5,                   # 对数标准差初始值
}

# 奖励函数配置
# 统一依据：参数一致性规范 v1.0 (2026-07-01)
REWARD_CONFIG = {
    "distance_weight": -0.1,        # 距离权重 (/m)
    "time_penalty": -1.0,           # 时间惩罚 (/step) — 统一值：代码-0.5/文档-2.0 → -1.0
    "collision_weight": -10.0,      # 碰撞惩罚权重 (/次)
    "energy_weight": -0.05,         # 能耗权重 (/J)
    "delivery_reward_single": 100,  # 单个配送完成奖励 — 统一值：代码+200/文档+50 → +100
    "delivery_reward_all": 500,     # 全部配送完成奖励 — 统一值：代码+500/文档无 → +500
    "cooperation_weight": 1.0,      # 协同奖励权重 (/次)
    "safety_weight": -5.0,          # 安全距离权重 (/次)
    "boundary_weight": -20.0,       # 边界惩罚权重 (/次)
}

# 训练配置
TRAIN_CONFIG = {
    "num_episodes": 10000,      # 训练回合数
    "eval_interval": 100,       # 评估间隔
    "save_interval": 500,       # 保存间隔
    "log_interval": 10,         # 日志间隔
    "seed": 42,                 # 随机种子
}

# 武汉脱敏参考场景环境配置
# 读取 outputs/环境示意图_3D预览 中的离线清单和碰撞数据。
WUHAN_ENV_CONFIG = {
    "zone_id": "A",                    # A/B/C/D 四个脱敏参考分区
    "num_drones": 1,                   # 单机闭环完成后再扩展共享策略多机
    "physics_dt_s": 0.1,               # 物理积分周期
    "control_dt_s": 0.5,               # 策略动作周期
    "max_episode_time_s": 900.0,       # 单分区课程场景上限
    "min_altitude_m": 10.0,
    "max_altitude_m": 200.0,
    "max_speed_xy_mps": 15.0,
    "max_speed_z_mps": 5.0,
    "max_accel_xy_mps2": 5.0,
    "max_accel_z_mps2": 3.0,
    "max_yaw_rate_rps": 1.2,
    "drone_radius_m": 1.0,
    "safety_clearance_m": 8.0,
    "battery_capacity_wh": 250.0,
    "reserve_soc": 0.15,
    "return_time_margin_s": 30.0,    # 返航时间安全裕量
    "return_route_factor": 1.25,     # 直线距离到可飞路径的保守系数
    "return_energy_wh_per_m": 0.015, # 返航能耗保守估计
    "score_per_delivery": 1000.0,    # 目标数严格优先于时间/能耗
    "score_safe_return": 150.0,      # 开悟式计分：安全返航
    "score_efficiency_per_s": 0.2,   # 安全返航后的剩余时间分
    "map_size": 32,
    "map_extent_m": 256.0,
    "lidar_rays": 36,
    "lidar_range_m": 180.0,
    "seed": 42,
}

WUHAN_SCENARIO_CONFIGS = {
    "WA": {"zone_id": "A", "num_drones": 1},
    "WB": {"zone_id": "B", "num_drones": 1},
    "WC": {"zone_id": "C", "num_drones": 1},
    "WD": {"zone_id": "D", "num_drones": 1},
    "WA_MULTI": {"zone_id": "A", "num_drones": 3},
    "WB_MULTI": {"zone_id": "B", "num_drones": 3},
}
