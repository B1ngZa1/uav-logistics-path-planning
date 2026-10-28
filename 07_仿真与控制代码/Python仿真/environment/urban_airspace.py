"""
城市空域环境（改进版）
项目：基于深度强化学习的多无人机协同物流配送路径规划

参考：无人机城市配送系统
- 建筑物（不同高度和类型）
- 河流 + 桥梁
- 高压线塔
- 树木区域
- 动态飞鸟区
- 禁飞区（政府机关等）
- GPS通信干扰区
- 风雨天气影响区
"""

import numpy as np
import gym
from gym import spaces
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from enum import Enum


class ObstacleType(Enum):
    """障碍物类型"""
    BUILDING = "building"
    RIVER = "river"
    BRIDGE = "bridge"
    POWER_LINE = "power_line"
    TREE_ZONE = "tree_zone"
    BIRD_ZONE = "bird_zone"
    NO_FLY_ZONE = "no_fly_zone"
    GPS_INTERFERENCE = "gps_interference"
    WEATHER_ZONE = "weather_zone"


@dataclass
class Obstacle:
    """障碍物数据类"""
    obs_type: ObstacleType
    position: np.ndarray  # [x, y, z] 中心位置
    size: np.ndarray      # [dx, dy, dz] 尺寸
    radius: float         # 碰撞半径
    height: float         # 高度
    color: str            # 可视化颜色
    penalty: float        # 碰撞惩罚
    is_dynamic: bool      # 是否动态
    speed: Optional[np.ndarray] = None  # 动态速度


@dataclass
class DeliveryPoint:
    """配送点数据类"""
    point_id: int
    position: np.ndarray  # [x, y, z]
    demand: float         # 需求量(g)
    time_window: Tuple[float, float]  # 时间窗口
    priority: int         # 优先级


@dataclass
class DroneState:
    """无人机状态"""
    position: np.ndarray      # [x, y, z]
    velocity: np.ndarray      # [vx, vy, vz]
    battery_soc: float        # 电池SOC(0-1)
    payload: float            # 当前载重(g)
    delivered: List[bool]     # 已配送状态
    current_target: int       # 当前目标点索引


class UrbanAirspaceEnv(gym.Env):
    """城市空域多无人机配送环境"""

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, config=None):
        super().__init__()

        # 默认配置
        self.config = config or {}

        # 空域参数
        self.space_size = self.config.get('space_size', (2000, 2000, 200))  # 2km x 2km x 200m
        self.num_drones = self.config.get('num_drones', 3)
        self.num_delivery_points = self.config.get('num_delivery_points', 12)

        # 无人机参数
        self.v_max = self.config.get('v_max', 15.0)  # 最大速度(m/s)
        self.a_max = self.config.get('a_max', 5.0)    # 最大加速度(m/s²)
        self.z_min = self.config.get('z_min', 10.0)   # 最低飞行高度(m)
        self.z_max = self.config.get('z_max', 150.0)  # 最高飞行高度(m)
        self.max_payload = self.config.get('max_payload', 3000.0)  # 最大载重(g)
        self.battery_capacity = self.config.get('battery_capacity', 10000.0)  # 电池容量(mAh)

        # 时间参数
        self.dt = self.config.get('dt', 0.1)  # 时间步长(s)
        self.max_steps = self.config.get('max_steps', 5000)  # 最大步数

        # 动作空间：每架无人机3维加速度
        self.action_space = spaces.Box(
            low=-self.a_max, high=self.a_max,
            shape=(self.num_drones, 3), dtype=np.float32
        )

        # 观测空间：每架无人机状态
        obs_dim = self._calc_obs_dim()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,), dtype=np.float32
        )

        # 初始化环境元素
        self.obstacles: List[Obstacle] = []
        self.delivery_points: List[DeliveryPoint] = []
        self.drone_states: List[DroneState] = []

        # 训练状态
        self.step_count = 0
        self.total_reward = 0.0

        # 创建环境
        self._create_environment()

    def _calc_obs_dim(self):
        """计算观测维度"""
        # 每架无人机: 位置(3) + 速度(3) + SOC(1) + 载重(1) + 目标距离(1) + 目标方向(3) + 最近障碍物距离(1) + 配送进度(1) = 14
        return self.num_drones * 14

    def _create_environment(self):
        """创建城市环境"""
        # ==================== 建筑物（10栋）====================
        buildings = [
            # (中心x, 中心y, 宽度, 深度, 高度, 颜色)
            (200, 200, 80, 60, 45, '#E53935'),    # 红色建筑
            (500, 300, 100, 80, 60, '#1E88E5'),   # 蓝色建筑
            (800, 200, 70, 50, 35, '#43A047'),    # 绿色建筑
            (300, 600, 90, 70, 55, '#FB8C00'),    # 橙色建筑
            (600, 500, 110, 90, 70, '#8E24AA'),   # 紫色建筑
            (900, 700, 85, 65, 50, '#00ACC1'),    # 青色建筑
            (400, 900, 95, 75, 65, '#F4511E'),    # 深橙色建筑
            (700, 800, 75, 55, 40, '#3949AB'),    # 深蓝色建筑
            (1000, 400, 120, 100, 80, '#7B1FA2'), # 深紫色建筑
            (150, 800, 60, 45, 30, '#00897B'),    # 深青色建筑
        ]

        for bx, by, bw, bh, bz, color in buildings:
            self.obstacles.append(Obstacle(
                obs_type=ObstacleType.BUILDING,
                position=np.array([bx, by, bz/2]),
                size=np.array([bw, bh, bz]),
                radius=max(bw, bh) / 2,
                height=bz,
                color=color,
                penalty=-100.0,
                is_dynamic=False
            ))

        # ==================== 河流 + 桥梁 ====================
        # 河流（从西到东）
        river_y = 1000
        river_width = 60
        self.obstacles.append(Obstacle(
            obs_type=ObstacleType.RIVER,
            position=np.array([1000, river_y, 0]),
            size=np.array([2000, river_width, 50]),
            radius=river_width / 2,
            height=50,
            color='#2196F3',
            penalty=-200.0,
            is_dynamic=False
        ))

        # 桥梁（可通行）
        bridges = [
            (400, river_y, 40, 80),   # 桥1
            (1200, river_y, 40, 80),  # 桥2
        ]
        for bx, by, bw, bh in bridges:
            self.obstacles.append(Obstacle(
                obs_type=ObstacleType.BRIDGE,
                position=np.array([bx, by, 20]),
                size=np.array([bw, bh, 40]),
                radius=bw / 2,
                height=40,
                color='#795548',
                penalty=0.0,  # 桥梁可通行
                is_dynamic=False
            ))

        # ==================== 高压线塔（3条）====================
        power_lines = [
            (300, 500, 300, 700, 60),
            (800, 300, 800, 800, 60),
            (1200, 400, 1200, 900, 60),
        ]
        for x1, y1, x2, y2, height in power_lines:
            # 线塔两端
            self.obstacles.append(Obstacle(
                obs_type=ObstacleType.POWER_LINE,
                position=np.array([x1, y1, height/2]),
                size=np.array([10, 10, height]),
                radius=15,
                height=height,
                color='#FFD600',
                penalty=-150.0,
                is_dynamic=False
            ))
            self.obstacles.append(Obstacle(
                obs_type=ObstacleType.POWER_LINE,
                position=np.array([x2, y2, height/2]),
                size=np.array([10, 10, height]),
                radius=15,
                height=height,
                color='#FFD600',
                penalty=-150.0,
                is_dynamic=False
            ))

        # ==================== 树木区域（4个）====================
        tree_zones = [
            (150, 400, 100, 100, 15),
            (1500, 300, 120, 120, 15),
            (250, 1200, 80, 80, 15),
            (1600, 1400, 100, 100, 15),
        ]
        for tx, ty, tw, th, tz in tree_zones:
            self.obstacles.append(Obstacle(
                obs_type=ObstacleType.TREE_ZONE,
                position=np.array([tx, ty, tz/2]),
                size=np.array([tw, th, tz]),
                radius=max(tw, th) / 2,
                height=tz,
                color='#2E7D32',
                penalty=-50.0,
                is_dynamic=False
            ))

        # ==================== 动态飞鸟区（1个）====================
        self.obstacles.append(Obstacle(
            obs_type=ObstacleType.BIRD_ZONE,
            position=np.array([1000, 600, 50]),
            size=np.array([200, 200, 100]),
            radius=100,
            height=100,
            color='#FF7043',
            penalty=-80.0,
            is_dynamic=True,
            speed=np.array([2.0, 1.5, 0.0])  # 移动速度
        ))

        # ==================== 禁飞区（2个）====================
        no_fly_zones = [
            (1800, 1800, 200, 200, '政府机关'),
            (100, 1800, 150, 150, '军事设施'),
        ]
        for nx, ny, nw, nh, name in no_fly_zones:
            self.obstacles.append(Obstacle(
                obs_type=ObstacleType.NO_FLY_ZONE,
                position=np.array([nx, ny, 100]),
                size=np.array([nw, nh, 200]),
                radius=max(nw, nh) / 2,
                height=200,
                color='#D32F2F',
                penalty=-500.0,  # 严重惩罚
                is_dynamic=False
            ))

        # ==================== GPS干扰区（1个）====================
        self.obstacles.append(Obstacle(
            obs_type=ObstacleType.GPS_INTERFERENCE,
            position=np.array([500, 1500, 80]),
            size=np.array([300, 300, 160]),
            radius=150,
            height=160,
            color='#FF9800',
            penalty=-100.0,
            is_dynamic=False
        ))

        # ==================== 风雨天气区（1个）====================
        self.obstacles.append(Obstacle(
            obs_type=ObstacleType.WEATHER_ZONE,
            position=np.array([1400, 800, 100]),
            size=np.array([400, 400, 200]),
            radius=200,
            height=200,
            color='#78909C',
            penalty=-60.0,
            is_dynamic=True,
            speed=np.array([1.0, 0.5, 0.0])  # 天气移动
        ))

        # ==================== 配送点（12个）====================
        # 配送中心（起点/终点）
        self.delivery_points.append(DeliveryPoint(
            point_id=1,
            position=np.array([100, 100, 10]),
            demand=0,
            time_window=(0, 9999),
            priority=0
        ))

        # 其他配送点（确保特殊点3、5、12远离干扰区）
        delivery_positions = [
            (300, 250, 10),    # 点2
            (600, 150, 10),    # 点3 - 远离干扰区
            (900, 300, 10),    # 点4
            (1200, 200, 10),   # 点5 - 远离干扰区
            (400, 500, 10),    # 点6
            (700, 600, 10),    # 点7
            (1000, 500, 10),   # 点8
            (1300, 400, 10),   # 点9
            (500, 800, 10),    # 点10
            (800, 900, 10),    # 点11
            (1100, 700, 10),   # 点12 - 远离干扰区
        ]

        for idx, (dx, dy, dz) in enumerate(delivery_positions):
            self.delivery_points.append(DeliveryPoint(
                point_id=idx + 2,
                position=np.array([dx, dy, dz]),
                demand=np.random.uniform(200, 500),
                time_window=(0, 30),
                priority=np.random.randint(1, 4)
            ))

    def reset(self):
        """重置环境"""
        self.step_count = 0
        self.total_reward = 0.0

        # 初始化无人机状态
        self.drone_states = []
        depot = self.delivery_points[0].position

        for i in range(self.num_drones):
            # 无人机从配送中心起飞，略有偏移
            offset = np.random.uniform(-20, 20, 3)
            offset[2] = 0  # 高度不变
            state = DroneState(
                position=depot + offset + np.array([0, 0, 50]),  # 50m高度
                velocity=np.zeros(3),
                battery_soc=1.0,
                payload=sum(dp.demand for dp in self.delivery_points[1:]) / self.num_drones,
                delivered=[False] * len(self.delivery_points),
                current_target=1  # 从第2个点开始（第1个是配送中心）
            )
            self.drone_states.append(state)

        return self._get_obs()

    def step(self, actions):
        """执行一步"""
        self.step_count += 1

        # 更新动态障碍物
        self._update_dynamic_obstacles()

        # 更新无人机状态
        total_reward = 0.0
        for i in range(self.num_drones):
            reward = self._update_drone(i, actions[i])
            total_reward += reward

        # 检查终止条件
        done = self._check_done()

        # 计算信息
        info = {
            'step': self.step_count,
            'delivered': sum(1 for ds in self.drone_states if all(ds.delivered)),
            'total_delivered': sum(sum(ds.delivered) for ds in self.drone_states),
            'collisions': self._count_collisions(),
            'battery_soc': [ds.battery_soc for ds in self.drone_states],
            'payload': [ds.payload for ds in self.drone_states],
        }

        self.total_reward += total_reward
        return self._get_obs(), total_reward, done, info

    def _update_drone(self, drone_idx, action):
        """更新单架无人机"""
        drone = self.drone_states[drone_idx]

        # 加速度裁剪
        action = np.clip(action, -self.a_max, self.a_max)

        # 速度更新
        drone.velocity += action * self.dt

        # 速度约束
        speed = np.linalg.norm(drone.velocity)
        if speed > self.v_max:
            drone.velocity = drone.velocity / speed * self.v_max

        # 位置更新
        drone.position += drone.velocity * self.dt

        # 边界约束
        drone.position[0] = np.clip(drone.position[0], 0, self.space_size[0])
        drone.position[1] = np.clip(drone.position[1], 0, self.space_size[1])
        drone.position[2] = np.clip(drone.position[2], self.z_min, self.z_max)

        # 计算奖励
        reward = self._compute_reward(drone_idx)

        # 更新电池SOC
        power = self._compute_power(drone)
        drone.battery_soc -= power * self.dt / (self.battery_capacity * 3.6)  # mAh -> Wh

        # 检查配送完成
        self._check_delivery(drone_idx)

        return reward

    def _compute_reward(self, drone_idx):
        """计算奖励"""
        drone = self.drone_states[drone_idx]
        reward = 0.0

        # 1. 进度奖励（朝目标移动）
        target = self.delivery_points[drone.current_target].position
        dist_to_target = np.linalg.norm(drone.position - target)

        if not hasattr(self, '_prev_target_dist'):
            self._prev_target_dist = {}
        prev_dist = self._prev_target_dist.get(drone_idx, dist_to_target)
        progress = (prev_dist - dist_to_target) / max(prev_dist, 1e-6)
        reward += 50.0 * progress  # 进度奖励
        self._prev_target_dist[drone_idx] = dist_to_target

        # 2. 接近奖励
        if dist_to_target < 100:
            reward += 100.0 * (1.0 - dist_to_target / 100)

        # 3. 到达奖励
        if dist_to_target < 15:
            reward += 500.0

        # 4. 障碍物惩罚
        for obs in self.obstacles:
            dist_to_obs = np.linalg.norm(drone.position - obs.position) - obs.radius
            if dist_to_obs < 0:
                reward += obs.penalty  # 碰撞
            elif dist_to_obs < 20:
                reward += obs.penalty * 0.1 * (1 - dist_to_obs / 20)  # 接近惩罚

        # 5. 能耗惩罚
        power = self._compute_power(drone)
        reward -= 0.01 * power

        # 6. 时间惩罚
        reward -= 0.1

        return reward

    def _compute_power(self, drone):
        """计算功率(W)"""
        speed = np.linalg.norm(drone.velocity)
        # 简化能耗模型
        P_hover = 100  # 悬停功率
        P_forward = 0.5 * speed ** 2  # 前进功率
        P_climb = 50 * max(drone.velocity[2], 0)  # 爬升功率
        return P_hover + P_forward + P_climb

    def _check_delivery(self, drone_idx):
        """检查配送完成"""
        drone = self.drone_states[drone_idx]
        target = self.delivery_points[drone.current_target]
        dist = np.linalg.norm(drone.position - target.position)

        if dist < 15 and not drone.delivered[drone.current_target]:
            drone.delivered[drone.current_target] = True
            drone.payload -= target.demand

            # 移动到下一个目标
            undelivered = [i for i, d in enumerate(drone.delivered) if not d and i > 0]
            if undelivered:
                drone.current_target = undelivered[0]
            else:
                drone.current_target = 0  # 返回配送中心

    def _update_dynamic_obstacles(self):
        """更新动态障碍物位置"""
        for obs in self.obstacles:
            if obs.is_dynamic and obs.speed is not None:
                obs.position = obs.position.astype(np.float64) + obs.speed * self.dt
                # 边界反弹
                for i in range(2):
                    if obs.position[i] < 0 or obs.position[i] > self.space_size[i]:
                        obs.speed[i] *= -1

    def _check_done(self):
        """检查终止条件"""
        # 所有配送完成
        if all(all(ds.delivered) for ds in self.drone_states):
            return True

        # 超过最大步数
        if self.step_count >= self.max_steps:
            return True

        # 电池耗尽
        if any(ds.battery_soc <= 0 for ds in self.drone_states):
            return True

        return False

    def _count_collisions(self):
        """统计碰撞次数"""
        collisions = 0
        for drone in self.drone_states:
            for obs in self.obstacles:
                dist = np.linalg.norm(drone.position - obs.position) - obs.radius
                if dist < 0:
                    collisions += 1
        return collisions

    def _get_obs(self):
        """获取观测"""
        obs = []
        for drone in self.drone_states:
            # 目标信息
            target = self.delivery_points[drone.current_target].position
            dist_to_target = np.linalg.norm(drone.position - target)
            direction_to_target = (target - drone.position) / max(dist_to_target, 1e-6)

            # 最近障碍物距离
            min_obs_dist = float('inf')
            for obs_item in self.obstacles:
                dist = np.linalg.norm(drone.position - obs_item.position) - obs_item.radius
                min_obs_dist = min(min_obs_dist, dist)

            # 配送进度
            delivery_progress = sum(drone.delivered) / len(drone.delivered)

            # 拼接状态
            state = np.concatenate([
                drone.position / 1000,  # 归一化位置
                drone.velocity / self.v_max,  # 归一化速度
                [drone.battery_soc],  # 电池SOC
                [drone.payload / self.max_payload],  # 归一化载重
                [dist_to_target / 1000],  # 归一化距离
                direction_to_target,  # 目标方向
                [min(min_obs_dist, 200) / 200],  # 归一化障碍物距离
                [delivery_progress],  # 配送进度
            ])
            obs.append(state)

        return np.concatenate(obs).astype(np.float32)

    def get_environment_info(self):
        """获取环境信息"""
        return {
            'space_size': self.space_size,
            'num_drones': self.num_drones,
            'num_delivery_points': self.num_delivery_points,
            'obstacles': [
                {
                    'type': obs.obs_type.value,
                    'position': obs.position.tolist(),
                    'size': obs.size.tolist(),
                    'radius': obs.radius,
                    'height': obs.height,
                    'color': obs.color,
                    'penalty': obs.penalty,
                    'is_dynamic': obs.is_dynamic,
                }
                for obs in self.obstacles
            ],
            'delivery_points': [
                {
                    'id': dp.point_id,
                    'position': dp.position.tolist(),
                    'demand': dp.demand,
                    'time_window': dp.time_window,
                    'priority': dp.priority,
                }
                for dp in self.delivery_points
            ],
        }


# 测试
if __name__ == '__main__':
    env = UrbanAirspaceEnv()
    state = env.reset()

    print(f"状态维度: {state.shape}")
    print(f"障碍物数量: {len(env.obstacles)}")
    print(f"配送点数量: {len(env.delivery_points)}")

    info = env.get_environment_info()
    print(f"\n障碍物类型:")
    for obs in info['obstacles']:
        print(f"  {obs['type']}: {obs['position']}, 惩罚={obs['penalty']}")

    print(f"\n配送点:")
    for dp in info['delivery_points']:
        print(f"  点{dp['id']}: {dp['position']}, 需求={dp['demand']:.0f}g")
