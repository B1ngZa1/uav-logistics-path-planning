"""
问题层级二：2D随机商家配送环境

在层级一的基础上新增：
- 每回合随机生成4-8个订单（商家 + 配送点 + 时间窗口）
- 8方向离散动作
- 多单携带（最多2单）
- 时间窗口约束

适合DQN、A2C、PPO等使用函数近似的算法。

使用示例:
    from environment.random_delivery_env import Random2DDeliveryEnv

    env = Random2DDeliveryEnv()
    obs, info = env.reset(seed=42)
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Dict, Any, Tuple, List

from environment.simple_delivery_env import Simple2DDeliveryEnv


class Random2DDeliveryEnv(gym.Env):
    """问题层级二：2D随机商家配送环境

    模拟高峰期随机涌来的外卖订单场景。每回合生成随机订单，
    包含时间窗口约束和支持同时携带多单。

    ## 新增特性（相对层级一）
    - 8方向离散动作（增加4个对角线方向）
    - 随机订单生成（4-8单/回合）
    - 时间窗口（超时扣分，超时后订单作废）
    - 多单携带（最多2单）

    ## 动作空间
    Discrete(8) — 0=上, 1=下, 2=左, 3=右, 4=左上, 5=右上, 6=左下, 7=右下

    ## 奖励
    - +100: 按时完成一单配送
    - +50: 时间窗口内完成奖励
    - +30: 超时后完成（仅给基础分）
    - -30: 订单超时未完成
    - +0.5: 多单携带每步鼓励
    - +500: 全部完成并返回
    - -1: 每步时间惩罚
    - -200: 碰撞（终止）
    """

    metadata = {"render_modes": ["human"], "render_fps": 10}

    # 8方向动作
    ACTION_DELTAS_8 = {
        0: np.array([0, 1]),     # 上
        1: np.array([0, -1]),    # 下
        2: np.array([-1, 0]),    # 左
        3: np.array([1, 0]),     # 右
        4: np.array([-1, 1]),    # 左上
        5: np.array([1, 1]),     # 右上
        6: np.array([-1, -1]),   # 左下
        7: np.array([1, -1]),    # 右下
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None, render_mode: Optional[str] = None):
        super().__init__()
        config = config or {}

        self.grid_size = config.get("grid_size", 40)
        self.max_carry = config.get("max_carry", 2)
        self.min_orders = config.get("min_orders", 4)
        self.max_orders = config.get("max_orders", 8)
        self.max_steps = config.get("max_steps", 800)
        self.render_mode = render_mode

        # 8方向动作
        self.action_space = spaces.Discrete(8)

        # 观测空间 — 固定大小的特征向量化
        self.max_orders_encode = self.max_orders
        self.state_dim = (2 +                          # 无人机位置
                          self.max_orders_encode * 6 +  # 每个订单6维特征
                          2)                            # 携带状态 + 步数比例
        self.observation_space = spaces.Dict({
            "position": spaces.Box(0, self.grid_size - 1, shape=(2,), dtype=np.int32),
            "vector": spaces.Box(-np.inf, np.inf, shape=(self.state_dim,), dtype=np.float32),
        })

        # 固定地图
        self.depot = np.array(config.get("depot", [30, 10]))
        self.obstacles = self._build_obstacle_mask()

        # 运行时状态
        self.position = self.depot.copy()
        self.orders: List[Dict] = []
        self.carrying_list: List[int] = []
        self.steps = 0

    def _build_obstacle_mask(self) -> np.ndarray:
        """与层级一相同的障碍物布局"""
        mask = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        mask[14:19, 14:21] = True
        mask[26:31, 20:25] = True
        mask[8:13, 4:9] = True
        mask[22, 12:17] = True
        mask[6:8, 22:24] = True
        mask[19:21, 10:12] = True
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, 0] = True
        mask[:, -1] = True
        mask[self.depot[0], self.depot[1]] = False
        return mask

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None
              ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)

        self.position = self.depot.copy()
        self.carrying_list = []
        self.steps = 0

        # 随机生成订单
        self.orders = self._generate_orders(self.np_random)

        return self._get_obs(), self._get_info()

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        # 移动
        dx, dy = self.ACTION_DELTAS_8.get(action, (0, 0))
        new_pos = self.position + np.array([dx, dy])
        if self._is_valid(new_pos):
            self.position = new_pos

        reward = -1.0
        self.steps += 1

        # 检查过期订单
        for order in self.orders:
            if order["status"] == 1 and self.steps > order["deadline"]:
                order["status"] = -1  # 超时失败
                reward -= 30.0

        # 取货逻辑
        if len(self.carrying_list) < self.max_carry:
            for order in self.orders:
                if order["status"] == 0:  # 未接单
                    if np.array_equal(self.position, order["merchant"]):
                        order["status"] = 1
                        order["pickup_time"] = self.steps
                        self.carrying_list.append(order["id"])
                        break

        # 送货逻辑
        delivered = []
        for oid in self.carrying_list:
            order = self.orders[oid]
            if np.array_equal(self.position, order["delivery"]):
                if self.steps <= order["deadline"]:
                    reward += 100.0  # 按时完成基础分
                    reward += 50.0   # 准时奖励
                else:
                    reward += 30.0   # 超时完成仅给基础分
                order["status"] = 2
                order["delivery_time"] = self.steps
                delivered.append(oid)

        for oid in delivered:
            self.carrying_list.remove(oid)

        # 多单携带鼓励
        if len(self.carrying_list) >= 2:
            reward += 0.5

        # 终止判断
        terminated = self._check_termination(reward)
        truncated = self.steps >= self.max_steps

        return self._get_obs(), float(reward), terminated, truncated, self._get_info()

    def _generate_orders(self, rng: np.random.Generator) -> List[Dict]:
        """随机生成4-8个订单"""
        num_orders = rng.integers(self.min_orders, self.max_orders + 1)
        orders = []

        for k in range(num_orders):
            # 随机商家（避开障碍物和已有商家）
            for _ in range(100):
                merchant = np.array([
                    rng.integers(1, self.grid_size - 1),
                    rng.integers(1, self.grid_size - 1)
                ])
                if not self.obstacles[merchant[0], merchant[1]]:
                    if not any(np.array_equal(merchant, o["merchant"]) for o in orders):
                        break

            # 随机配送点
            for _ in range(100):
                delivery = np.array([
                    rng.integers(1, self.grid_size - 1),
                    rng.integers(1, self.grid_size - 1)
                ])
                if not self.obstacles[delivery[0], delivery[1]]:
                    if not np.array_equal(delivery, merchant):
                        break

            # 基于距离估算时间窗口
            dist = (abs(merchant - self.depot).sum() +
                    abs(delivery - merchant).sum() +
                    abs(self.depot - delivery).sum())
            deadline = int(dist * 1.5 + rng.integers(50, 200))
            deadline = min(deadline, self.max_steps - 50)

            # 优先级
            priority = int(rng.choice([1, 2, 3], p=[0.6, 0.3, 0.1]))

            orders.append({
                "id": k,
                "merchant": merchant,
                "delivery": delivery,
                "deadline": deadline,
                "status": 0,   # 0=未接单, 1=已取货, 2=已完成, -1=已超时
                "priority": priority,
                "pickup_time": None,
                "delivery_time": None,
            })

        return orders

    def _is_valid(self, pos: np.ndarray) -> bool:
        x, y = int(pos[0]), int(pos[1])
        return (0 <= x < self.grid_size and 0 <= y < self.grid_size
                and not self.obstacles[x, y])

    def _check_termination(self, current_reward: float) -> bool:
        """检查是否终止"""
        # 碰撞
        if not self._is_valid(self.position):
            return True

        # 全部订单已完成/失败
        all_done = all(o["status"] in (2, -1) for o in self.orders)
        if all_done and np.array_equal(self.position, self.depot):
            return True

        # 所有订单都已超时失败
        all_failed = all(o["status"] == -1 for o in self.orders)
        if all_failed:
            return True

        return False

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """构建观测 — 包含固定维度的向量化特征"""
        # 基础特征
        features = [
            self.position[0] / self.grid_size,
            self.position[1] / self.grid_size,
        ]

        # 每个订单的6维特征
        for k in range(self.max_orders_encode):
            if k < len(self.orders):
                order = self.orders[k]
                features.extend([
                    order["merchant"][0] / self.grid_size,
                    order["merchant"][1] / self.grid_size,
                    order["delivery"][0] / self.grid_size,
                    order["delivery"][1] / self.grid_size,
                    order["status"] / 2.0,  # 归一化状态
                    min(order["deadline"] / self.max_steps, 1.0),
                ])
            else:
                features.extend([0.0] * 6)

        # 自身状态
        features.append(len(self.carrying_list) / self.max_carry)
        features.append(self.steps / self.max_steps)

        return {
            "position": self.position.copy(),
            "vector": np.array(features, dtype=np.float32),
        }

    def _get_info(self) -> Dict[str, Any]:
        completed = sum(1 for o in self.orders if o["status"] == 2)
        failed = sum(1 for o in self.orders if o["status"] == -1)
        on_time = sum(1 for o in self.orders
                      if o["status"] == 2 and o.get("delivery_time", 9999) <= o["deadline"])

        return {
            "position": self.position.tolist(),
            "total_orders": len(self.orders),
            "completed": completed,
            "failed": failed,
            "on_time": on_time,
            "carrying_count": len(self.carrying_list),
            "steps": self.steps,
            "depot_reached": bool(np.array_equal(self.position, self.depot)),
        }

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode != "human":
            return None

        grid = [["·" for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if self.obstacles[x, y]:
                    grid[x][y] = "█"

        grid[self.depot[0]][self.depot[1]] = "D"

        for order in self.orders:
            if order["status"] == 0:
                grid[order["merchant"][0]][order["merchant"][1]] = f"M{order['id']}"
            elif order["status"] == 1:
                grid[order["delivery"][0]][order["delivery"][1]] = f"T{order['id']}"
            elif order["status"] == -1:
                grid[order["merchant"][0]][order["merchant"][1]] = "X"

        px, py = int(self.position[0]), int(self.position[1])
        grid[px][py] = "●" if not self.carrying_list else f"◉{len(self.carrying_list)}"

        lines = [" ".join(row) for row in grid]
        print("\n".join(lines))
        print(f"Steps: {self.steps}, Carrying: {len(self.carrying_list)}")
        print(f"Active orders: {sum(1 for o in self.orders if o['status'] == 0)}")
        print(f"In-transit: {sum(1 for o in self.orders if o['status'] == 1)}")
        print(f"Completed: {sum(1 for o in self.orders if o['status'] == 2)}")
        print(f"Failed: {sum(1 for o in self.orders if o['status'] == -1)}")

        return None


gym.envs.registration.register(
    id="Random2DDelivery-v0",
    entry_point="environment.random_delivery_env:Random2DDeliveryEnv",
    max_episode_steps=800,
)
