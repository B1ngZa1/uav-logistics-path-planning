"""
多订单动态任务分配算法实现

项目：基于深度强化学习的多无人机协同物流配送路径规划
模块：任务分配
创建日期：2026-07-01
版本：v1.0
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import heapq
from scipy.optimize import linear_sum_assignment


# ============================================================
# 数据结构定义
# ============================================================

class OrderStatus(Enum):
    """订单状态枚举"""
    PENDING = "pending"          # 待分配
    ASSIGNED = "assigned"        # 已分配
    PICKING = "picking"          # 取货中
    DELIVERING = "delivering"    # 配送中
    COMPLETED = "completed"      # 已完成
    OVERDUE = "overdue"          # 已超时


class DroneStatus(Enum):
    """无人机状态枚举"""
    IDLE = "idle"                # 空闲
    FLYING = "flying"            # 飞行中
    PICKING = "picking"          # 取货中
    DELIVERING = "delivering"    # 配送中
    CHARGING = "charging"        # 充电中
    FAULT = "fault"              # 故障


class TriggerEvent(Enum):
    """触发事件类型"""
    NEW_ORDER = "new_order"              # 新订单到达
    BATTERY_LOW = "battery_low"          # 电量预警
    TIMEOUT_RISK = "timeout_risk"        # 超时风险
    UAV_FAULT = "uav_fault"              # 无人机故障
    PERIODIC_REOPT = "periodic_reopt"    # 定期重优化


@dataclass
class Position3D:
    """三维位置"""
    x: float
    y: float
    z: float

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    def distance_to(self, other: 'Position3D') -> float:
        return np.sqrt((self.x - other.x)**2 +
                       (self.y - other.y)**2 +
                       (self.z - other.z)**2)


@dataclass
class Order:
    """订单数据类"""
    order_id: str
    pickup_pos: Position3D           # 取货位置
    delivery_pos: Position3D         # 配送位置
    weight: float                    # 货物重量 (kg)
    created_time: float              # 创建时间
    deadline: float                  # 截止时间
    priority: int = 3                # 基础优先级 (1-5)
    status: OrderStatus = OrderStatus.PENDING

    @property
    def delivery_distance(self) -> float:
        """取货点到配送点的距离"""
        return self.pickup_pos.distance_to(self.delivery_pos)


@dataclass
class Drone:
    """无人机数据类"""
    drone_id: str
    position: Position3D             # 当前位置
    velocity: Position3D             # 当前速度
    remaining_energy: float          # 剩余电量 (Wh)
    max_energy: float                # 最大电量 (Wh)
    max_load: float                  # 最大载重 (kg)
    current_load: float = 0.0        # 当前载重 (kg)
    avg_speed: float = 15.0          # 平均飞行速度 (m/s)
    status: DroneStatus = DroneStatus.IDLE
    task_list: List[str] = field(default_factory=list)  # 已分配的订单ID列表

    @property
    def is_available(self) -> bool:
        """是否可接受新任务"""
        return self.status in [DroneStatus.IDLE, DroneStatus.FLYING]

    @property
    def energy_ratio(self) -> float:
        """剩余电量比例"""
        return self.remaining_energy / self.max_energy


@dataclass
class AllocationResult:
    """分配结果"""
    allocation: Dict[str, List[str]]  # {drone_id: [order_ids]}
    total_cost: float
    algorithm: str
    timestamp: float


# ============================================================
# 代价矩阵构建
# ============================================================

class CostMatrixBuilder:
    """代价矩阵构建器"""

    def __init__(self,
                 alpha: float = 0.3,      # 距离权重
                 beta: float = 0.3,       # 时间权重
                 gamma: float = 0.2,      # 能耗权重
                 delta: float = 0.2,      # 紧急度权重
                 lambda_balance: float = 0.3):  # 负载均衡权重
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.lambda_balance = lambda_balance

    def build(self,
              drones: List[Drone],
              orders: List[Order],
              current_time: float) -> np.ndarray:
        """
        构建代价矩阵

        参数:
            drones: 无人机列表
            orders: 订单列表
            current_time: 当前时间

        返回:
            cost_matrix: numpy数组, 形状 (N, M)
        """
        N = len(drones)
        M = len(orders)
        cost_matrix = np.zeros((N, M))

        for i, drone in enumerate(drones):
            for j, order in enumerate(orders):
                cost_matrix[i, j] = self._compute_cost(
                    drone, order, current_time
                )

        return cost_matrix

    def _compute_cost(self,
                      drone: Drone,
                      order: Order,
                      current_time: float) -> float:
        """计算单个无人机-订单对的综合代价"""
        # 距离代价: 当前位置 -> 取货点 -> 配送点
        dist_to_pickup = drone.position.distance_to(order.pickup_pos)
        dist_pickup_to_delivery = order.pickup_pos.distance_to(order.delivery_pos)
        total_distance = dist_to_pickup + dist_pickup_to_delivery

        # 时间代价
        estimated_time = total_distance / drone.avg_speed

        # 能耗代价 (简化模型)
        energy_consumption = total_distance * 0.5  # Wh/m (简化)

        # 紧急度代价
        time_remaining = order.deadline - current_time
        if time_remaining > 0:
            urgency = estimated_time / time_remaining
        else:
            urgency = 10.0  # 已超时，高代价

        # 负载均衡代价
        load_factor = len(drone.task_list) / 10.0  # 归一化

        # 综合代价
        cost = (self.alpha * total_distance +
                self.beta * estimated_time +
                self.gamma * energy_consumption +
                self.delta * urgency +
                self.lambda_balance * load_factor)

        return cost


# ============================================================
# 约束检查
# ============================================================

class ConstraintChecker:
    """约束条件检查器"""

    def __init__(self,
                 energy_safety_margin: float = 0.2,  # 电量安全余量
                 time_safety_margin: float = 0.1):    # 时间安全余量
        self.energy_safety_margin = energy_safety_margin
        self.time_safety_margin = time_safety_margin

    def check_all(self,
                  drone: Drone,
                  order: Order,
                  current_time: float) -> Tuple[bool, str]:
        """
        检查所有约束条件

        返回:
            (is_valid, reason): 是否有效及原因
        """
        # 容量约束
        if drone.current_load + order.weight > drone.max_load:
            return False, "容量超限"

        # 电量约束
        energy_required = self._estimate_energy(drone, order)
        if drone.remaining_energy < energy_required * (1 + self.energy_safety_margin):
            return False, "电量不足"

        # 时间约束
        estimated_arrival = self._estimate_arrival_time(drone, order, current_time)
        if estimated_arrival > order.deadline * (1 - self.time_safety_margin):
            return False, "无法按时送达"

        return True, "满足所有约束"

    def _estimate_energy(self, drone: Drone, order: Order) -> float:
        """估算配送所需能耗"""
        total_distance = (drone.position.distance_to(order.pickup_pos) +
                          order.pickup_pos.distance_to(order.delivery_pos))
        return total_distance * 0.5  # 简化能耗模型

    def _estimate_arrival_time(self,
                                drone: Drone,
                                order: Order,
                                current_time: float) -> float:
        """估算到达配送点的时间"""
        total_distance = (drone.position.distance_to(order.pickup_pos) +
                          order.pickup_pos.distance_to(order.delivery_pos))
        flight_time = total_distance / drone.avg_speed
        hover_time = 30.0  # 取货+配送悬停时间 (秒)
        return current_time + flight_time + hover_time


# ============================================================
# 贪心算法
# ============================================================

class GreedyAllocator:
    """
    贪心任务分配算法

    策略：每次为空闲无人机分配代价最小的未分配订单
    时间复杂度: O(M^2)
    适用场景: 实时动态分配
    """

    def __init__(self):
        self.cost_builder = CostMatrixBuilder()
        self.constraint_checker = ConstraintChecker()

    def allocate(self,
                 orders: List[Order],
                 drones: List[Drone],
                 current_time: float) -> AllocationResult:
        """
        执行贪心分配

        参数:
            orders: 待分配订单列表
            drones: 无人机列表
            current_time: 当前时间

        返回:
            AllocationResult: 分配结果
        """
        allocation = {drone.drone_id: [] for drone in drones}
        unassigned = list(orders)
        total_cost = 0.0

        # 构建初始代价矩阵
        cost_matrix = self.cost_builder.build(drones, orders, current_time)

        iteration = 0
        max_iterations = len(orders) * len(drones)

        while unassigned and iteration < max_iterations:
            iteration += 1
            assigned_in_round = False

            for i, drone in enumerate(drones):
                if not drone.is_available:
                    continue

                if not unassigned:
                    break

                # 计算当前无人机到所有未分配订单的代价
                best_order = None
                best_cost = float('inf')
                best_idx = -1

                for j, order in enumerate(unassigned):
                    # 找到订单在原代价矩阵中的索引
                    original_idx = orders.index(order)
                    cost = cost_matrix[i, original_idx]

                    # 检查约束
                    is_valid, _ = self.constraint_checker.check_all(
                        drone, order, current_time
                    )

                    if is_valid and cost < best_cost:
                        best_cost = cost
                        best_order = order
                        best_idx = j

                if best_order is not None:
                    # 执行分配
                    allocation[drone.drone_id].append(best_order.order_id)
                    drone.task_list.append(best_order.order_id)
                    drone.current_load += best_order.weight
                    best_order.status = OrderStatus.ASSIGNED
                    unassigned.pop(best_idx)
                    total_cost += best_cost
                    assigned_in_round = True

            if not assigned_in_round:
                break  # 无法继续分配

        return AllocationResult(
            allocation=allocation,
            total_cost=total_cost,
            algorithm="greedy",
            timestamp=current_time
        )


# ============================================================
# 匈牙利算法
# ============================================================

class HungarianAllocator:
    """
    匈牙利任务分配算法

    策略：求解二分图最优匹配，保证全局最优
    时间复杂度: O(max(N,M)^3)
    适用场景: 静态批量分配
    """

    def __init__(self):
        self.cost_builder = CostMatrixBuilder()
        self.constraint_checker = ConstraintChecker()

    def allocate(self,
                 orders: List[Order],
                 drones: List[Drone],
                 current_time: float) -> AllocationResult:
        """
        执行匈牙利算法分配

        参数:
            orders: 待分配订单列表
            drones: 无人机列表
            current_time: 当前时间

        返回:
            AllocationResult: 分配结果
        """
        N = len(drones)
        M = len(orders)

        if N == 0 or M == 0:
            return AllocationResult({}, 0.0, "hungarian", current_time)

        # 构建代价矩阵
        cost_matrix = self.cost_builder.build(drones, orders, current_time)

        # 处理非方阵情况
        extended_cost, padding_info = self._extend_to_square(cost_matrix, N, M)

        # 应用匈牙利算法
        row_ind, col_ind = linear_sum_assignment(extended_cost)

        # 构建分配结果
        allocation = {drone.drone_id: [] for drone in drones}
        total_cost = 0.0

        for i, j in zip(row_ind, col_ind):
            # 排除虚拟行列
            if i >= N or j >= M:
                continue

            drone = drones[i]
            order = orders[j]

            # 检查约束
            is_valid, reason = self.constraint_checker.check_all(
                drone, order, current_time
            )

            if is_valid:
                allocation[drone.drone_id].append(order.order_id)
                drone.task_list.append(order.order_id)
                drone.current_load += order.weight
                order.status = OrderStatus.ASSIGNED
                total_cost += cost_matrix[i, j]

        return AllocationResult(
            allocation=allocation,
            total_cost=total_cost,
            algorithm="hungarian",
            timestamp=current_time
        )

    def _extend_to_square(self,
                          cost_matrix: np.ndarray,
                          N: int,
                          M: int) -> Tuple[np.ndarray, dict]:
        """将代价矩阵扩展为方阵"""
        n = max(N, M)
        extended = np.zeros((n, n))

        if N < M:
            # 无人机少于订单，添加虚拟无人机（高代价）
            extended[:N, :M] = cost_matrix
            extended[N:, :] = 1e6  # 虚拟无人机的高代价
        elif N > M:
            # 无人机多于订单，添加虚拟订单（零代价）
            extended[:N, :M] = cost_matrix
            # 虚拟订单代价为0，表示不执行任务
        else:
            extended = cost_matrix.copy()

        padding_info = {"original_N": N, "original_M": M, "extended_size": n}
        return extended, padding_info


# ============================================================
# 优先级队列管理
# ============================================================

class PriorityOrderWrapper:
    """用于优先级队列的订单包装类"""

    def __init__(self, order: Order, priority_score: float):
        self.order = order
        self.priority_score = priority_score

    def __lt__(self, other):
        # 注意：负号使得最高优先级排在前面
        return self.priority_score > other.priority_score


class PriorityManager:
    """优先级管理器"""

    def __init__(self,
                 lambda_urgency: float = 0.6,
                 lambda_wait: float = 0.4):
        self.lambda_urgency = lambda_urgency
        self.lambda_wait = lambda_wait
        self._queue: List[PriorityOrderWrapper] = []

    def add_order(self, order: Order, current_time: float):
        """添加订单到优先级队列"""
        score = self._compute_priority(order, current_time)
        wrapper = PriorityOrderWrapper(order, score)
        heapq.heappush(self._queue, wrapper)

    def pop_highest(self) -> Optional[Order]:
        """取出最高优先级订单"""
        if self._queue:
            wrapper = heapq.heappop(self._queue)
            return wrapper.order
        return None

    def update_all_priorities(self, current_time: float):
        """更新所有订单的优先级"""
        updated = []
        while self._queue:
            wrapper = heapq.heappop(self._queue)
            new_score = self._compute_priority(wrapper.order, current_time)
            wrapper.priority_score = new_score
            updated.append(wrapper)

        for wrapper in updated:
            heapq.heappush(self._queue, wrapper)

    def _compute_priority(self, order: Order, current_time: float) -> float:
        """
        计算订单综合优先级

        priority_total = priority_base + lambda_1 * urgency + lambda_2 * wait
        """
        # 基础优先级（直接使用订单的priority字段）
        base_priority = order.priority

        # 紧急度
        time_ratio = (order.deadline - current_time) / (order.deadline - order.created_time)
        if time_ratio <= 0.2:
            urgency = 5
        elif time_ratio <= 0.5:
            urgency = 2
        else:
            urgency = 0

        # 等待时间
        wait_time = min(3, int((current_time - order.created_time) / 300))

        # 综合优先级
        total = base_priority + self.lambda_urgency * urgency + self.lambda_wait * wait_time
        return total

    @property
    def size(self) -> int:
        return len(self._queue)

    @property
    def is_empty(self) -> bool:
        return len(self._queue) == 0


# ============================================================
# 负载均衡器
# ============================================================

class LoadBalancer:
    """负载均衡管理器"""

    def compute_load_metrics(self, drones: List[Drone]) -> Dict[str, float]:
        """
        计算各无人机的负载指标

        返回: {drone_id: load_metric}
        """
        metrics = {}
        for drone in drones:
            # 负载指标 = 任务数归一化 + 能耗归一化
            task_factor = len(drone.task_list) / 10.0  # 假设最大10个任务
            energy_factor = (drone.max_energy - drone.remaining_energy) / drone.max_energy
            metrics[drone.drone_id] = task_factor + energy_factor
        return metrics

    def compute_balance_score(self, drones: List[Drone]) -> float:
        """
        计算负载均衡度

        返回: [0, 1] 之间的值，越大越均衡
        """
        metrics = self.compute_load_metrics(drones)
        values = list(metrics.values())

        if not values or all(v == 0 for v in values):
            return 1.0

        mean_val = np.mean(values)
        std_val = np.std(values)

        if mean_val == 0:
            return 1.0

        balance = 1.0 - (std_val / mean_val)
        return max(0.0, min(1.0, balance))

    def get_load_balanced_cost_adjustment(self,
                                          drones: List[Drone],
                                          base_cost_matrix: np.ndarray) -> np.ndarray:
        """
        根据负载均衡调整代价矩阵

        参数:
            drones: 无人机列表
            base_cost_matrix: 原始代价矩阵

        返回:
            adjusted_cost_matrix: 调整后的代价矩阵
        """
        metrics = self.compute_load_metrics(drones)
        adjusted = base_cost_matrix.copy()

        for i, drone in enumerate(drones):
            load_factor = metrics[drone.drone_id]
            adjusted[i, :] += 0.3 * load_factor  # lambda_balance = 0.3

        return adjusted


# ============================================================
# 动态重分配器
# ============================================================

class DynamicReallocator:
    """动态任务重分配器"""

    def __init__(self):
        self.greedy_allocator = GreedyAllocator()
        self.hungarian_allocator = HungarianAllocator()
        self.priority_manager = PriorityManager()
        self.load_balancer = LoadBalancer()
        self.constraint_checker = ConstraintChecker()

    def handle_event(self,
                     event: TriggerEvent,
                     drones: List[Drone],
                     orders: List[Order],
                     current_allocation: Dict[str, List[str]],
                     current_time: float,
                     **kwargs) -> AllocationResult:
        """
        处理触发事件并执行重分配

        参数:
            event: 触发事件类型
            drones: 无人机列表
            orders: 订单列表
            current_allocation: 当前分配
            current_time: 当前时间
            **kwargs: 事件相关参数

        返回:
            AllocationResult: 更新后的分配结果
        """
        if event == TriggerEvent.NEW_ORDER:
            return self._handle_new_order(
                kwargs.get('new_order'),
                drones, orders, current_allocation, current_time
            )
        elif event == TriggerEvent.BATTERY_LOW:
            return self._handle_battery_low(
                kwargs.get('drone_id'),
                drones, orders, current_allocation, current_time
            )
        elif event == TriggerEvent.TIMEOUT_RISK:
            return self._handle_timeout_risk(
                drones, orders, current_allocation, current_time
            )
        elif event == TriggerEvent.UAV_FAULT:
            return self._handle_uav_fault(
                kwargs.get('drone_id'),
                drones, orders, current_allocation, current_time
            )
        elif event == TriggerEvent.PERIODIC_REOPT:
            return self._handle_periodic_reopt(
                drones, orders, current_allocation, current_time
            )
        else:
            return AllocationResult(
                current_allocation, 0.0, "none", current_time
            )

    def _handle_new_order(self,
                          new_order: Order,
                          drones: List[Drone],
                          orders: List[Order],
                          current_allocation: Dict[str, List[str]],
                          current_time: float) -> AllocationResult:
        """处理新订单到达"""
        # 将新订单加入待分配列表
        pending_orders = [o for o in orders if o.status == OrderStatus.PENDING]
        pending_orders.append(new_order)

        # 获取空闲无人机
        idle_drones = [d for d in drones if d.is_available]

        if not idle_drones or not pending_orders:
            return AllocationResult(
                current_allocation, 0.0, "no_change", current_time
            )

        # 使用贪心算法快速分配
        result = self.greedy_allocator.allocate(
            pending_orders, idle_drones, current_time
        )

        # 合并分配结果
        merged_allocation = current_allocation.copy()
        for drone_id, order_ids in result.allocation.items():
            if drone_id in merged_allocation:
                merged_allocation[drone_id].extend(order_ids)
            else:
                merged_allocation[drone_id] = order_ids

        return AllocationResult(
            merged_allocation, result.total_cost, "greedy_new_order", current_time
        )

    def _handle_battery_low(self,
                            drone_id: str,
                            drones: List[Drone],
                            orders: List[Order],
                            current_allocation: Dict[str, List[str]],
                            current_time: float) -> AllocationResult:
        """处理电量预警"""
        # 找到低电量无人机
        low_battery_drone = next((d for d in drones if d.drone_id == drone_id), None)
        if not low_battery_drone:
            return AllocationResult(
                current_allocation, 0.0, "no_change", current_time
            )

        # 标记为不可用
        low_battery_drone.status = DroneStatus.CHARGING

        # 获取该无人机的待执行订单
        pending_order_ids = current_allocation.get(drone_id, [])
        pending_orders = [o for o in orders if o.order_id in pending_order_ids
                         and o.status in [OrderStatus.ASSIGNED, OrderStatus.PICKING]]

        # 将未取货的订单退回待分配队列
        orders_to_reassign = []
        for order in pending_orders:
            if order.status == OrderStatus.ASSIGNED:
                order.status = OrderStatus.PENDING
                orders_to_reassign.append(order)

        # 重新分配这些订单
        if orders_to_reassign:
            available_drones = [d for d in drones
                               if d.is_available and d.drone_id != drone_id]
            if available_drones:
                result = self.greedy_allocator.allocate(
                    orders_to_reassign, available_drones, current_time
                )

                # 更新分配
                merged_allocation = current_allocation.copy()
                merged_allocation[drone_id] = []  # 清空低电量无人机的任务
                for did, oids in result.allocation.items():
                    if did in merged_allocation:
                        merged_allocation[did].extend(oids)
                    else:
                        merged_allocation[did] = oids

                return AllocationResult(
                    merged_allocation, result.total_cost,
                    "greedy_battery_low", current_time
                )

        return AllocationResult(
            current_allocation, 0.0, "no_reassignment", current_time
        )

    def _handle_timeout_risk(self,
                             drones: List[Drone],
                             orders: List[Order],
                             current_allocation: Dict[str, List[str]],
                             current_time: float) -> AllocationResult:
        """处理超时风险"""
        # 找出有超时风险的订单
        at_risk_orders = []
        for order in orders:
            if order.status in [OrderStatus.ASSIGNED, OrderStatus.PICKING, OrderStatus.DELIVERING]:
                time_remaining = order.deadline - current_time
                # 估算剩余配送时间
                assigned_drone = self._find_assigned_drone(
                    order.order_id, drones, current_allocation
                )
                if assigned_drone:
                    dist = assigned_drone.position.distance_to(order.delivery_pos)
                    est_time = dist / assigned_drone.avg_speed + 30  # +悬停时间
                    if est_time > time_remaining:
                        at_risk_orders.append((order, assigned_drone))

        if not at_risk_orders:
            return AllocationResult(
                current_allocation, 0.0, "no_timeout_risk", current_time
            )

        # 尝试转移高风险订单
        merged_allocation = current_allocation.copy()
        for order, current_drone in at_risk_orders:
            # 寻找更近的可用无人机
            best_drone = None
            best_time = float('inf')

            for drone in drones:
                if not drone.is_available or drone.drone_id == current_drone.drone_id:
                    continue

                dist = drone.position.distance_to(order.pickup_pos)
                est_time = dist / drone.avg_speed + order.delivery_distance / drone.avg_speed + 30

                if est_time < best_time and est_time < (order.deadline - current_time):
                    best_time = est_time
                    best_drone = drone

            if best_drone:
                # 执行转移
                if current_drone.drone_id in merged_allocation:
                    merged_allocation[current_drone.drone_id] = [
                        oid for oid in merged_allocation[current_drone.drone_id]
                        if oid != order.order_id
                    ]
                if best_drone.drone_id not in merged_allocation:
                    merged_allocation[best_drone.drone_id] = []
                merged_allocation[best_drone.drone_id].append(order.order_id)

        return AllocationResult(
            merged_allocation, 0.0, "timeout_transfer", current_time
        )

    def _handle_uav_fault(self,
                          drone_id: str,
                          drones: List[Drone],
                          orders: List[Order],
                          current_allocation: Dict[str, List[str]],
                          current_time: float) -> AllocationResult:
        """处理无人机故障"""
        # 找到故障无人机
        fault_drone = next((d for d in drones if d.drone_id == drone_id), None)
        if not fault_drone:
            return AllocationResult(
                current_allocation, 0.0, "no_change", current_time
            )

        # 标记故障
        fault_drone.status = DroneStatus.FAULT

        # 获取故障无人机的所有任务
        fault_order_ids = current_allocation.get(drone_id, [])
        fault_orders = [o for o in orders if o.order_id in fault_order_ids
                       and o.status != OrderStatus.COMPLETED]

        # 将未完成订单退回待分配
        for order in fault_orders:
            order.status = OrderStatus.PENDING

        # 使用匈牙利算法全局重分配
        available_drones = [d for d in drones
                           if d.is_available and d.drone_id != drone_id]
        pending_orders = [o for o in orders if o.status == OrderStatus.PENDING]

        if available_drones and pending_orders:
            result = self.hungarian_allocator.allocate(
                pending_orders, available_drones, current_time
            )

            merged_allocation = current_allocation.copy()
            merged_allocation[drone_id] = []
            for did, oids in result.allocation.items():
                if did in merged_allocation:
                    merged_allocation[did].extend(oids)
                else:
                    merged_allocation[did] = oids

            return AllocationResult(
                merged_allocation, result.total_cost,
                "hungarian_fault_recovery", current_time
            )

        return AllocationResult(
            current_allocation, 0.0, "partial_recovery", current_time
        )

    def _handle_periodic_reopt(self,
                               drones: List[Drone],
                               orders: List[Order],
                               current_allocation: Dict[str, List[str]],
                               current_time: float) -> AllocationResult:
        """处理定期重优化"""
        # 检查负载均衡度
        balance_score = self.load_balancer.compute_balance_score(drones)

        if balance_score < 0.6:
            # 负载不均衡，执行重分配
            pending_orders = [o for o in orders if o.status == OrderStatus.PENDING]
            idle_drones = [d for d in drones if d.is_available]

            if pending_orders and idle_drones:
                # 使用带负载均衡的贪心算法
                result = self.greedy_allocator.allocate(
                    pending_orders, idle_drones, current_time
                )

                merged_allocation = current_allocation.copy()
                for did, oids in result.allocation.items():
                    if did in merged_allocation:
                        merged_allocation[did].extend(oids)
                    else:
                        merged_allocation[did] = oids

                return AllocationResult(
                    merged_allocation, result.total_cost,
                    "periodic_reopt", current_time
                )

        # 更新优先级
        self.priority_manager.update_all_priorities(current_time)

        return AllocationResult(
            current_allocation, 0.0, "no_reopt_needed", current_time
        )

    def _find_assigned_drone(self,
                             order_id: str,
                             drones: List[Drone],
                             allocation: Dict[str, List[str]]) -> Optional[Drone]:
        """找到分配了指定订单的无人机"""
        for drone_id, order_ids in allocation.items():
            if order_id in order_ids:
                return next((d for d in drones if d.drone_id == drone_id), None)
        return None


# ============================================================
# 自适应分配器（主入口）
# ============================================================

class AdaptiveAllocator:
    """
    自适应任务分配器

    根据场景特点自动选择最合适的分配算法：
    - 订单少或紧急度高: 贪心算法（实时性优先）
    - 订单适中且场景静态: 匈牙利算法（最优性优先）
    - 复杂动态场景: 动态重分配（自适应性优先）
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.greedy_allocator = GreedyAllocator()
        self.hungarian_allocator = HungarianAllocator()
        self.dynamic_reallocator = DynamicReallocator()
        self.priority_manager = PriorityManager()
        self.load_balancer = LoadBalancer()

    def allocate(self,
                 orders: List[Order],
                 drones: List[Drone],
                 current_time: float,
                 is_static: bool = True) -> AllocationResult:
        """
        执行自适应分配

        参数:
            orders: 待分配订单列表
            drones: 无人机列表
            current_time: 当前时间
            is_static: 是否为静态场景

        返回:
            AllocationResult: 分配结果
        """
        num_orders = len(orders)
        urgency = self._compute_urgency(orders, current_time)

        # 策略选择
        if num_orders <= 3 or urgency > 0.8:
            # 订单少或紧急度高：使用贪心算法
            return self.greedy_allocator.allocate(orders, drones, current_time)
        elif num_orders <= 20 and is_static:
            # 订单适中且场景静态：使用匈牙利算法
            return self.hungarian_allocator.allocate(orders, drones, current_time)
        else:
            # 复杂场景：先用匈牙利算法初始分配，再用动态调整优化
            initial_result = self.hungarian_allocator.allocate(
                orders, drones, current_time
            )
            return initial_result

    def _compute_urgency(self, orders: List[Order], current_time: float) -> float:
        """计算整体紧急度"""
        if not orders:
            return 0.0

        urgency_scores = []
        for order in orders:
            time_ratio = (order.deadline - current_time) / max(
                order.deadline - order.created_time, 1
            )
            urgency_scores.append(1 - time_ratio)

        return sum(urgency_scores) / len(urgency_scores)


# ============================================================
# 示例用法
# ============================================================

def example_usage():
    """示例：展示如何使用任务分配算法"""

    # 创建无人机
    drones = [
        Drone(
            drone_id=f"UAV_{i}",
            position=Position3D(np.random.uniform(0, 100),
                               np.random.uniform(0, 100),
                               np.random.uniform(10, 50)),
            velocity=Position3D(0, 0, 0),
            remaining_energy=500.0,
            max_energy=500.0,
            max_load=5.0
        )
        for i in range(4)
    ]

    # 创建订单
    orders = [
        Order(
            order_id=f"ORDER_{j}",
            pickup_pos=Position3D(np.random.uniform(0, 100),
                                  np.random.uniform(0, 100), 0),
            delivery_pos=Position3D(np.random.uniform(0, 100),
                                    np.random.uniform(0, 100), 0),
            weight=np.random.uniform(0.5, 3.0),
            created_time=0.0,
            deadline=600.0,
            priority=np.random.randint(1, 6)
        )
        for j in range(8)
    ]

    current_time = 0.0

    # 使用自适应分配器
    allocator = AdaptiveAllocator()
    result = allocator.allocate(orders, drones, current_time)

    print(f"分配算法: {result.algorithm}")
    print(f"总代价: {result.total_cost:.2f}")
    print(f"分配结果:")
    for drone_id, order_ids in result.allocation.items():
        if order_ids:
            print(f"  {drone_id}: {order_ids}")

    # 模拟动态事件
    reallocator = DynamicReallocator()

    # 新订单到达
    new_order = Order(
        order_id="ORDER_NEW",
        pickup_pos=Position3D(50, 50, 0),
        delivery_pos=Position3D(80, 80, 0),
        weight=1.0,
        created_time=100.0,
        deadline=300.0,
        priority=5  # 高优先级
    )

    result2 = reallocator.handle_event(
        TriggerEvent.NEW_ORDER,
        drones, orders + [new_order],
        result.allocation, current_time,
        new_order=new_order
    )

    print(f"\n新订单到达后:")
    print(f"分配算法: {result2.algorithm}")
    for drone_id, order_ids in result2.allocation.items():
        if order_ids:
            print(f"  {drone_id}: {order_ids}")


if __name__ == "__main__":
    example_usage()
