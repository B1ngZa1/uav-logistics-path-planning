"""
能耗模型模块
项目：基于深度强化学习的多无人机协同物流配送路径规划
"""

import numpy as np


class EnergyModel:
    """无人机能耗模型"""

    def __init__(self, config):
        self.k1 = config.get('k1', 0.8)              # 前进功率系数
        self.k2 = config.get('k2', 1.2)              # 爬升功率系数
        self.k3 = config.get('k3', 0.08)             # 悬停功率系数
        self.eta_motor = config.get('eta_motor', 0.8) # 电机效率
        self.E_capacity = config.get('E_capacity', 500.0)  # 电池容量 (Wh)
        self.eta_discharge = config.get('eta_discharge', 0.9)
        self.alpha_safe = config.get('alpha_safe', 0.15)
        self.SOC_threshold = config.get('SOC_threshold', 0.2)
        self.m_empty = config.get('m_empty', 10.0)    # 空机质量 (kg)

        self.E_usable = self.E_capacity * self.eta_discharge * (1 - self.alpha_safe)

    def compute_flight_power(self, velocity, mass):
        """计算飞行功率"""
        v_horizontal = np.sqrt(velocity[0]**2 + velocity[1]**2)
        v_z = velocity[2]

        P_forward = self.k1 * v_horizontal**3
        P_climb = self.k2 * mass * 9.81 * max(v_z, 0)

        P_flight = (P_forward + P_climb) / self.eta_motor
        return max(P_flight, 0)

    def compute_hover_power(self, mass):
        """计算悬停功率"""
        return self.k3 * mass * 9.81

    def compute_energy(self, power, dt):
        """计算能耗 (Wh)"""
        return power * dt / 3600.0

    def update_soc(self, soc_current, power, dt):
        """更新SOC"""
        energy = self.compute_energy(power, dt)
        soc_new = soc_current - energy / self.E_capacity
        return max(soc_new, 0)

    def check_return(self, soc_current):
        """检查是否需要返航"""
        return soc_current <= self.SOC_threshold

    def compute_total_mass(self, payload_mass):
        """计算总质量"""
        return self.m_empty + payload_mass
