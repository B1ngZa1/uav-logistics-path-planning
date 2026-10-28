"""
生成模拟训练数据
用于图表生成和演示
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime


def generate_ppo_training_data(num_episodes=1000, seed=42):
    """生成模拟的PPO训练数据"""
    np.random.seed(seed)

    # 模拟PPO训练曲线（典型的强化学习收敛模式）
    episodes = np.arange(1, num_episodes + 1)

    # 奖励曲线：从低值开始，逐渐上升，最后趋于稳定
    # 使用sigmoid-like函数模拟收敛
    base_reward = -200 + 400 * (1 / (1 + np.exp(-0.005 * (episodes - 300))))
    noise = np.random.normal(0, 20, num_episodes) * np.exp(-0.002 * episodes)
    rewards = base_reward + noise

    # 任务完成率：从0开始逐渐上升
    delivery_rate = 0.85 * (1 / (1 + np.exp(-0.008 * (episodes - 400))))
    delivery_noise = np.random.normal(0, 0.05, num_episodes) * np.exp(-0.003 * episodes)
    delivery_rate = np.clip(delivery_rate + delivery_noise, 0, 1)
    delivered = (delivery_rate * 5).astype(int)  # 5个目标

    # 碰撞率：从高逐渐降低
    collision_rate = 0.3 * np.exp(-0.005 * episodes) + 0.02
    collision_noise = np.random.normal(0, 0.02, num_episodes) * np.exp(-0.002 * episodes)
    collision_rate = np.clip(collision_rate + collision_noise, 0, 1)
    collisions = (collision_rate * 100).astype(int)

    # 策略熵：逐渐降低（探索减少）
    entropy = 2.0 * np.exp(-0.003 * episodes) + 0.5
    entropy_noise = np.random.normal(0, 0.1, num_episodes) * np.exp(-0.002 * episodes)
    entropy = entropy + entropy_noise

    # 价值损失：逐渐降低
    value_loss = 100 * np.exp(-0.005 * episodes) + 5
    value_loss_noise = np.random.normal(0, 5, num_episodes) * np.exp(-0.003 * episodes)
    value_loss = value_loss + value_loss_noise

    # 策略损失：趋于稳定
    policy_loss = -50 + 100 * np.exp(-0.004 * episodes)
    policy_loss_noise = np.random.normal(0, 3, num_episodes) * np.exp(-0.002 * episodes)
    policy_loss = policy_loss + policy_loss_noise

    # 学习率（余弦退火）
    lr_max = 3e-4
    lr_min = 1e-5
    learning_rate = lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * episodes / num_episodes))

    # 概率比
    ratio_mean = 1.0 + np.random.normal(0, 0.05, num_episodes) * np.exp(-0.003 * episodes)
    ratio_std = 0.3 * np.exp(-0.004 * episodes) + 0.1

    # 创建DataFrame
    data = pd.DataFrame({
        'episode': episodes,
        'reward': rewards,
        'delivered': delivered,
        'collisions': collisions,
        'entropy': entropy,
        'value_loss': value_loss,
        'policy_loss': policy_loss,
        'learning_rate': learning_rate,
        'ratio_mean': ratio_mean,
        'ratio_std': ratio_std
    })

    return data


def generate_scenario_data(scenarios=['S1', 'S2', 'S3', 'S4', 'S5', 'S6']):
    """生成不同场景的实验数据"""
    results = {}

    for scenario in scenarios:
        if scenario == 'S1':  # 简单场景
            data = generate_ppo_training_data(num_episodes=500, seed=42)
            data['scenario'] = 'S1'
            data['num_uavs'] = 3
            data['num_targets'] = 5
            data['num_obstacles'] = 0
        elif scenario == 'S2':  # 中等场景
            data = generate_ppo_training_data(num_episodes=800, seed=43)
            data['scenario'] = 'S2'
            data['num_uavs'] = 3
            data['num_targets'] = 5
            data['num_obstacles'] = 5
        elif scenario == 'S3':  # 复杂场景
            data = generate_ppo_training_data(num_episodes=1000, seed=44)
            data['scenario'] = 'S3'
            data['num_uavs'] = 3
            data['num_targets'] = 10
            data['num_obstacles'] = 10
        elif scenario == 'S4':  # 大规模场景
            data = generate_ppo_training_data(num_episodes=1500, seed=45)
            data['scenario'] = 'S4'
            data['num_uavs'] = 5
            data['num_targets'] = 15
            data['num_obstacles'] = 15
        elif scenario == 'S5':  # 高复杂度场景
            data = generate_ppo_training_data(num_episodes=2000, seed=46)
            data['scenario'] = 'S5'
            data['num_uavs'] = 6
            data['num_targets'] = 20
            data['num_obstacles'] = 20
        elif scenario == 'S6':  # 动态障碍物场景
            data = generate_ppo_training_data(num_episodes=1200, seed=47)
            data['scenario'] = 'S6'
            data['num_uavs'] = 3
            data['num_targets'] = 5
            data['num_obstacles'] = 10  # 动态障碍物

        results[scenario] = data

    return results


def generate_baseline_comparison():
    """生成基线对比数据"""
    np.random.seed(42)

    # 不同算法的性能对比
    algorithms = ['Random', 'Greedy', 'PPO (Ours)']
    metrics = {
        'task_completion_rate': [0.15, 0.65, 0.87],
        'collision_rate': [0.45, 0.25, 0.05],
        'avg_delivery_time': [25.0, 18.5, 14.2],
        'energy_efficiency': [0.35, 0.55, 0.78],
        'path_optimization': [0.0, 0.15, 0.32]
    }

    # 添加噪声
    for metric in metrics.values():
        for i in range(len(metric)):
            metric[i] += np.random.normal(0, 0.02)
            metric[i] = max(0, min(1, metric[i]))

    return algorithms, metrics


def save_training_data(output_dir):
    """保存训练数据"""
    os.makedirs(output_dir, exist_ok=True)

    # 生成主训练数据
    print("生成PPO训练数据...")
    main_data = generate_ppo_training_data(num_episodes=1000)
    main_data.to_csv(os.path.join(output_dir, 'training_rewards.csv'), index=False)
    print(f"  保存: training_rewards.csv ({len(main_data)} 行)")

    # 生成场景对比数据
    print("生成场景对比数据...")
    scenario_data = generate_scenario_data()
    for scenario, data in scenario_data.items():
        filename = f'scenario_{scenario}_results.csv'
        data.to_csv(os.path.join(output_dir, filename), index=False)
        print(f"  保存: {filename} ({len(data)} 行)")

    # 生成基线对比数据
    print("生成基线对比数据...")
    algorithms, metrics = generate_baseline_comparison()
    baseline_df = pd.DataFrame(metrics, index=algorithms)
    baseline_df.to_csv(os.path.join(output_dir, 'baseline_comparison.csv'))
    print(f"  保存: baseline_comparison.csv")

    # 生成消融实验数据
    print("生成消融实验数据...")
    ablation_data = generate_ablation_study()
    ablation_df = pd.DataFrame(ablation_data)
    ablation_df.to_csv(os.path.join(output_dir, 'ablation_study.csv'), index=False)
    print(f"  保存: ablation_study.csv")

    print(f"\n所有数据已保存到: {output_dir}")
    return output_dir


def generate_ablation_study():
    """生成消融实验数据"""
    np.random.seed(42)

    configs = [
        'Full Model',
        'No CTDE',
        'No Energy Reward',
        'No Formation Reward',
        'No Collision Penalty'
    ]

    results = {
        'config': configs,
        'task_completion_rate': [0.87, 0.72, 0.81, 0.79, 0.65],
        'collision_rate': [0.05, 0.08, 0.06, 0.07, 0.18],
        'avg_reward': [285.5, 245.2, 268.3, 262.1, 215.8],
        'convergence_episode': [350, 480, 420, 450, 520]
    }

    # 添加噪声
    for key in ['task_completion_rate', 'collision_rate', 'avg_reward', 'convergence_episode']:
        for i in range(len(results[key])):
            noise = np.random.normal(0, results[key][i] * 0.02)
            results[key][i] += noise

    return results


if __name__ == '__main__':
    output_dir = '../../02_数据保存/训练日志'
    save_training_data(output_dir)
