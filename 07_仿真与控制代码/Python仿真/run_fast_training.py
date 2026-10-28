"""
快速PPO训练脚本（优化版）
"""

import numpy as np
import os
import sys
import json
import time
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environment.airspace_env import AirspaceEnv
from algorithm.ppo_agent import PPOAgent
from config import ENV_CONFIG, PPO_CONFIG


def run_fast_training(num_episodes=200, max_steps=200, scenario='S1'):
    """快速训练"""
    print("=" * 50, flush=True)
    print(f"快速PPO训练 - 场景: {scenario}", flush=True)
    print("=" * 50, flush=True)

    # 简化配置
    config = ENV_CONFIG.copy()
    config['max_steps'] = max_steps
    config['num_obstacles'] = 3  # 减少障碍物

    env = AirspaceEnv(config)
    state_dim = config['num_uavs'] * 11
    action_dim = config['num_uavs'] * 3

    ppo_config = {
        'state_dim': state_dim,
        'action_dim': action_dim,
        'hidden_dim': 128,  # 减小网络
        'lr': PPO_CONFIG['learning_rate'],
        'gamma': PPO_CONFIG['gamma'],
        'gae_lambda': PPO_CONFIG['gae_lambda'],
        'clip_epsilon': PPO_CONFIG['clip_epsilon'],
        'entropy_coef': PPO_CONFIG['entropy_coef'],
        'value_coef': PPO_CONFIG['value_coef'],
        'max_grad_norm': PPO_CONFIG['max_grad_norm'],
        'update_epochs': PPO_CONFIG['update_epochs'],
        'batch_size': 128,  # 减小批次
        'action_limit': config['a_max'],
    }
    agent = PPOAgent(ppo_config)

    # 训练数据
    all_rewards = []
    all_delivered = []
    all_collisions = []
    start_time = time.time()

    print(f"配置: {config['num_uavs']}UAV, {config['num_targets']}目标, {config['num_obstacles']}障碍物", flush=True)
    print(f"每轮最大步数: {max_steps}", flush=True)
    print("-" * 50, flush=True)

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        delivered = 0
        collisions = 0
        done = False
        steps = 0

        while not done and steps < max_steps:
            action, log_prob, value = agent.select_action(state)
            actions = action.reshape(env.num_uavs, 3)
            next_state, reward, done, info = env.step(actions)
            agent.store_transition(state, action, reward, done, log_prob, value)
            state = next_state
            episode_reward += reward
            delivered = info.get('delivered', 0)
            collisions = info.get('collisions', 0)
            steps += 1

        # 更新
        next_value = 0
        actor_loss, critic_loss = agent.update(next_value)

        all_rewards.append(episode_reward)
        all_delivered.append(delivered)
        all_collisions.append(collisions)

        # 输出
        if (episode + 1) % 10 == 0:
            avg_r = np.mean(all_rewards[-10:])
            avg_d = np.mean(all_delivered[-10:])
            avg_c = np.mean(all_collisions[-10:])
            elapsed = time.time() - start_time
            fps = (episode + 1) * steps / elapsed

            print(f"Ep {episode+1:3d}/{num_episodes} | "
                  f"R: {avg_r:8.1f} | "
                  f"D: {avg_d:.1f}/{config['num_targets']} | "
                  f"C: {avg_c:.1f} | "
                  f"FPS: {fps:.0f}", flush=True)

            # 保存中间结果
            save_results(all_rewards, all_delivered, all_collisions, scenario)

    # 保存最终结果
    save_results(all_rewards, all_delivered, all_collisions, scenario)

    # 保存模型
    model_dir = '../../02_数据保存/模型检查点'
    os.makedirs(model_dir, exist_ok=True)
    agent.save(f'{model_dir}/ppo_{scenario}_fast_final.pth')

    total_time = time.time() - start_time
    print("-" * 50, flush=True)
    print(f"训练完成! 耗时: {total_time:.1f}秒", flush=True)
    print(f"最终奖励: {np.mean(all_rewards[-20:]):.1f}", flush=True)
    print(f"最终完成率: {np.mean(all_delivered[-20:]):.1f}/{config['num_targets']}", flush=True)

    return all_rewards, all_delivered, all_collisions


def save_results(rewards, delivered, collisions, scenario):
    """保存训练结果"""
    log_dir = '../../02_数据保存/训练日志'
    os.makedirs(log_dir, exist_ok=True)

    csv_file = os.path.join(log_dir, f'training_rewards_{scenario}_fast.csv')
    with open(csv_file, 'w') as f:
        f.write('episode,reward,delivered,collisions\n')
        for i in range(len(rewards)):
            f.write(f"{i+1},{rewards[i]},{delivered[i]},{collisions[i]}\n")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--max_steps', type=int, default=200)
    parser.add_argument('--scenario', type=str, default='S1')
    args = parser.parse_args()

    run_fast_training(args.episodes, args.max_steps, args.scenario)
