"""
长时训练脚本 - 增加学习步数
项目：基于深度强化学习的多无人机协同物流配送路径规划
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


def run_long_training(num_episodes=1000, max_steps=500, scenario='S1_long'):
    """长时训练"""
    print("=" * 60, flush=True)
    print(f"长时PPO训练 - 场景: {scenario}", flush=True)
    print("=" * 60, flush=True)

    # 配置
    config = ENV_CONFIG.copy()
    config['max_steps'] = max_steps
    config['num_obstacles'] = 5

    env = AirspaceEnv(config)
    state_dim = config['num_uavs'] * 11
    action_dim = config['num_uavs'] * 3

    ppo_config = {
        'state_dim': state_dim,
        'action_dim': action_dim,
        'hidden_dim': 256,  # 增大网络
        'lr': PPO_CONFIG['learning_rate'],
        'gamma': PPO_CONFIG['gamma'],
        'gae_lambda': PPO_CONFIG['gae_lambda'],
        'clip_epsilon': PPO_CONFIG['clip_epsilon'],
        'entropy_coef': PPO_CONFIG['entropy_coef'],
        'value_coef': PPO_CONFIG['value_coef'],
        'max_grad_norm': PPO_CONFIG['max_grad_norm'],
        'update_epochs': PPO_CONFIG['update_epochs'],
        'batch_size': PPO_CONFIG['batch_size'],
        'action_limit': config['a_max'],
    }
    agent = PPOAgent(ppo_config)

    # 训练数据
    all_rewards = []
    all_delivered = []
    all_collisions = []
    all_entropy = []
    all_value_loss = []
    all_policy_loss = []
    all_learning_rate = []
    start_time = time.time()

    print(f"配置: {config['num_uavs']}UAV, {config['num_targets']}目标, {config['num_obstacles']}障碍物", flush=True)
    print(f"每轮最大步数: {max_steps}", flush=True)
    print(f"总训练轮次: {num_episodes}", flush=True)
    print(f"预计总步数: {num_episodes * max_steps}", flush=True)
    print("-" * 60, flush=True)

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        delivered = 0
        collisions = 0
        done = False
        steps = 0
        episode_entropy = []
        episode_value_loss = []
        episode_policy_loss = []

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

        # 记录
        all_rewards.append(episode_reward)
        all_delivered.append(delivered)
        all_collisions.append(collisions)
        all_entropy.append(0)  # 简化
        all_value_loss.append(critic_loss if critic_loss else 0)
        all_policy_loss.append(actor_loss if actor_loss else 0)
        all_learning_rate.append(ppo_config['lr'])

        # 输出
        if (episode + 1) % 50 == 0:
            avg_r = np.mean(all_rewards[-50:])
            avg_d = np.mean(all_delivered[-50:])
            avg_c = np.mean(all_collisions[-50:])
            elapsed = time.time() - start_time
            fps = (episode + 1) * steps / elapsed
            eta = (num_episodes - episode - 1) * elapsed / (episode + 1) / 60

            print(f"Ep {episode+1:4d}/{num_episodes} | "
                  f"R: {avg_r:8.1f} | "
                  f"D: {avg_d:.1f}/{config['num_targets']} | "
                  f"C: {avg_c:.1f} | "
                  f"FPS: {fps:.0f} | "
                  f"ETA: {eta:.1f}min", flush=True)

            # 保存中间结果
            save_results(all_rewards, all_delivered, all_collisions,
                        all_entropy, all_value_loss, all_policy_loss,
                        all_learning_rate, scenario)

            # 保存模型
            if (episode + 1) % 200 == 0:
                model_dir = '../../02_数据保存/模型检查点'
                os.makedirs(model_dir, exist_ok=True)
                agent.save(f'{model_dir}/ppo_{scenario}_ep{episode+1}.pth')
                print(f"  模型已保存: ppo_{scenario}_ep{episode+1}.pth", flush=True)

    # 保存最终结果
    save_results(all_rewards, all_delivered, all_collisions,
                all_entropy, all_value_loss, all_policy_loss,
                all_learning_rate, scenario)

    # 保存最终模型
    model_dir = '../../02_数据保存/模型检查点'
    os.makedirs(model_dir, exist_ok=True)
    agent.save(f'{model_dir}/ppo_{scenario}_final.pth')

    total_time = time.time() - start_time
    print("-" * 60, flush=True)
    print(f"训练完成! 总耗时: {total_time/60:.1f}分钟", flush=True)
    print(f"总训练步数: {num_episodes * max_steps}", flush=True)
    print(f"最终奖励: {np.mean(all_rewards[-100:]):.1f}", flush=True)
    print(f"最终完成率: {np.mean(all_delivered[-100:]):.1f}/{config['num_targets']}", flush=True)

    return all_rewards, all_delivered, all_collisions


def save_results(rewards, delivered, collisions, entropy, value_loss, policy_loss, lr, scenario):
    """保存训练结果"""
    log_dir = '../../02_数据保存/训练日志'
    os.makedirs(log_dir, exist_ok=True)

    csv_file = os.path.join(log_dir, f'training_rewards_{scenario}.csv')
    with open(csv_file, 'w') as f:
        f.write('episode,reward,delivered,collisions,entropy,value_loss,policy_loss,learning_rate\n')
        for i in range(len(rewards)):
            f.write(f"{i+1},{rewards[i]},{delivered[i]},{collisions[i]},"
                    f"{entropy[i]},{value_loss[i]},{policy_loss[i]},{lr[i]}\n")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--max_steps', type=int, default=500)
    parser.add_argument('--scenario', type=str, default='S1_long')
    args = parser.parse_args()

    run_long_training(args.episodes, args.max_steps, args.scenario)
