"""
主训练脚本
项目：基于深度强化学习的多无人机协同物流配送路径规划
"""

import numpy as np
import os
import sys
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environment.airspace_env import AirspaceEnv
from algorithm.ppo_agent import PPOAgent
from config import ENV_CONFIG, PPO_CONFIG, TRAIN_CONFIG


def train():
    """主训练函数"""
    print("=" * 60)
    print("基于深度强化学习的多无人机协同物流配送路径规划")
    print("=" * 60)

    # 初始化环境
    env = AirspaceEnv(ENV_CONFIG)
    print(f"环境初始化完成：{ENV_CONFIG['num_uavs']}架无人机，{ENV_CONFIG['num_targets']}个目标")

    # 初始化PPO智能体（全局状态 = num_uavs × 11维）
    state_dim = ENV_CONFIG['num_uavs'] * 11
    ppo_config = {
        'state_dim': state_dim,
        'action_dim': ENV_CONFIG['num_uavs'] * 3,
        'hidden_dim': PPO_CONFIG['hidden_dim'],
        'lr': PPO_CONFIG['learning_rate'],
        'gamma': PPO_CONFIG['gamma'],
        'gae_lambda': PPO_CONFIG['gae_lambda'],
        'clip_epsilon': PPO_CONFIG['clip_epsilon'],
        'entropy_coef': PPO_CONFIG['entropy_coef'],
        'value_coef': PPO_CONFIG['value_coef'],
        'max_grad_norm': PPO_CONFIG['max_grad_norm'],
        'update_epochs': PPO_CONFIG['update_epochs'],
        'batch_size': PPO_CONFIG['batch_size'],
        'action_limit': ENV_CONFIG['a_max'],
    }
    agent = PPOAgent(ppo_config)
    print(f"PPO智能体初始化完成：lr={ppo_config['lr']}, γ={ppo_config['gamma']}")

    # 训练参数
    num_episodes = TRAIN_CONFIG['num_episodes']
    eval_interval = TRAIN_CONFIG['eval_interval']
    save_interval = TRAIN_CONFIG['save_interval']
    log_interval = TRAIN_CONFIG['log_interval']

    # 日志
    reward_history = []
    delivery_history = []
    collision_history = []

    print(f"\n开始训练，共{num_episodes}回合...")
    print("-" * 60)

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_delivered = 0
        episode_collisions = 0
        done = False

        while not done:
            # 选择动作
            action, log_prob, value = agent.select_action(state)

            # 执行动作
            actions = action.reshape(env.num_uavs, 3)
            next_state, reward, done, info = env.step(actions)

            # 存储经验
            agent.store_transition(state, action, reward, done, log_prob, value)

            state = next_state
            episode_reward += reward
            episode_delivered = info['delivered']
            episode_collisions = info['collisions']

        # 更新网络
        next_value = 0  # 终止状态价值为0
        actor_loss, critic_loss = agent.update(next_value)

        # 记录
        reward_history.append(episode_reward)
        delivery_history.append(episode_delivered)
        collision_history.append(episode_collisions)

        # 日志输出
        if (episode + 1) % log_interval == 0:
            avg_reward = np.mean(reward_history[-log_interval:])
            avg_delivery = np.mean(delivery_history[-log_interval:])
            avg_collision = np.mean(collision_history[-log_interval:])
            print(f"Episode {episode+1}/{num_episodes} | "
                  f"Reward: {avg_reward:.1f} | "
                  f"Delivered: {avg_delivery:.1f}/{env.num_targets} | "
                  f"Collisions: {avg_collision:.1f}")

        # 保存模型
        if (episode + 1) % save_interval == 0:
            save_dir = '../../02_数据保存/模型检查点'
            os.makedirs(save_dir, exist_ok=True)
            agent.save(f'{save_dir}/ppo_episode_{episode+1}.pth')
            print(f"模型已保存：ppo_episode_{episode+1}.pth")

    print("-" * 60)
    print("训练完成！")

    # 保存最终模型
    save_dir = '../../02_数据保存/模型检查点'
    agent.save(f'{save_dir}/ppo_final.pth')

    return reward_history, delivery_history, collision_history


if __name__ == '__main__':
    train()
