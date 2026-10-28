"""
实际PPO训练脚本
项目：基于深度强化学习的多无人机协同物流配送路径规划
"""

import numpy as np
import os
import sys
import json
import time
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environment.airspace_env import AirspaceEnv
from algorithm.ppo_agent import PPOAgent
from config import ENV_CONFIG, PPO_CONFIG, TRAIN_CONFIG


def smooth_data(data, window=10):
    """平滑数据"""
    return np.convolve(data, np.ones(window)/window, mode='valid')


def run_training(num_episodes=1000, scenario='S1', save_dir='../../02_数据保存'):
    """运行PPO训练"""
    print("=" * 60)
    print(f"PPO训练开始 - 场景: {scenario}")
    print("=" * 60)

    # 根据场景调整配置
    config = ENV_CONFIG.copy()
    if scenario == 'S2':
        config['num_obstacles'] = 5
    elif scenario == 'S3':
        config['num_targets'] = 10
        config['num_obstacles'] = 10
    elif scenario == 'S4':
        config['num_uavs'] = 5
        config['num_targets'] = 15
        config['num_obstacles'] = 15
    elif scenario == 'S5':
        config['num_uavs'] = 6
        config['num_targets'] = 20
        config['num_obstacles'] = 20

    # 初始化环境
    env = AirspaceEnv(config)
    print(f"环境: {config['num_uavs']}架无人机, {config['num_targets']}个目标, {config['num_obstacles']}个障碍物")

    # 初始化PPO智能体
    state_dim = config['num_uavs'] * 11
    action_dim = config['num_uavs'] * 3
    ppo_config = {
        'state_dim': state_dim,
        'action_dim': action_dim,
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
        'action_limit': config['a_max'],
    }
    agent = PPOAgent(ppo_config)
    print(f"PPO: lr={ppo_config['lr']}, γ={ppo_config['gamma']}, ε={ppo_config['clip_epsilon']}")

    # 训练参数
    log_interval = 10
    save_interval = 100

    # 日志记录
    training_log = {
        'scenario': scenario,
        'config': config,
        'ppo_config': ppo_config,
        'episodes': [],
        'rewards': [],
        'delivered': [],
        'collisions': [],
        'actor_losses': [],
        'critic_losses': [],
        'start_time': datetime.now().isoformat()
    }

    reward_history = []
    delivery_history = []
    collision_history = []
    start_time = time.time()

    print(f"\n开始训练，共{num_episodes}回合...")
    print("-" * 60)

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_delivered = 0
        episode_collisions = 0
        done = False
        step_count = 0

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
            episode_delivered = info.get('delivered', 0)
            episode_collisions = info.get('collisions', 0)
            step_count += 1

        # 更新网络
        next_value = 0
        actor_loss, critic_loss = agent.update(next_value)

        # 记录
        reward_history.append(episode_reward)
        delivery_history.append(episode_delivered)
        collision_history.append(episode_collisions)

        training_log['episodes'].append(episode + 1)
        training_log['rewards'].append(float(episode_reward))
        training_log['delivered'].append(int(episode_delivered))
        training_log['collisions'].append(int(episode_collisions))
        training_log['actor_losses'].append(float(actor_loss) if actor_loss else 0)
        training_log['critic_losses'].append(float(critic_loss) if critic_loss else 0)

        # 日志输出
        if (episode + 1) % log_interval == 0:
            avg_reward = np.mean(reward_history[-log_interval:])
            avg_delivery = np.mean(delivery_history[-log_interval:])
            avg_collision = np.mean(collision_history[-log_interval:])
            elapsed = time.time() - start_time
            fps = (episode + 1) * step_count / elapsed

            print(f"Episode {episode+1}/{num_episodes} | "
                  f"Reward: {avg_reward:7.1f} | "
                  f"Delivered: {avg_delivery:.1f}/{config['num_targets']} | "
                  f"Collisions: {avg_collision:.1f} | "
                  f"FPS: {fps:.0f}")

        # 保存模型
        if (episode + 1) % save_interval == 0:
            model_dir = os.path.join(save_dir, '模型检查点')
            os.makedirs(model_dir, exist_ok=True)
            agent.save(f'{model_dir}/ppo_{scenario}_episode_{episode+1}.pth')
            print(f"  模型已保存: ppo_{scenario}_episode_{episode+1}.pth")

    # 保存最终模型
    model_dir = os.path.join(save_dir, '模型检查点')
    agent.save(f'{model_dir}/ppo_{scenario}_final.pth')

    # 保存训练日志
    training_log['end_time'] = datetime.now().isoformat()
    training_log['total_time'] = time.time() - start_time
    training_log['final_reward'] = float(np.mean(reward_history[-100:]))
    training_log['final_delivery'] = float(np.mean(delivery_history[-100:]))
    training_log['final_collision'] = float(np.mean(collision_history[-100:]))

    log_dir = os.path.join(save_dir, '训练日志')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'training_log_{scenario}.json')
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(training_log, f, indent=2, ensure_ascii=False)

    # 保存CSV格式
    csv_file = os.path.join(log_dir, f'training_rewards_{scenario}.csv')
    with open(csv_file, 'w') as f:
        f.write('episode,reward,delivered,collisions,actor_loss,critic_loss\n')
        for i in range(len(training_log['episodes'])):
            f.write(f"{training_log['episodes'][i]},"
                    f"{training_log['rewards'][i]},"
                    f"{training_log['delivered'][i]},"
                    f"{training_log['collisions'][i]},"
                    f"{training_log['actor_losses'][i]},"
                    f"{training_log['critic_losses'][i]}\n")

    print("-" * 60)
    print(f"训练完成！总耗时: {training_log['total_time']:.1f}秒")
    print(f"最终奖励: {training_log['final_reward']:.1f}")
    print(f"最终完成率: {training_log['final_delivery']:.1f}/{config['num_targets']}")
    print(f"最终碰撞率: {training_log['final_collision']:.1f}")
    print(f"训练日志: {log_file}")
    print(f"训练数据: {csv_file}")

    return training_log


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='PPO训练')
    parser.add_argument('--episodes', type=int, default=500, help='训练轮次')
    parser.add_argument('--scenario', type=str, default='S1', help='场景名称')
    parser.add_argument('--save_dir', type=str, default='../../02_数据保存', help='保存目录')
    args = parser.parse_args()

    run_training(args.episodes, args.scenario, args.save_dir)
