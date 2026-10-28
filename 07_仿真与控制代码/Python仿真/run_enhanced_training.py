"""
增强版PPO训练脚本
使用增强奖励函数，重点提升完成率
"""

import numpy as np
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environment.airspace_env import AirspaceEnv
from algorithm.ppo_agent import PPOAgent
from algorithm.enhanced_reward import EnhancedRewardFunction
from config import ENV_CONFIG, PPO_CONFIG


def run_enhanced_training():
    """运行增强训练"""
    print("=" * 60, flush=True)
    print("增强PPO训练 - 重点提升完成率", flush=True)
    print("=" * 60, flush=True)

    # 配置
    config = ENV_CONFIG.copy()
    config['num_uavs'] = 1  # 从单UAV开始
    config['num_targets'] = 1
    config['num_obstacles'] = 3
    config['max_steps'] = 300

    env = AirspaceEnv(config)
    state_dim = config['num_uavs'] * 11
    action_dim = config['num_uavs'] * 3

    # 增强奖励函数
    reward_func = EnhancedRewardFunction({
        'k_progress': 50.0,
        'k_direction': 20.0,
        'k_approach': 100.0,
        'approach_radius': 50.0,
        'R_arrival': 500.0,
        'R_delivery': 1000.0,
        'R_collision': -30.0,
        'k_time': 0.1,
        'k_obstacle': 10.0,
        'd_safe': 10.0,
    })

    # PPO配置
    ppo_config = {
        'state_dim': state_dim,
        'action_dim': action_dim,
        'hidden_dim': 256,
        'lr': 1e-4,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_epsilon': 0.2,
        'entropy_coef': 0.02,
        'value_coef': 0.5,
        'max_grad_norm': 0.5,
        'batch_size': 512,
        'update_epochs': 4,
        'action_limit': config['a_max'],
    }

    agent = PPOAgent(ppo_config)

    print(f"配置: {config['num_uavs']}UAV, {config['num_targets']}目标, {config['num_obstacles']}障碍物", flush=True)
    print(f"每轮最大步数: {config['max_steps']}", flush=True)
    print(f"训练轮次: 1000", flush=True)
    print("-" * 60, flush=True)

    # 训练数据
    rewards = []
    delivered = []
    collisions = []
    start_time = time.time()

    # 训练循环
    for episode in range(1000):
        state = env.reset()
        reward_func.reset()

        episode_reward = 0
        episode_delivered = 0
        episode_collisions = 0
        done = False
        steps = 0

        while not done and steps < config['max_steps']:
            action, log_prob, value = agent.select_action(state)
            actions = action.reshape(config['num_uavs'], 3)
            next_state, env_reward, done, info = env.step(actions)

            # 使用增强奖励函数
            enhanced_reward = 0.0
            for uav_id in range(config['num_uavs']):
                uav_pos = info.get('positions', [[0, 0, 0]])[uav_id] if 'positions' in info else [0, 0, 0]
                uav_vel = info.get('velocities', [[0, 0, 0]])[uav_id] if 'velocities' in info else [0, 0, 0]
                target_pos = info.get('target_positions', [[50, 50, 10]])[uav_id % len(info.get('target_positions', [[50, 50, 10]]))]

                obstacles = info.get('obstacles', [])
                if not obstacles:
                    obstacles = [{'center': [50, 50, 25], 'radius': 5.0}]

                # 获取配送状态
                delivered_status = info.get('delivered', 0)
                try:
                    if hasattr(delivered_status, '__len__'):
                        is_delivered = delivered_status[uav_id] if uav_id < len(delivered_status) else False
                    else:
                        is_delivered = float(delivered_status) > 0
                except:
                    is_delivered = False

                collisions_info = {
                    'obstacle_collision': False,
                    'uav_collision': False,
                }

                uav_reward, _ = reward_func.compute(
                    uav_id=uav_id,
                    uav_pos=uav_pos,
                    uav_vel=uav_vel,
                    target_pos=target_pos,
                    obstacles=obstacles,
                    delivered=is_delivered,
                    collisions=collisions_info,
                    step=steps
                )
                enhanced_reward += uav_reward

            # 存储经验
            agent.store_transition(state, action, enhanced_reward, done, log_prob, value)

            state = next_state
            episode_reward += enhanced_reward
            episode_delivered = info.get('delivered', 0)
            episode_collisions = info.get('collisions', 0)
            steps += 1

        # 更新
        next_value = 0
        actor_loss, critic_loss = agent.update(next_value)

        # 记录
        rewards.append(episode_reward)
        delivered.append(episode_delivered)
        collisions.append(episode_collisions)

        # 输出
        if (episode + 1) % 50 == 0:
            avg_r = np.mean(rewards[-50:])
            avg_d = np.mean(delivered[-50:])
            avg_c = np.mean(collisions[-50:])
            elapsed = time.time() - start_time
            fps = (episode + 1) * steps / elapsed
            eta = (1000 - episode - 1) * elapsed / (episode + 1) / 60

            print(f"Ep {episode+1:4d}/1000 | "
                  f"R: {avg_r:8.1f} | "
                  f"D: {avg_d:.1f}/{config['num_targets']} | "
                  f"C: {avg_c:.1f} | "
                  f"FPS: {fps:.0f} | "
                  f"ETA: {eta:.1f}min", flush=True)

        # 保存模型
        if (episode + 1) % 200 == 0:
            model_dir = '../../02_数据保存/模型检查点'
            os.makedirs(model_dir, exist_ok=True)
            agent.save(f'{model_dir}/ppo_enhanced_ep{episode+1}.pth')
            print(f"  模型已保存", flush=True)

            # 保存数据
            log_dir = '../../02_数据保存/训练日志'
            os.makedirs(log_dir, exist_ok=True)
            with open(os.path.join(log_dir, 'training_enhanced.csv'), 'w') as f:
                f.write('episode,reward,delivered,collisions\n')
                for i in range(len(rewards)):
                    f.write(f"{i+1},{rewards[i]},{delivered[i]},{collisions[i]}\n")

    # 保存最终结果
    model_dir = '../../02_数据保存/模型检查点'
    agent.save(f'{model_dir}/ppo_enhanced_final.pth')

    log_dir = '../../02_数据保存/训练日志'
    with open(os.path.join(log_dir, 'training_enhanced.csv'), 'w') as f:
        f.write('episode,reward,delivered,collisions\n')
        for i in range(len(rewards)):
            f.write(f"{i+1},{rewards[i]},{delivered[i]},{collisions[i]}\n")

    total_time = time.time() - start_time
    print("-" * 60, flush=True)
    print(f"增强训练完成!", flush=True)
    print(f"总耗时: {total_time/60:.1f}分钟", flush=True)
    print(f"最终奖励: {np.mean(rewards[-100:]):.1f}", flush=True)
    print(f"最终完成率: {np.mean(delivered[-100:]):.1f}/{config['num_targets']}", flush=True)


if __name__ == '__main__':
    run_enhanced_training()
