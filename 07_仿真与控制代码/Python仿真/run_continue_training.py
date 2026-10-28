"""
继续训练脚本
从阶段3开始，继续改进PPO训练
"""

import numpy as np
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environment.airspace_env import AirspaceEnv
from algorithm.ppo_agent import PPOAgent
from config import ENV_CONFIG, PPO_CONFIG


def run_continue_training():
    """继续训练"""
    print("=" * 60, flush=True)
    print("继续改进PPO训练 - 阶段3&4", flush=True)
    print("=" * 60, flush=True)

    # 阶段3配置
    config = ENV_CONFIG.copy()
    config['num_uavs'] = 3
    config['num_targets'] = 1
    config['num_obstacles'] = 5
    config['max_steps'] = 400

    env = AirspaceEnv(config)
    state_dim = config['num_uavs'] * 11
    action_dim = config['num_uavs'] * 3

    # 加载阶段2的模型继续训练
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

    # 尝试加载阶段2的模型
    model_path = '../../02_数据保存/模型检查点/ppo_improved_stage2.pth'
    if os.path.exists(model_path):
        try:
            agent.load(model_path)
            print(f"已加载阶段2模型: {model_path}", flush=True)
        except:
            print("模型加载失败，从头开始训练", flush=True)

    print(f"配置: {config['num_uavs']}UAV, {config['num_targets']}目标, {config['num_obstacles']}障碍物", flush=True)
    print(f"每轮最大步数: {config['max_steps']}", flush=True)
    print(f"训练轮次: 500", flush=True)
    print("-" * 60, flush=True)

    # 训练数据
    rewards = []
    delivered = []
    collisions = []
    start_time = time.time()

    # 训练循环
    for episode in range(500):
        state = env.reset()
        episode_reward = 0
        episode_delivered = 0
        episode_collisions = 0
        done = False
        steps = 0

        while not done and steps < config['max_steps']:
            action, log_prob, value = agent.select_action(state)
            actions = action.reshape(config['num_uavs'], 3)
            next_state, reward, done, info = env.step(actions)
            agent.store_transition(state, action, reward, done, log_prob, value)
            state = next_state
            episode_reward += reward
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
        if (episode + 1) % 20 == 0:
            avg_r = np.mean(rewards[-20:])
            avg_d = np.mean(delivered[-20:])
            avg_c = np.mean(collisions[-20:])
            elapsed = time.time() - start_time
            fps = (episode + 1) * steps / elapsed
            eta = (500 - episode - 1) * elapsed / (episode + 1) / 60

            print(f"Ep {episode+1:3d}/500 | "
                  f"R: {avg_r:8.1f} | "
                  f"D: {avg_d:.1f}/{config['num_targets']} | "
                  f"C: {avg_c:.1f} | "
                  f"FPS: {fps:.0f} | "
                  f"ETA: {eta:.1f}min", flush=True)

        # 保存模型
        if (episode + 1) % 100 == 0:
            model_dir = '../../02_数据保存/模型检查点'
            os.makedirs(model_dir, exist_ok=True)
            agent.save(f'{model_dir}/ppo_improved_stage3_ep{episode+1}.pth')
            print(f"  模型已保存: ppo_improved_stage3_ep{episode+1}.pth", flush=True)

    # 保存最终模型和数据
    model_dir = '../../02_数据保存/模型检查点'
    agent.save(f'{model_dir}/ppo_improved_stage3_final.pth')

    log_dir = '../../02_数据保存/训练日志'
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, 'training_improved_stage3.csv'), 'w') as f:
        f.write('episode,reward,delivered,collisions\n')
        for i in range(len(rewards)):
            f.write(f"{i+1},{rewards[i]},{delivered[i]},{collisions[i]}\n")

    total_time = time.time() - start_time
    print("-" * 60, flush=True)
    print(f"阶段3训练完成!", flush=True)
    print(f"总耗时: {total_time/60:.1f}分钟", flush=True)
    print(f"最终奖励: {np.mean(rewards[-50:]):.1f}", flush=True)
    print(f"最终完成率: {np.mean(delivered[-50:]):.1f}/{config['num_targets']}", flush=True)

    # ==================== 阶段4 ====================
    print("\n" + "=" * 60, flush=True)
    print("阶段4: 完整任务 (3UAV, 5目标, 5障碍物)", flush=True)
    print("=" * 60, flush=True)

    config4 = ENV_CONFIG.copy()
    config4['num_uavs'] = 3
    config4['num_targets'] = 5
    config4['num_obstacles'] = 5
    config4['max_steps'] = 500

    env4 = AirspaceEnv(config4)
    state_dim4 = config4['num_uavs'] * 11
    action_dim4 = config4['num_uavs'] * 3

    ppo_config4 = ppo_config.copy()
    ppo_config4['state_dim'] = state_dim4
    ppo_config4['action_dim'] = action_dim4
    ppo_config4['action_limit'] = config4['a_max']

    agent4 = PPOAgent(ppo_config4)

    # 加载阶段3模型
    try:
        agent4.load(f'{model_dir}/ppo_improved_stage3_final.pth')
        print("已加载阶段3模型", flush=True)
    except:
        print("模型加载失败，从头开始", flush=True)

    print(f"配置: {config4['num_uavs']}UAV, {config4['num_targets']}目标, {config4['num_obstacles']}障碍物", flush=True)
    print("-" * 60, flush=True)

    rewards4 = []
    delivered4 = []
    collisions4 = []
    start_time4 = time.time()

    for episode in range(600):
        state = env4.reset()
        episode_reward = 0
        episode_delivered = 0
        episode_collisions = 0
        done = False
        steps = 0

        while not done and steps < config4['max_steps']:
            action, log_prob, value = agent4.select_action(state)
            actions = action.reshape(config4['num_uavs'], 3)
            next_state, reward, done, info = env4.step(actions)
            agent4.store_transition(state, action, reward, done, log_prob, value)
            state = next_state
            episode_reward += reward
            episode_delivered = info.get('delivered', 0)
            episode_collisions = info.get('collisions', 0)
            steps += 1

        next_value = 0
        actor_loss, critic_loss = agent4.update(next_value)

        rewards4.append(episode_reward)
        delivered4.append(episode_delivered)
        collisions4.append(episode_collisions)

        if (episode + 1) % 20 == 0:
            avg_r = np.mean(rewards4[-20:])
            avg_d = np.mean(delivered4[-20:])
            avg_c = np.mean(collisions4[-20:])
            elapsed = time.time() - start_time4
            fps = (episode + 1) * steps / elapsed
            eta = (600 - episode - 1) * elapsed / (episode + 1) / 60

            print(f"Ep {episode+1:3d}/600 | "
                  f"R: {avg_r:8.1f} | "
                  f"D: {avg_d:.1f}/{config4['num_targets']} | "
                  f"C: {avg_c:.1f} | "
                  f"FPS: {fps:.0f} | "
                  f"ETA: {eta:.1f}min", flush=True)

        if (episode + 1) % 100 == 0:
            agent4.save(f'{model_dir}/ppo_improved_stage4_ep{episode+1}.pth')
            print(f"  模型已保存", flush=True)

    agent4.save(f'{model_dir}/ppo_improved_stage4_final.pth')

    with open(os.path.join(log_dir, 'training_improved_stage4.csv'), 'w') as f:
        f.write('episode,reward,delivered,collisions\n')
        for i in range(len(rewards4)):
            f.write(f"{i+1},{rewards4[i]},{delivered4[i]},{collisions4[i]}\n")

    total_time4 = time.time() - start_time4
    print("-" * 60, flush=True)
    print(f"阶段4训练完成!", flush=True)
    print(f"总耗时: {total_time4/60:.1f}分钟", flush=True)
    print(f"最终奖励: {np.mean(rewards4[-50:]):.1f}", flush=True)
    print(f"最终完成率: {np.mean(delivered4[-50:]):.1f}/{config4['num_targets']}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("全部训练完成!", flush=True)
    print("=" * 60, flush=True)


if __name__ == '__main__':
    run_continue_training()
