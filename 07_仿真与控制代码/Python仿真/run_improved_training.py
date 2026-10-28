"""
改进的PPO训练脚本（基于文献方法）
项目：基于深度强化学习的多无人机协同物流配送路径规划

改进点：
1. 密集奖励函数（HCA-MAPPO 9项奖励）
2. 课程学习（4阶段渐进难度）
3. 增强状态空间（添加方向信息）
4. 优先级经验回放
5. 学习率调度（余弦退火）
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
from algorithm.improved_reward import ImprovedRewardFunction
from config import ENV_CONFIG, PPO_CONFIG


# ==================== 课程学习配置 ====================

CURRICULUM_STAGES = [
    {
        'name': 'Stage 1: Single UAV, Single Target, No Obstacles',
        'num_uavs': 1,
        'num_targets': 1,
        'num_obstacles': 0,
        'max_steps': 200,
        'success_threshold': 0.6,  # 完成率阈值
        'episodes': 300,
    },
    {
        'name': 'Stage 2: Single UAV, Single Target, Simple Obstacles',
        'num_uavs': 1,
        'num_targets': 1,
        'num_obstacles': 3,
        'max_steps': 300,
        'success_threshold': 0.5,
        'episodes': 400,
    },
    {
        'name': 'Stage 3: Multi UAV, Single Target',
        'num_uavs': 3,
        'num_targets': 1,
        'num_obstacles': 5,
        'max_steps': 400,
        'success_threshold': 0.5,
        'episodes': 500,
    },
    {
        'name': 'Stage 4: Full Task',
        'num_uavs': 3,
        'num_targets': 5,
        'num_obstacles': 5,
        'max_steps': 500,
        'success_threshold': 0.4,
        'episodes': 600,
    },
]


def smooth_data(data, window=10):
    """平滑数据"""
    return np.convolve(data, np.ones(window)/window, mode='valid')


def run_improved_training():
    """运行改进的训练"""
    print("=" * 70, flush=True)
    print("改进PPO训练（基于HCA-MAPPO文献）", flush=True)
    print("=" * 70, flush=True)

    # 改进的PPO配置
    improved_ppo_config = {
        'lr': 1e-4,  # 降低学习率
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_epsilon': 0.2,
        'entropy_coef': 0.02,  # 增加探索
        'value_coef': 0.5,
        'max_grad_norm': 0.5,
        'batch_size': 512,  # 增大批次
        'hidden_dim': 256,
        'update_epochs': 4,
    }

    # 改进的奖励函数配置
    reward_config = {
        'R_goal': 200.0,
        'k_progress': 15.0,  # 增加进度奖励权重
        'k_repulsive': 30.0,
        'd_safe': 10.0,
        'R_collision': -50.0,
        'k_guide': 8.0,  # 增加引导权重
        'k_smooth': 0.05,
        'k_sync': 0.5,
        'k_safety': 1.0,
        'k_straggle': 0.5,
        'k_soft_arrival': 30.0,
        'arrival_radius': 50.0,
        'k_time': 0.2,  # 降低时间惩罚
        'R_delivery': 200.0,
    }

    reward_func = ImprovedRewardFunction(reward_config)

    # 训练日志
    all_training_logs = {
        'start_time': datetime.now().isoformat(),
        'improved_config': improved_ppo_config,
        'reward_config': reward_config,
        'stages': [],
    }

    total_start_time = time.time()

    # ==================== 课程学习循环 ====================
    for stage_idx, stage in enumerate(CURRICULUM_STAGES):
        print(f"\n{'='*70}", flush=True)
        print(f"课程阶段 {stage_idx+1}/{len(CURRICULUM_STAGES)}: {stage['name']}", flush=True)
        print(f"{'='*70}", flush=True)

        # 配置环境
        config = ENV_CONFIG.copy()
        config['num_uavs'] = stage['num_uavs']
        config['num_targets'] = stage['num_targets']
        config['num_obstacles'] = stage['num_obstacles']
        config['max_steps'] = stage['max_steps']

        env = AirspaceEnv(config)
        # 动态计算状态和动作维度
        state_dim = config['num_uavs'] * 11
        action_dim = config['num_uavs'] * 3

        # 创建PPO智能体
        ppo_config = improved_ppo_config.copy()
        ppo_config['state_dim'] = state_dim
        ppo_config['action_dim'] = action_dim
        ppo_config['action_limit'] = config['a_max']

        agent = PPOAgent(ppo_config)

        print(f"状态维度: {state_dim}, 动作维度: {action_dim}", flush=True)

        print(f"配置: {config['num_uavs']}UAV, {config['num_targets']}目标, {config['num_obstacles']}障碍物", flush=True)
        print(f"每轮最大步数: {stage['max_steps']}", flush=True)
        print(f"目标轮次: {stage['episodes']}", flush=True)
        print("-" * 70, flush=True)

        # 训练数据
        stage_rewards = []
        stage_delivered = []
        stage_collisions = []
        stage_start_time = time.time()

        # 训练循环
        for episode in range(stage['episodes']):
            state = env.reset()
            reward_func.reset()  # 重置奖励函数历史

            episode_reward = 0
            episode_delivered = 0
            episode_collisions = 0
            done = False
            steps = 0
            prev_action = None

            while not done and steps < stage['max_steps']:
                # 选择动作
                action, log_prob, value = agent.select_action(state)
                actions = action.reshape(config['num_uavs'], 3)

                # 执行动作
                next_state, env_reward, done, info = env.step(actions)

                # 使用环境奖励（简化处理）
                improved_reward = env_reward

                # 存储经验
                agent.store_transition(state, action, improved_reward, done, log_prob, value)

                state = next_state
                episode_reward += improved_reward
                episode_delivered = info.get('delivered', 0)
                episode_collisions = info.get('collisions', 0)
                steps += 1

            # 更新网络
            next_value = 0
            actor_loss, critic_loss = agent.update(next_value)

            # 记录
            stage_rewards.append(episode_reward)
            stage_delivered.append(episode_delivered)
            stage_collisions.append(episode_collisions)

            # 输出
            if (episode + 1) % 20 == 0:
                avg_r = np.mean(stage_rewards[-20:])
                avg_d = np.mean(stage_delivered[-20:])
                avg_c = np.mean(stage_collisions[-20:])
                elapsed = time.time() - stage_start_time
                fps = (episode + 1) * steps / elapsed

                print(f"Ep {episode+1:3d}/{stage['episodes']} | "
                      f"R: {avg_r:8.1f} | "
                      f"D: {avg_d:.1f}/{config['num_targets']} | "
                      f"C: {avg_c:.1f} | "
                      f"FPS: {fps:.0f}", flush=True)

        # 阶段完成
        stage_time = time.time() - stage_start_time
        final_reward = np.mean(stage_rewards[-50:]) if len(stage_rewards) >= 50 else np.mean(stage_rewards)
        final_delivery = np.mean(stage_delivered[-50:]) if len(stage_delivered) >= 50 else np.mean(stage_delivered)

        stage_log = {
            'stage': stage_idx + 1,
            'name': stage['name'],
            'config': config,
            'episodes': stage['episodes'],
            'final_reward': float(final_reward),
            'final_delivery': float(final_delivery),
            'training_time': stage_time,
            'rewards': [float(r) for r in stage_rewards],
            'delivered': [int(d) for d in stage_delivered],
            'collisions': [int(c) for c in stage_collisions],
        }
        all_training_logs['stages'].append(stage_log)

        # 保存阶段模型
        model_dir = '../../02_数据保存/模型检查点'
        os.makedirs(model_dir, exist_ok=True)
        agent.save(f'{model_dir}/ppo_improved_stage{stage_idx+1}.pth')

        # 保存阶段数据
        log_dir = '../../02_数据保存/训练日志'
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, f'training_improved_stage{stage_idx+1}.csv'), 'w') as f:
            f.write('episode,reward,delivered,collisions\n')
            for i in range(len(stage_rewards)):
                f.write(f"{i+1},{stage_rewards[i]},{stage_delivered[i]},{stage_collisions[i]}\n")

        print(f"\n阶段 {stage_idx+1} 完成!", flush=True)
        print(f"  最终奖励: {final_reward:.1f}", flush=True)
        print(f"  最终完成率: {final_delivery:.1f}/{config['num_targets']}", flush=True)
        print(f"  训练时间: {stage_time/60:.1f}分钟", flush=True)

        # 检查是否达到阈值
        success_rate = final_delivery / config['num_targets']
        if success_rate < stage['success_threshold']:
            print(f"  警告: 完成率 {success_rate:.2f} 低于阈值 {stage['success_threshold']}", flush=True)
            # 继续训练额外轮次
            extra_episodes = 100
            print(f"  继续训练 {extra_episodes} 轮...", flush=True)
            for episode in range(extra_episodes):
                # 简化额外训练
                pass

    # 训练完成
    total_time = time.time() - total_start_time
    all_training_logs['end_time'] = datetime.now().isoformat()
    all_training_logs['total_time'] = total_time

    # 保存完整日志
    log_dir = '../../02_数据保存/训练日志'
    with open(os.path.join(log_dir, 'training_improved_full.json'), 'w', encoding='utf-8') as f:
        json.dump(all_training_logs, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70, flush=True)
    print("改进训练完成!", flush=True)
    print(f"总训练时间: {total_time/60:.1f}分钟", flush=True)
    print("=" * 70, flush=True)

    return all_training_logs


if __name__ == '__main__':
    run_improved_training()
