"""
绘制训练曲线图
项目：基于深度强化学习的多无人机协同物流配送路径规划
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


def smooth_data(data, window=10):
    """平滑数据"""
    return pd.Series(data).rolling(window=window, min_periods=1).mean()


def plot_training_curves(data_file, output_dir):
    """绘制训练曲线"""
    # 读取数据
    df = pd.read_csv(data_file)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False

    # 1. 奖励曲线
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['episode'], smooth_data(df['reward'], 20), 'b-', linewidth=2, label='平滑奖励')
    ax.plot(df['episode'], df['reward'], 'b-', alpha=0.2, linewidth=0.5)
    ax.fill_between(df['episode'],
                    smooth_data(df['reward'], 20) - df['reward'].rolling(20).std(),
                    smooth_data(df['reward'], 20) + df['reward'].rolling(20).std(),
                    alpha=0.2, color='blue')
    ax.set_xlabel('训练回合 (Episode)', fontsize=12)
    ax.set_ylabel('累积奖励', fontsize=12)
    ax.set_title('PPO训练奖励曲线', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_reward_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  保存: training_reward_curve.png")

    # 2. 任务完成率曲线
    fig, ax = plt.subplots(figsize=(10, 6))
    completion_rate = df['delivered'] / 5 * 100  # 5个目标
    ax.plot(df['episode'], smooth_data(completion_rate, 20), 'g-', linewidth=2, label='任务完成率')
    ax.set_xlabel('训练回合 (Episode)', fontsize=12)
    ax.set_ylabel('任务完成率 (%)', fontsize=12)
    ax.set_title('任务完成率变化曲线', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'completion_rate_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  保存: completion_rate_curve.png")

    # 3. 碰撞率曲线
    fig, ax = plt.subplots(figsize=(10, 6))
    collision_rate = df['collisions'] / 100 * 100
    ax.plot(df['episode'], smooth_data(collision_rate, 20), 'r-', linewidth=2, label='碰撞率')
    ax.set_xlabel('训练回合 (Episode)', fontsize=12)
    ax.set_ylabel('碰撞率 (%)', fontsize=12)
    ax.set_title('碰撞率变化曲线', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 50)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'collision_rate_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  保存: collision_rate_curve.png")

    # 4. 策略熵曲线
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['episode'], smooth_data(df['entropy'], 20), 'm-', linewidth=2, label='策略熵')
    ax.set_xlabel('训练回合 (Episode)', fontsize=12)
    ax.set_ylabel('策略熵', fontsize=12)
    ax.set_title('策略熵变化曲线', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'entropy_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  保存: entropy_curve.png")

    # 5. 损失曲线
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(df['episode'], smooth_data(df['value_loss'], 20), 'b-', linewidth=2, label='价值损失')
    ax1.set_xlabel('训练回合 (Episode)', fontsize=12)
    ax1.set_ylabel('价值损失', fontsize=12)
    ax1.set_title('价值损失变化', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)

    ax2.plot(df['episode'], smooth_data(df['policy_loss'], 20), 'r-', linewidth=2, label='策略损失')
    ax2.set_xlabel('训练回合 (Episode)', fontsize=12)
    ax2.set_ylabel('策略损失', fontsize=12)
    ax2.set_title('策略损失变化', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  保存: loss_curves.png")

    # 6. 学习率曲线
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['episode'], df['learning_rate'] * 1000, 'g-', linewidth=2, label='学习率')
    ax.set_xlabel('训练回合 (Episode)', fontsize=12)
    ax.set_ylabel('学习率 (×10⁻³)', fontsize=12)
    ax.set_title('余弦退火学习率调度', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learning_rate_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  保存: learning_rate_curve.png")

    # 7. 综合训练指标图
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # 奖励
    axes[0, 0].plot(df['episode'], smooth_data(df['reward'], 20), 'b-', linewidth=2)
    axes[0, 0].set_title('累积奖励', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].grid(True, alpha=0.3)

    # 完成率
    axes[0, 1].plot(df['episode'], smooth_data(completion_rate, 20), 'g-', linewidth=2)
    axes[0, 1].set_title('任务完成率 (%)', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylim(0, 100)
    axes[0, 1].grid(True, alpha=0.3)

    # 碰撞率
    axes[0, 2].plot(df['episode'], smooth_data(collision_rate, 20), 'r-', linewidth=2)
    axes[0, 2].set_title('碰撞率 (%)', fontsize=12, fontweight='bold')
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].grid(True, alpha=0.3)

    # 熵
    axes[1, 0].plot(df['episode'], smooth_data(df['entropy'], 20), 'm-', linewidth=2)
    axes[1, 0].set_title('策略熵', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].grid(True, alpha=0.3)

    # 价值损失
    axes[1, 1].plot(df['episode'], smooth_data(df['value_loss'], 20), 'c-', linewidth=2)
    axes[1, 1].set_title('价值损失', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].grid(True, alpha=0.3)

    # 学习率
    axes[1, 2].plot(df['episode'], df['learning_rate'] * 1000, 'y-', linewidth=2)
    axes[1, 2].set_title('学习率 (×10⁻³)', fontsize=12, fontweight='bold')
    axes[1, 2].set_xlabel('Episode')
    axes[1, 2].grid(True, alpha=0.3)

    plt.suptitle('PPO训练综合指标', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_overview.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  保存: training_overview.png")

    print(f"\n所有训练曲线已保存到: {output_dir}")


if __name__ == '__main__':
    data_file = '../../02_数据保存/训练日志/training_rewards.csv'
    output_dir = '../../04_脚本图片输出/论文插图'
    plot_training_curves(data_file, output_dir)
