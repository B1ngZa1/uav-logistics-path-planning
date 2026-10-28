"""
SCI风格训练曲线绘图
项目：基于深度强化学习的多无人机协同物流配送路径规划
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys

# 添加SCI风格配置
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from sci_style_config import *


def smooth_data(data, window=10):
    """平滑数据"""
    return pd.Series(data).rolling(window=window, min_periods=1).mean()


def plot_sci_training_curves(data_file, output_dir):
    """绘制SCI风格训练曲线"""
    # 设置SCI风格
    set_sci_style()

    # 读取数据
    df = pd.read_csv(data_file)
    print(f"加载数据: {len(df)} 轮", flush=True)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # ==================== 1. 奖励曲线 ====================
    fig, ax = get_figure(figsize=(10, 6))
    style_axis(ax, xlabel='Training Episode', ylabel='Cumulative Reward',
               title='PPO Training Reward Curve')

    # 平滑曲线
    reward_smooth = smooth_data(df['reward'], 20)
    reward_std = df['reward'].rolling(20).std()

    # 绘制
    plot_line(ax, df['episode'], reward_smooth, label='Smoothed Reward',
              color=SCI_COLORS['primary'], linewidth=2.5)
    plot_fill(ax, df['episode'], reward_smooth, reward_std,
              color=SCI_COLORS['primary'], alpha=0.2)
    plot_line(ax, df['episode'], df['reward'], label='Raw Reward',
              color=SCI_COLORS['tertiary'], linewidth=0.5, alpha=0.5)

    add_legend(ax, loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sci_reward_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("保存: sci_reward_curve.png", flush=True)

    # ==================== 2. 完成率曲线 ====================
    fig, ax = get_figure(figsize=(10, 6))
    style_axis(ax, xlabel='Training Episode', ylabel='Task Completion Rate (%)',
               title='Task Completion Rate')

    completion_rate = df['delivered'] / 5 * 100
    completion_smooth = smooth_data(completion_rate, 20)

    plot_line(ax, df['episode'], completion_smooth, label='Completion Rate',
              color=SCI_COLORS['accent'], linewidth=2.5)
    plot_fill(ax, df['episode'], completion_smooth,
              completion_rate.rolling(20).std() * 0.5,
              color=SCI_COLORS['accent'], alpha=0.2)

    ax.set_ylim(0, 100)
    add_legend(ax, loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sci_completion_rate.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("保存: sci_completion_rate.png", flush=True)

    # ==================== 3. 碰撞率曲线 ====================
    fig, ax = get_figure(figsize=(10, 6))
    style_axis(ax, xlabel='Training Episode', ylabel='Collision Rate (%)',
               title='Collision Rate')

    collision_rate = df['collisions'] / 100 * 100
    collision_smooth = smooth_data(collision_rate, 20)

    plot_line(ax, df['episode'], collision_smooth, label='Collision Rate',
              color='#E53935', linewidth=2.5)
    plot_fill(ax, df['episode'], collision_smooth,
              collision_rate.rolling(20).std() * 0.5,
              color='#E53935', alpha=0.2)

    ax.set_ylim(0, 50)
    add_legend(ax, loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sci_collision_rate.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("保存: sci_collision_rate.png", flush=True)

    # ==================== 4. 综合指标图 ====================
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=150)

    # 检查可用的列
    available_cols = df.columns.tolist()

    metrics = [
        (df['episode'], smooth_data(df['reward'], 20), 'Reward', SCI_COLORS['primary']),
        (df['episode'], smooth_data(completion_rate, 20), 'Completion (%)', SCI_COLORS['accent']),
        (df['episode'], smooth_data(collision_rate, 20), 'Collision (%)', '#E53935'),
    ]

    # 添加可选指标
    if 'entropy' in available_cols:
        metrics.append((df['episode'], smooth_data(df['entropy'], 20), 'Entropy', '#7B1FA2'))
    else:
        metrics.append((df['episode'], smooth_data(df['reward'].rolling(20).std(), 20), 'Reward Std', '#7B1FA2'))

    if 'value_loss' in available_cols:
        metrics.append((df['episode'], smooth_data(df['value_loss'], 20), 'Value Loss', '#1565C0'))
    else:
        metrics.append((df['episode'], smooth_data(df['delivered'], 20), 'Delivered', '#1565C0'))

    if 'learning_rate' in available_cols:
        metrics.append((df['episode'], df['learning_rate'] * 1000, 'LR (×10⁻³)', '#F57F17'))
    else:
        metrics.append((df['episode'], smooth_data(df['collisions'], 20), 'Collisions', '#F57F17'))

    for idx, (x, y, title, color) in enumerate(metrics):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]

        plot_line(ax, x, y, color=color, linewidth=2)
        style_axis(ax, xlabel='Episode', title=title)

    plt.suptitle('PPO Training Metrics Overview', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sci_training_overview.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("保存: sci_training_overview.png", flush=True)

    print(f"\nSCI风格图表已保存到: {output_dir}", flush=True)


if __name__ == '__main__':
    data_file = '../../02_数据保存/训练日志/training_rewards_S1_fast.csv'
    output_dir = '../../04_脚本图片输出/论文插图'
    plot_sci_training_curves(data_file, output_dir)
