"""
工作流与训练步骤可视化
项目：基于深度强化学习的多无人机协同物流配送路径规划
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from sci_style_config import *


def plot_project_workflow(output_dir):
    """绘制项目工作流图"""
    set_sci_style()

    fig, ax = plt.subplots(figsize=(16, 10), dpi=150)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # 标题
    ax.text(8, 9.5, 'Multi-UAV Cooperative Logistics Delivery Path Planning',
            ha='center', va='center', fontsize=16, fontweight='bold')
    ax.text(8, 9.0, 'Project Workflow Overview',
            ha='center', va='center', fontsize=12, style='italic')

    # 阶段定义
    phases = [
        {'name': 'Phase 0\nLiterature\nReview', 'pos': (1, 7.5), 'color': SCI_COLORS['tertiary']},
        {'name': 'Phase 1\nMathematical\nModeling', 'pos': (4, 7.5), 'color': SCI_COLORS['secondary']},
        {'name': 'Phase 2\nAlgorithm\nDesign', 'pos': (7, 7.5), 'color': SCI_COLORS['primary']},
        {'name': 'Phase 3\nSimulation\nImplementation', 'pos': (10, 7.5), 'color': SCI_COLORS['secondary']},
        {'name': 'Phase 4\nExperiment\nVerification', 'pos': (13, 7.5), 'color': SCI_COLORS['tertiary']},
    ]

    # 绘制阶段框
    box_width = 2.2
    box_height = 1.2
    for phase in phases:
        x, y = phase['pos']
        rect = mpatches.FancyBboxPatch(
            (x - box_width/2, y - box_height/2), box_width, box_height,
            boxstyle="round,pad=0.1",
            facecolor=phase['color'],
            edgecolor=SCI_COLORS['accent'],
            linewidth=2
        )
        ax.add_patch(rect)
        ax.text(x, y, phase['name'], ha='center', va='center',
                fontsize=9, fontweight='bold')

    # 绘制箭头
    for i in range(len(phases) - 1):
        x1 = phases[i]['pos'][0] + box_width/2
        x2 = phases[i+1]['pos'][0] - box_width/2
        y = phases[i]['pos'][1]
        ax.annotate('', xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle='->', color=SCI_COLORS['accent'], lw=2))

    # 子模块 - Phase 1: Mathematical Modeling
    modules_p1 = [
        ('3D Airspace\nModel', (2.5, 5.5)),
        ('UAV Kinematics\nModel', (4, 5.5)),
        ('Energy\nModel', (5.5, 5.5)),
    ]

    for name, pos in modules_p1:
        rect = mpatches.FancyBboxPatch(
            (pos[0] - 0.7, pos[1] - 0.4), 1.4, 0.8,
            boxstyle="round,pad=0.05",
            facecolor=SCI_COLORS['light'],
            edgecolor=SCI_COLORS['secondary'],
            linewidth=1.5
        )
        ax.add_patch(rect)
        ax.text(pos[0], pos[1], name, ha='center', va='center', fontsize=7)

    # 子模块 - Phase 2: Algorithm Design
    modules_p2 = [
        ('State\nSpace', (6, 5.5)),
        ('Action\nSpace', (7.5, 5.5)),
        ('Reward\nFunction', (9, 5.5)),
    ]

    for name, pos in modules_p2:
        rect = mpatches.FancyBboxPatch(
            (pos[0] - 0.7, pos[1] - 0.4), 1.4, 0.8,
            boxstyle="round,pad=0.05",
            facecolor=SCI_COLORS['light'],
            edgecolor=SCI_COLORS['primary'],
            linewidth=1.5
        )
        ax.add_patch(rect)
        ax.text(pos[0], pos[1], name, ha='center', va='center', fontsize=7)

    # 子模块 - Phase 3: Implementation
    modules_p3 = [
        ('PPO\nAlgorithm', (9, 3.5)),
        ('CTDE\nFramework', (11, 3.5)),
        ('Task\nAllocation', (13, 3.5)),
    ]

    for name, pos in modules_p3:
        rect = mpatches.FancyBboxPatch(
            (pos[0] - 0.7, pos[1] - 0.4), 1.4, 0.8,
            boxstyle="round,pad=0.05",
            facecolor=SCI_COLORS['light'],
            edgecolor=SCI_COLORS['secondary'],
            linewidth=1.5
        )
        ax.add_patch(rect)
        ax.text(pos[0], pos[1], name, ha='center', va='center', fontsize=7)

    # 子模块 - Phase 4: Experiments
    modules_p4 = [
        ('Scenario\nS1-S6', (12, 5.5)),
        ('Baseline\nComparison', (14, 5.5)),
        ('Ablation\nStudy', (14, 3.5)),
    ]

    for name, pos in modules_p4:
        rect = mpatches.FancyBboxPatch(
            (pos[0] - 0.7, pos[1] - 0.4), 1.4, 0.8,
            boxstyle="round,pad=0.05",
            facecolor=SCI_COLORS['light'],
            edgecolor=SCI_COLORS['tertiary'],
            linewidth=1.5
        )
        ax.add_patch(rect)
        ax.text(pos[0], pos[1], name, ha='center', va='center', fontsize=7)

    # 连接线
    connections = [
        ((4, 6.9), (2.5, 5.9)),
        ((4, 6.9), (4, 5.9)),
        ((4, 6.9), (5.5, 5.9)),
        ((7, 6.9), (6, 5.9)),
        ((7, 6.9), (7.5, 5.9)),
        ((7, 6.9), (9, 5.9)),
        ((10, 6.9), (9, 3.9)),
        ((10, 6.9), (11, 3.9)),
        ((10, 6.9), (13, 3.9)),
        ((13, 6.9), (12, 5.9)),
        ((13, 6.9), (14, 5.9)),
    ]

    for start, end in connections:
        ax.plot([start[0], end[0]], [start[1], end[1]],
                color=SCI_COLORS['accent'], linewidth=1, alpha=0.5, linestyle='--')

    # 底部图例
    legend_items = [
        mpatches.Patch(color=SCI_COLORS['tertiary'], label='Literature & Experiments'),
        mpatches.Patch(color=SCI_COLORS['secondary'], label='Modeling & Implementation'),
        mpatches.Patch(color=SCI_COLORS['primary'], label='Algorithm Design'),
    ]
    ax.legend(handles=legend_items, loc='lower center', ncol=3, fontsize=10,
              frameon=True, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'project_workflow.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("保存: project_workflow.png", flush=True)


def plot_training_pipeline(output_dir):
    """绘制训练流程图"""
    set_sci_style()

    fig, ax = plt.subplots(figsize=(14, 8), dpi=150)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # 标题
    ax.text(7, 7.5, 'PPO Training Pipeline', ha='center', va='center',
            fontsize=16, fontweight='bold')

    # 主流程步骤
    main_steps = [
        {'name': 'Environment\nReset', 'pos': (1, 5), 'color': SCI_COLORS['tertiary']},
        {'name': 'State\nObservation', 'pos': (3.5, 5), 'color': SCI_COLORS['secondary']},
        {'name': 'Action\nSelection', 'pos': (6, 5), 'color': SCI_COLORS['primary']},
        {'name': 'Environment\nStep', 'pos': (8.5, 5), 'color': SCI_COLORS['secondary']},
        {'name': 'Reward\nCalculation', 'pos': (11, 5), 'color': SCI_COLORS['tertiary']},
    ]

    # PPO更新步骤
    ppo_steps = [
        {'name': 'Experience\nCollection', 'pos': (2, 2.5), 'color': SCI_COLORS['primary']},
        {'name': 'GAE\nComputation', 'pos': (4.5, 2.5), 'color': SCI_COLORS['primary']},
        {'name': 'Policy\nUpdate', 'pos': (7, 2.5), 'color': SCI_COLORS['accent']},
        {'name': 'Value\nUpdate', 'pos': (9.5, 2.5), 'color': SCI_COLORS['accent']},
        {'name': 'Logging\n& Saving', 'pos': (12, 2.5), 'color': SCI_COLORS['tertiary']},
    ]

    # 绘制主流程框
    box_w, box_h = 2, 1.2
    for step in main_steps:
        x, y = step['pos']
        rect = mpatches.FancyBboxPatch(
            (x - box_w/2, y - box_h/2), box_w, box_h,
            boxstyle="round,pad=0.1",
            facecolor=step['color'],
            edgecolor=SCI_COLORS['accent'],
            linewidth=2
        )
        ax.add_patch(rect)
        ax.text(x, y, step['name'], ha='center', va='center',
                fontsize=9, fontweight='bold')

    # 绘制PPO更新框
    for step in ppo_steps:
        x, y = step['pos']
        rect = mpatches.FancyBboxPatch(
            (x - box_w/2, y - box_h/2), box_w, box_h,
            boxstyle="round,pad=0.1",
            facecolor=step['color'],
            edgecolor=SCI_COLORS['accent'],
            linewidth=2
        )
        ax.add_patch(rect)
        ax.text(x, y, step['name'], ha='center', va='center',
                fontsize=9, fontweight='bold')

    # 主流程箭头
    for i in range(len(main_steps) - 1):
        x1 = main_steps[i]['pos'][0] + box_w/2
        x2 = main_steps[i+1]['pos'][0] - box_w/2
        y = main_steps[i]['pos'][1]
        ax.annotate('', xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle='->', color=SCI_COLORS['accent'], lw=2))

    # PPO更新箭头
    for i in range(len(ppo_steps) - 1):
        x1 = ppo_steps[i]['pos'][0] + box_w/2
        x2 = ppo_steps[i+1]['pos'][0] - box_w/2
        y = ppo_steps[i]['pos'][1]
        ax.annotate('', xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle='->', color=SCI_COLORS['accent'], lw=2))

    # 循环箭头（从Reward到Experience Collection）
    ax.annotate('', xy=(2, 3.1), xytext=(11, 4.4),
                arrowprops=dict(arrowstyle='->', color=SCI_COLORS['accent'],
                               lw=2, connectionstyle='arc3,rad=0.3'))

    # 循环箭头（从Logging到Environment Reset）
    ax.annotate('', xy=(1, 4.4), xytext=(12, 3.1),
                arrowprops=dict(arrowstyle='->', color=SCI_COLORS['accent'],
                               lw=2, connectionstyle='arc3,rad=-0.3'))

    # 标签
    ax.text(6.5, 3.8, 'Training Loop', ha='center', va='center',
            fontsize=11, fontweight='bold', style='italic',
            bbox=dict(boxstyle='round', facecolor=SCI_COLORS['light'],
                     edgecolor=SCI_COLORS['accent'], alpha=0.8))

    # 底部说明
    ax.text(7, 0.8, 'Main Loop: Episode Collection → PPO Update → Repeat',
            ha='center', va='center', fontsize=10, style='italic')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_pipeline.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("保存: training_pipeline.png", flush=True)


def plot_algorithm_architecture(output_dir):
    """绘制算法架构图"""
    set_sci_style()

    fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # 标题
    ax.text(6, 7.5, 'PPO + CTDE Algorithm Architecture',
            ha='center', va='center', fontsize=16, fontweight='bold')

    # Actor网络
    actor_modules = [
        ('State\nInput', (1.5, 5.5)),
        ('FC Layer\n256', (3, 5.5)),
        ('FC Layer\n128', (4.5, 5.5)),
        ('FC Layer\n64', (6, 5.5)),
        ('Action\nOutput', (7.5, 5.5)),
    ]

    # Critic网络
    critic_modules = [
        ('Global\nState', (1.5, 3)),
        ('FC Layer\n256', (3, 3)),
        ('FC Layer\n128', (4.5, 3)),
        ('FC Layer\n64', (6, 3)),
        ('Value\nOutput', (7.5, 3)),
    ]

    # CTDE框架
    ctde_modules = [
        ('Local\nObservation', (9, 5.5)),
        ('Shared\nActor', (10.5, 5.5)),
        ('Centralized\nCritic', (10.5, 3)),
        ('Global\nReward', (9, 3)),
    ]

    # 绘制Actor
    for name, pos in actor_modules:
        rect = mpatches.FancyBboxPatch(
            (pos[0] - 0.6, pos[1] - 0.5), 1.2, 1,
            boxstyle="round,pad=0.05",
            facecolor=SCI_COLORS['primary'],
            edgecolor=SCI_COLORS['accent'],
            linewidth=1.5
        )
        ax.add_patch(rect)
        ax.text(pos[0], pos[1], name, ha='center', va='center',
                fontsize=8, fontweight='bold', color='white')

    # 绘制Critic
    for name, pos in critic_modules:
        rect = mpatches.FancyBboxPatch(
            (pos[0] - 0.6, pos[1] - 0.5), 1.2, 1,
            boxstyle="round,pad=0.05",
            facecolor=SCI_COLORS['secondary'],
            edgecolor=SCI_COLORS['accent'],
            linewidth=1.5
        )
        ax.add_patch(rect)
        ax.text(pos[0], pos[1], name, ha='center', va='center',
                fontsize=8, fontweight='bold')

    # 绘制CTDE
    for name, pos in ctde_modules:
        rect = mpatches.FancyBboxPatch(
            (pos[0] - 0.6, pos[1] - 0.5), 1.2, 1,
            boxstyle="round,pad=0.05",
            facecolor=SCI_COLORS['tertiary'],
            edgecolor=SCI_COLORS['accent'],
            linewidth=1.5
        )
        ax.add_patch(rect)
        ax.text(pos[0], pos[1], name, ha='center', va='center',
                fontsize=8, fontweight='bold')

    # Actor箭头
    for i in range(len(actor_modules) - 1):
        x1 = actor_modules[i][1][0] + 0.6
        x2 = actor_modules[i+1][1][0] - 0.6
        y = actor_modules[i][1][1]
        ax.annotate('', xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle='->', color=SCI_COLORS['accent'], lw=1.5))

    # Critic箭头
    for i in range(len(critic_modules) - 1):
        x1 = critic_modules[i][1][0] + 0.6
        x2 = critic_modules[i+1][1][0] - 0.6
        y = critic_modules[i][1][1]
        ax.annotate('', xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle='->', color=SCI_COLORS['accent'], lw=1.5))

    # CTDE连接 - 从Action Output到Local Observation
    ax.annotate('', xy=(8.4, 5.5), xytext=(8.1, 5.5),
                arrowprops=dict(arrowstyle='->', color=SCI_COLORS['accent'], lw=1.5))
    # 从Value Output到Global Reward
    ax.annotate('', xy=(8.4, 3), xytext=(8.1, 3),
                arrowprops=dict(arrowstyle='->', color=SCI_COLORS['accent'], lw=1.5))

    # 标签
    ax.text(4.5, 6.5, 'Actor Network (Policy)', ha='center', va='center',
            fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=SCI_COLORS['light'],
                     edgecolor=SCI_COLORS['primary'], alpha=0.8))

    ax.text(4.5, 2, 'Critic Network (Value)', ha='center', va='center',
            fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=SCI_COLORS['light'],
                     edgecolor=SCI_COLORS['secondary'], alpha=0.8))

    ax.text(10.5, 6.5, 'CTDE Framework', ha='center', va='center',
            fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=SCI_COLORS['light'],
                     edgecolor=SCI_COLORS['tertiary'], alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'algorithm_architecture.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("保存: algorithm_architecture.png", flush=True)


if __name__ == '__main__':
    output_dir = '../../04_脚本图片输出/论文插图'
    os.makedirs(output_dir, exist_ok=True)

    print("生成工作流可视化...", flush=True)
    plot_project_workflow(output_dir)

    print("\n生成训练流程图...", flush=True)
    plot_training_pipeline(output_dir)

    print("\n生成算法架构图...", flush=True)
    plot_algorithm_architecture(output_dir)

    print("\n所有可视化图表生成完成!", flush=True)
