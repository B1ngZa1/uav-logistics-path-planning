"""
问题层级一：2D固定商家配送 — 算法对比实验

运行Q-learning、SARSA、Monte Carlo的对比训练，
并用A*计算理论上界。

使用方式:
    python run_level1_experiments.py --episodes 2000 --eval-interval 100
"""
import argparse
import json
import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment.simple_delivery_env import Simple2DDeliveryEnv
from algorithm.classical_rl import (
    QLearningAgent, SARSAAgent, MonteCarloAgent,
    train_qlearning, train_sarsa, train_monte_carlo,
    evaluate_agent, a_star_2d, a_star_tsp_order,
)


def run_a_star_baseline(env: Simple2DDeliveryEnv) -> dict:
    """使用A*计算理想最优配送方案（理论上界）"""
    start = tuple(env.depot)
    merchants = [tuple(m) for m in env.merchants]
    deliveries = [tuple(d) for d in env.delivery_points]
    depot = tuple(env.depot)

    order_sequence, total_dist = a_star_tsp_order(
        start, merchants, deliveries, depot, env.obstacles
    )

    return {
        "order_sequence": order_sequence,
        "total_distance": total_dist,
        "num_deliveries": len(order_sequence),
    }


def plot_comparison(results: dict, output_dir: str):
    """生成算法对比图"""
    os.makedirs(output_dir, exist_ok=True)

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 奖励曲线
    ax = axes[0, 0]
    window = 50
    for algo_name, data in results.items():
        if "rewards" not in data:
            continue
        rewards = np.array(data["rewards"])
        if len(rewards) > window:
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax.plot(smoothed, label=algo_name, alpha=0.8)
        else:
            ax.plot(rewards, label=algo_name, alpha=0.8)
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Average Reward (window={window})")
    ax.set_title("Training Reward Convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. 配送完成数
    ax = axes[0, 1]
    for algo_name, data in results.items():
        if "deliveries" not in data:
            continue
        deliveries = np.array(data["deliveries"])
        if len(deliveries) > window:
            smoothed = np.convolve(deliveries, np.ones(window)/window, mode='valid')
            ax.plot(smoothed, label=algo_name, alpha=0.8)
        else:
            ax.plot(deliveries, label=algo_name, alpha=0.8)
    ax.axhline(y=5, color='gray', linestyle='--', alpha=0.5, label='Max (5 orders)')
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Avg Deliveries (window={window})")
    ax.set_title("Delivery Count Convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. 最终性能柱状图
    ax = axes[1, 0]
    algo_names = []
    final_rewards = []
    final_deliveries = []
    for algo_name, data in results.items():
        if "eval" not in data:
            continue
        algo_names.append(algo_name)
        final_rewards.append(data["eval"]["avg_reward"])
        final_deliveries.append(data["eval"]["avg_deliveries"])

    x = np.arange(len(algo_names))
    width = 0.35
    bars1 = ax.bar(x - width/2, final_rewards, width, label='Avg Reward/10', color='steelblue')
    bars2 = ax.bar(x + width/2, final_deliveries, width, label='Avg Deliveries', color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels(algo_names)
    ax.set_title("Final Performance Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 4. 训练时长对比
    ax = axes[1, 1]
    times = [results.get(name, {}).get("train_time", 0) for name in algo_names]
    bars = ax.bar(algo_names, times, color=['steelblue', 'coral', 'seagreen'])
    ax.set_ylabel("Training Time (seconds)")
    ax.set_title("Training Time Comparison")
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{t:.0f}s', ha='center', va='bottom', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = os.path.join(output_dir, "level1_algorithm_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Chart saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Level 1: Algorithm Comparison Experiments")
    parser.add_argument("--episodes", type=int, default=2000, help="Number of training episodes")
    parser.add_argument("--eval-interval", type=int, default=100, help="Evaluation interval")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Final evaluation episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="outputs/experiments/level1",
                        help="Output directory for results and charts")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    results = {}

    # ================================================================
    # A* 基线
    # ================================================================
    print("=" * 60)
    print("Computing A* baseline...")
    print("=" * 60)
    env = Simple2DDeliveryEnv()
    env.reset(seed=args.seed)
    a_star_result = run_a_star_baseline(env)
    print(f"A* optimal route: {a_star_result['order_sequence']}")
    print(f"A* total distance: {a_star_result['total_distance']} steps")
    print(f"A* deliveries: {a_star_result['num_deliveries']}/5")
    results["A*"] = {
        "optimal_distance": a_star_result["total_distance"],
        "num_deliveries": a_star_result["num_deliveries"],
        "train_time": 0.0,
    }

    # ================================================================
    # Q-learning
    # ================================================================
    print("\n" + "=" * 60)
    print("Training Q-learning...")
    print("=" * 60)
    np.random.seed(args.seed)
    env = Simple2DDeliveryEnv()
    agent = QLearningAgent(
        action_dim=4, lr=0.1, gamma=0.99,
        epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.9995,
    )

    t0 = time.time()
    train_metrics = train_qlearning(env, agent, args.episodes, args.eval_interval)
    train_time = time.time() - t0

    eval_metrics = evaluate_agent(env, agent, args.eval_episodes)
    results["Q-learning"] = {
        "rewards": train_metrics["rewards"],
        "deliveries": train_metrics["deliveries"],
        "eval": eval_metrics,
        "train_time": train_time,
        "agent_stats": agent.get_stats(),
    }
    print(f"Q-learning finished in {train_time:.1f}s")
    print(f"  Eval: avg_reward={eval_metrics['avg_reward']:.1f}, "
          f"avg_deliveries={eval_metrics['avg_deliveries']:.1f}/{env.num_orders}")

    # ================================================================
    # SARSA
    # ================================================================
    print("\n" + "=" * 60)
    print("Training SARSA...")
    print("=" * 60)
    np.random.seed(args.seed)
    env = Simple2DDeliveryEnv()
    agent = SARSAAgent(
        action_dim=4, lr=0.1, gamma=0.99,
        epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.9995,
    )

    t0 = time.time()
    train_metrics = train_sarsa(env, agent, args.episodes, args.eval_interval)
    train_time = time.time() - t0

    eval_metrics = evaluate_agent(env, agent, args.eval_episodes)
    results["SARSA"] = {
        "rewards": train_metrics["rewards"],
        "deliveries": train_metrics["deliveries"],
        "eval": eval_metrics,
        "train_time": train_time,
        "agent_stats": agent.get_stats(),
    }
    print(f"SARSA finished in {train_time:.1f}s")
    print(f"  Eval: avg_reward={eval_metrics['avg_reward']:.1f}, "
          f"avg_deliveries={eval_metrics['avg_deliveries']:.1f}/{env.num_orders}")

    # ================================================================
    # Monte Carlo
    # ================================================================
    print("\n" + "=" * 60)
    print("Training Monte Carlo...")
    print("=" * 60)
    np.random.seed(args.seed)
    env = Simple2DDeliveryEnv()
    agent = MonteCarloAgent(
        action_dim=4, lr=0.01, gamma=0.99,
        epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.999,
    )

    t0 = time.time()
    train_metrics = train_monte_carlo(env, agent, args.episodes, args.eval_interval)
    train_time = time.time() - t0

    eval_metrics = evaluate_agent(env, agent, args.eval_episodes)
    results["Monte Carlo"] = {
        "rewards": train_metrics["rewards"],
        "deliveries": train_metrics["deliveries"],
        "eval": eval_metrics,
        "train_time": train_time,
        "agent_stats": agent.get_stats(),
    }
    print(f"Monte Carlo finished in {train_time:.1f}s")
    print(f"  Eval: avg_reward={eval_metrics['avg_reward']:.1f}, "
          f"avg_deliveries={eval_metrics['avg_deliveries']:.1f}/{env.num_orders}")

    # ================================================================
    # 汇总与可视化
    # ================================================================
    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)
    print(f"{'Algorithm':<15} {'Avg Reward':>12} {'Avg Deliveries':>15} {'Time(s)':>10}")
    print("-" * 55)
    print(f"{'A* (upper bound)':<15} {'N/A':>12} {a_star_result['num_deliveries']:>15} {'N/A':>10}")
    for name in ["Q-learning", "SARSA", "Monte Carlo"]:
        if name in results and "eval" in results[name]:
            e = results[name]["eval"]
            t = results[name]["train_time"]
            print(f"{name:<15} {e['avg_reward']:>12.1f} {e['avg_deliveries']:>15.1f} {t:>10.1f}")

    # 生成图表
    plot_comparison(results, args.output)

    # 保存JSON结果
    json_path = os.path.join(args.output, "level1_results.json")
    # 转换numpy数组为列表
    json_results = {}
    for algo, data in results.items():
        json_results[algo] = {}
        for k, v in data.items():
            if isinstance(v, np.ndarray):
                json_results[algo][k] = v.tolist()
            elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], (np.integer, np.floating)):
                json_results[algo][k] = [float(x) for x in v]
            elif isinstance(v, dict):
                json_results[algo][k] = {kk: float(vv) if isinstance(vv, (np.integer, np.floating)) else vv
                                          for kk, vv in v.items()}
            else:
                json_results[algo][k] = v

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {json_path}")

    print("\nDone! Level 1 experiments complete.")


if __name__ == "__main__":
    main()
