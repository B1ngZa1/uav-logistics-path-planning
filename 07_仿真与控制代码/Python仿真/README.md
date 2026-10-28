# Python仿真代码

> 项目：基于深度强化学习的多无人机协同物流配送路径规划
> 创建日期：2026-07-01

---

## 目录结构

```
Python仿真/
├── main_train.py          # 主训练脚本
├── config.py              # 配置文件
├── requirements.txt       # 依赖包
├── environment/           # 环境模块
│   ├── airspace_env.py    # 三维空域环境
│   ├── uav_dynamics.py    # 无人机动力学
│   ├── energy_model.py    # 能耗模型
│   ├── collision_detector.py  # 碰撞检测
│   └── task_allocator.py  # 任务分配
├── algorithm/             # 算法模块
│   ├── ppo_agent.py       # PPO智能体
│   ├── network.py         # 神经网络
│   ├── reward_function.py # 奖励函数
│   └── ctde_framework.py  # CTDE框架
├── utils/                 # 工具模块
│   ├── state_encoder.py   # 状态编码
│   ├── action_processor.py # 动作处理
│   ├── logger.py          # 日志工具
│   └── visualization.py   # 可视化工具
├── visualization/         # 可视化模块
│   ├── plot_3d.py         # 三维绘图
│   ├── animation.py       # 动画生成
│   └── dashboard.py       # 仪表盘
└── experiments/           # 实验模块
    ├── run_scenario.py    # 场景运行
    ├── ablation_study.py  # 消融实验
    └── baseline_compare.py # 基线对比
```

## 运行方法

```bash
# 安装依赖
pip install -r requirements.txt

# 训练
python main_train.py

# 运行实验
python experiments/run_scenario.py

# 确定性评估（S1 为三机：state=33，policy action=9，环境动作=(3, 3)）
python experiments/run_scenario.py --scenario S1 --episodes 3 --checkpoint outputs/models/ppo_s1.pth --seed 42 --deterministic

# 冒烟评估可限制每回合步数；需要策略采样时显式使用 --stochastic
python experiments/run_scenario.py --scenario S1 --episodes 3 --max-steps 2
```

## 武汉 PPO 训练、检查点与复现

权威训练入口是 `run_wuhan_training.py`。默认产物为：

* `outputs/models/wuhan_ppo.pt`：最终完整检查点；
* `outputs/models/wuhan_ppo.checkpoints/checkpoint_epXXXXXX.pt`：周期检查点；
* `outputs/models/wuhan_ppo.checkpoints/latest.pt`：最近周期检查点；
* `outputs/models/wuhan_ppo.jsonl`：追加写入的结构化训练日志。

从 `Python仿真` 目录启动一个可复现实验：

```bash
python run_wuhan_training.py --zone B --episodes 1000 --eval-episodes 10 --seed 123 --checkpoint-interval 100
```

`--episodes` 表示本次命令新增的训练回合数。检查点保存了策略、优化器、已完成回合数、运行 ID、基础 seed、环境/PPO 配置，以及 Python、NumPy 和 PyTorch 的随机数状态，因此可以从最近检查点继续：

```bash
python run_wuhan_training.py \
  --resume outputs/models/wuhan_ppo.checkpoints/latest.pt \
  --episodes 500 --eval-episodes 10
```

恢复时会沿用检查点中的 zone、seed、输出路径和日志路径；不要传入不同的 `--seed`。例如检查点已完成 1000 回合，上面的命令会训练到累计 1500 回合。也可以指定某个固定检查点：

```bash
python run_wuhan_training.py --resume outputs/models/wuhan_ppo.checkpoints/checkpoint_ep000500.pt --episodes 500
```

每行 JSONL 都包含 `run_id`、`seed`、`timestamp_utc` 和 `event`。主要事件为 `run_start`（含完整 config 和 seed）、`train_episode`（含 episode seed 和训练指标）、`checkpoint_saved`、`evaluation_episode`、`evaluation_summary`（均值、标准差、任务得分、完成/安全返航率）以及 `run_end`。恢复训练会继续使用原 run ID，并向同一日志文件追加记录。

快速验证训练、日志和恢复流程：

```bash
python -m unittest discover -s tests -p "test_wuhan_training.py"
```

该 smoke test 使用 2 个训练回合、每回合 1 步和 1 个评估回合，不代表正式训练质量。
