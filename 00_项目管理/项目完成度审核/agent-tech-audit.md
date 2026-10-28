# 技术实现与可复现性完成度审计

> 任务 ID：AUDIT-TECH-01  
> 审计日期：2026-07-11  
> 审计范围：`03_图表脚本`、`04_脚本图片输出`、`06_模型与算法`、`07_仿真与控制代码`、`09_软件原型`、`02_数据保存`、`data`、`outputs`、根目录压缩包及开发指南  
> 审计方式：静态检查、AST 解析、现有测试、最小无保存运行、临时目录输出验证；未安装依赖、未联网、未修改源码或项目数据。

## 1. 技术总评

**技术综合完成度：43/100。**

项目不是“只有策划书”：PPO、基础三维环境、武汉离线场景加载、碰撞/能耗模块、路径规划器、四区 2D/3D 资产、测试和一批模型检查点确实存在。`unittest` 的 29 项测试全部通过，武汉环境的形状、碰撞、返航、种子确定性等关键接口有回归测试，32 个 `.pth` 检查点也都能以 `weights_only=True` 加载。

但项目尚未形成可用于证明策划书核心结论的研究闭环：

1. `CLAUDE.md` 指定的全算法冒烟命令中，Q-learning、SARSA、Monte Carlo、DQN、A2C **五条路径全部失败**；脚本吞掉异常并以退出码 0 结束，形成“假绿灯”。
2. 基线实验和消融实验仍是 `TODO`；`experiments/run_scenario.py` 因状态维度 33 与 Agent 维度 11 不匹配而直接崩溃。
3. 标注为 PPO 收敛、S1-S6、基线和消融的主要展示数据中，有一批由 `generate_training_data.py` 按 sigmoid/指数趋势合成，不能作为训练结果或性能证据。
4. 当前真实 Level-1 结果在 500 回合评估中，Q-learning、SARSA、Monte Carlo 的 `avg_deliveries` 均为 0，尚不支持“完成率 ≥85%”“路径优化 ≥20%”等目标。
5. CTDE 类仅被定义，未进入 PPO 更新主链；基础环境 SOC 固定为 `0.8`；8 项奖励中的 `formation` 没有计算逻辑，`delivery` 按累计已送达数量逐步重复发奖。
6. `09_软件原型` 和 `AirSim仿真` 只有说明与空目录，软件原型和 AirSim 控制文件并不存在。
7. 根目录 `.git` 为空，所有 Git 命令均判定当前目录不是仓库，无法证明版本、提交、作者或产物对应代码快照。

因此应区分两种完成度：

| 维度 | 评分 | 判断 |
|---|---:|---|
| 代码与场景资产“存在度” | 61/100 | 有大量真实源码、测试、模型和场景资产 |
| 研究结论与实验“证据度” | 25/100 | 合成数据混入、对比/消融未实现、真实指标未达标 |
| 综合技术完成度 | **43/100** | 可继续开发的原型，不是可验收的最终研究成果 |

## 2. 代码、模型、数据、输出与测试清单

### 2.1 项目自有源码与文档

| 类别 | 现有内容 | 性质判断 |
|---|---|---|
| 图表源码 | `03_图表脚本` 有 15 个 `.py`、3 个 Mermaid 文件 | 多数为可运行生成器；3 个极小脚本仍为 `TODO + pass` |
| 算法/模型文档 | `06_模型与算法` 有 23 个 Markdown，约 445 KB | 设计材料较完整，但大量是目标设计，不等同于已接入实现 |
| 仿真源码 | `07_仿真与控制代码/Python仿真` 有 134 个项目 Python 文件；连同图表共 149 个 Python 文件通过 AST 解析 | 源码真实存在，但包含旧体系、新 Agent 体系和教学 DIY 模板三种成熟度 |
| 测试 | `tests/` 下 4 个测试文件；pytest 可收集 36 项；unittest 可发现 29 项 | 覆盖基础环境、PPO、武汉场景；未覆盖五 Agent 冒烟、CTDE训练、基线/消融和性能指标 |
| 开发文档 | `README.md`、`CLAUDE.md`、多版开发指南 | `无人机项目设计开发指南_v1.0.md` 对旧链路问题披露较诚实；根 README 与新 Agent 指南存在过时/过度声明 |

### 2.2 模型与实验数据

| 类别 | 数量/状态 | 性质判断 |
|---|---|---|
| PPO 检查点 | `02_数据保存/模型检查点` 有 32 个 `.pth`；32/32 可加载，27 个独立 SHA-256 | 是真实二进制模型产物，但多数只含 `actor`、`critic`，缺优化器、配置、代码版本、数据来源和环境指纹 |
| 合成演示数据 | `training_rewards.csv`、`scenario_S1_results.csv` 至 `scenario_S6_results.csv`、`baseline_comparison.csv`、`ablation_study.csv` | 由 `generate_training_data.py` 明确生成，禁止作为实验结论 |
| 训练脚本日志 | `training_rewards_S*.csv`、`training_improved_stage*.csv`、`training_enhanced.csv`、两个 JSON 日志 | 较可能来自真实运行；多数配送数长期为 0、损失极大，尚无合格性能证据 |
| Level-1 结果 | `outputs/experiments/level1/level1_results.json` + PNG | 500 回合数据存在，但三算法评估均 `avg_reward=-500`、`avg_deliveries=0`、`delivery_rate=0` |
| 根目录 `database.db` | 0 字节 | 空占位物，不计成果 |

### 2.3 生成物与第三方材料

| 类别 | 现有内容 | 性质判断 |
|---|---|---|
| 论文/展示图 | `04_脚本图片输出` 有 21 张 PNG，约 11.4 MB | 生成物存在；部分来自合成数据，且当前源码找不到 6 张改进图的直接生产入口 |
| 四区环境输出 | `outputs` 含 122 PNG、19 JSON、12 OBJ、12 MTL、PDF/HTML/开发指南 | 场景资产链较完整；2D 清单披露 OSM、参数化补绘和脱敏任务点，3D 清单披露 `flat_no_dem_input` |
| OSM/参考缓存 | `data/cache` 18 个文件，约 6.35 MB | 输入缓存/参考源，不是算法实验结果 |
| Kaiwu 参考源码 | `data/reference` 152 个文件；根目录两份 zip 与 Gorge Walk / Infinity Valley 目录对应 | 第三方参考/模板，应保留来源与许可，不应计为本项目自研实现量 |
| Python 缓存 | 106 个左右 `.pyc`、`.pytest_cache` | 运行缓存，不计成果 |
| 软件原型 | `09_软件原型` 仅 1 个 README；`web_prototype/css/js/assets` 和“截图”均为空 | 占位，未实现 |
| AirSim | `07_仿真与控制代码/AirSim仿真` 仅 README | README 所列 `airsim_config.json`、`airsim_control.py` 均不存在 |

## 3. 模块评分

| 模块 | 分数 | 主要证据与扣分原因 |
|---|---:|---|
| 数学建模与算法设计文档 | 62 | 23 篇文档和多版指南存在；参数口径在 README、config、奖励类间不完全一致，设计与接入状态混杂 |
| PPO 与基础环境 | 65 | PPO 动作边界、更新轮数、检查点往返测试通过；基础主链能被单测覆盖；旧环境电量为固定占位且奖励配置被硬编码旁路 |
| 武汉离线场景与连续控制 | 72 | 四区加载、建筑碰撞、多机形状、返航、能耗与种子测试通过；最小训练可运行；默认训练仍是单机，尚无达标性能实验 |
| CTDE / 多机协同 | 28 | `ctde_framework.py` 和集中式 Critic 存在，但无调用方接入 PPO 更新；主链是全局状态到联合动作，不是宣称的 CTDE |
| 渐进式五算法 Agent | 24 | 目录和大量源码存在，但指定 `--algo all` 冒烟五算法全失败；DIY 明确是教学占位模板 |
| 实验、基线与消融 | 18 | Level-1 管线可运行但结果为 0 配送；`run_scenario.py` 崩溃；baseline/ablation 只打印“待实现” |
| 数据与模型可追溯性 | 38 | 32 个模型可加载且多份真实日志存在；合成/真实混放，模型缺 run manifest，无法一一追溯 |
| 图表与场景生成 | 70 | 3 套数据绘图函数在临时目录成功生成 15 图；2D/3D CLI 完整、场景清单详细；部分图无当前生产入口且训练图依赖合成数据 |
| 测试与环境复现 | 50 | 29 个 unittest 通过、149 文件语法通过；无 CI、无锁文件、冒烟脚本假绿；当前环境缺 requirements 所列 `stable-baselines3` 和 `plotly` |
| 软件原型 | 5 | 只有 README 和空目录 |
| AirSim / Gazebo | 5 | AirSim 仅 README；Gazebo 无实现 |
| Git / 版本追溯 | 0 | `.git` 空；不能读取 HEAD、状态、提交或标签 |

## 4. 可运行性矩阵

| 入口/模块 | 审计命令 | 结果 | 结论 |
|---|---|---|---|
| Python 语法 | 对 149 个项目 `.py` 执行 `ast.parse` | `AST_OK=149, AST_FAIL=0` | 通过 |
| unittest | `python -m unittest discover -s tests -v` | **29/29 通过，退出码 0** | 通过；同时出现 Gym 停止维护/NumPy 2.0 警告 |
| pytest | `python -m pytest --collect-only -q -p no:cacheprovider` | 收集 36 项 | 通过 |
| pytest 非写图用例 | `python -m pytest -q -p no:cacheprovider -k 'not visualization'` | 35 通过、1 deselected、1 warning | 通过；warning 为测试函数返回非 None |
| 五 Agent 冒烟 | `python drone_delivery/train_test.py --algo all --episodes 1` | 五算法全部打印 `FAILED`，但进程退出码仍为 0 | **失败且存在假绿门禁** |
| Q-learning Agent | 同上 | `'list' object has no attribute 'sum'` | 失败；主循环已运行到结果打印附近，但入口仍不可验收 |
| SARSA Agent | 同上 | `No module named 'common_python'` | 失败；残留平台依赖 |
| Monte Carlo Agent | 同上 | `No module named 'kaiwudrl'` | 失败；残留平台依赖 |
| DQN Agent | 同上 | `No module named 'conf.conf'` | 失败；包路径/配置导入错误 |
| A2C Agent | 同上 | `add() got an unexpected keyword argument 'obs'` | 失败；样本管理接口不一致 |
| 武汉训练最小链 | `python run_wuhan_training.py --zone B --episodes 1 --eval-episodes 0 --max-steps 1 --no-save` | 退出码 0，PPO 更新 4 次 | 部分通过；结果同时标记 `terminated=true` 和 `truncated=true`，且 0 配送即安全返航，需修语义测试 |
| 旧场景评估 | `python experiments/run_scenario.py --scenario S1 --episodes 1` | `1x33 and 11x256` 矩阵维度错误，退出码 1 | 失败 |
| Level-1 实验 | 临时目录运行 `run_level1_experiments.py --episodes 1 --eval-episodes 1` | 退出码 0，生成 JSON+PNG | 管线可运行；单回合无性能意义 |
| Level-1 已存结果 | 解析 `level1_results.json` | 三算法 500 回合评估均 0 配送 | 未达到研究指标 |
| 数据驱动绘图 | 临时目录分别调用训练曲线、算法对比、SCI 图函数 | 三函数均 PASS，共生成 15 PNG | 代码可运行；输入中的主展示数据为合成数据 |
| 2D/3D 场景 CLI | 两脚本 `--help` | 均退出码 0，支持 offline/cache/zone/LOD | 入口完整；本轮未重新生成全量资产 |
| 模型加载 | 对 32 个 `.pth` 执行 `torch.load(..., weights_only=True)` | 32 成功、0 失败 | 二进制有效；不代表策略性能合格 |
| Git | `git status` / `rev-parse` / `log` | 均为 `not a git repository` | 失败 |

## 5. 策划书核心宣称与证据链核对

### 5.1 PPO：存在，但实验闭环不足

`algorithm/ppo_agent.py` 是实际实现，不是空壳；动作 tanh 限幅、更新轮数和保存加载有测试。`run_wuhan_training.py` 的一回合最小训练也能完成。但现有检查点没有 run manifest，无法证明某张曲线、某份 CSV 使用了哪个代码版本、配置和检查点；当前真实日志大多没有配送成功。因此只能确认“PPO代码存在并能更新”，不能确认“PPO已达到策划书指标”。

### 5.2 CTDE：目标设计存在，执行证据缺失

`algorithm/ctde_framework.py` 定义 `CentralizedCritic` 和状态拼接/通信辅助方法，但全项目搜索不到主训练或 PPO 更新对该框架的调用。旧主链使用一个 PPO Agent 读取所有无人机状态并输出联合动作；武汉训练入口又强制 `num_drones=1`。这两者都不足以证明“集中训练、分散执行”已经闭环。

### 5.3 8 项奖励：名义 8 项，实质不完整

`algorithm/reward_function.py` 初始化了 `goal/collision/proximity/formation/energy/time/smooth/delivery` 八个键，但：

- `formation` 只初始化为 0，没有任何更新逻辑；
- `delivery` 使用 `w_delivery * np.sum(delivered)`，会对已完成订单逐步重复奖励，而不是只在状态跃迁时奖励一次；
- 基础环境 `AirspaceEnv._compute_reward()` 使用另一套硬编码奖励，没有读取 `REWARD_CONFIG` 或上述 `RewardFunction`；
- 基础环境观测中的剩余电量固定为 `0.8`，真实能耗模块未进入该主链。

因此“8 项奖励函数已实现并用于训练”不成立。

### 5.4 多订单/任务分配：有规划器和独立模块，但并非统一主链

`algorithm/delivery_strategy.py` 的 `ScoreFirstDeliveryPlanner` 已被武汉训练和测试使用，这是积极证据。另一方面，`algorithm/task_allocator.py`/`environment/task_allocator.py` 没有进入旧主训练环境。当前应明确区分“武汉单机目标规划器已接入”“通用多机动态任务分配尚未接入”。

### 5.5 实验指标：当前不能引用

根 README 给出完成率 ≥85%、碰撞率 ≤5%、配送时间优化 ≥20%、500 回合内收敛等指标，但当前没有一份满足以下条件的证据包：真实训练脚本、固定版本、配置、随机种子、原始日志、检查点、独立评估、均值/标准差、图表。现存接近理想指标的 S1-S6/基线/消融 CSV 来自合成生成器；真实 Level-1 评估为 0 配送。

## 6. 数据—脚本—图表—文档追溯

### 6.1 可以追溯的链

- `plot_training_curves.py` → `02_数据保存/训练日志/training_rewards.csv` → 7 类训练图；函数可运行。
- `plot_algorithm_compare.py` → `baseline_comparison.csv` → 4 类算法对比图；函数可运行。
- `plot_sci_training.py` → `training_rewards_S1_fast.csv` → 4 类 SCI 图；函数可运行。
- `plot_wuhan_reference_scenes_2d.py` → `data/cache` → `scene_2d_manifest.json` + 四区 2D 图。
- `convert_wuhan_reference_scenes_2d_to_3d.py` → 2D/cache 数据 → `scene_3d_index.json`、OBJ/MTL、碰撞 JSON、预览图。
- `run_level1_experiments.py` → `level1_results.json` + `level1_algorithm_comparison.png`。

### 6.2 追溯断点

- `training_rewards.csv`、S1-S6、baseline、ablation 是合成数据，虽然能画图，但不能支撑论文实验。
- 32 个模型没有统一的配置/代码 SHA/数据 SHA/随机种子/依赖版本清单。
- 现有 `04_脚本图片输出` 不包含当前绘图源码声明的全部输出；`actual_*` 和多张 `improved_*` 图片在现有 `.py` 中找不到直接生产入口。
- 2D 任务点是脱敏训练设定，B 区存在公开参考形态泛化；3D 地形是平面且无 DEM。这些资产适合作为算法仿真场景，不应描述为测绘级武汉数字孪生。
- 文档、代码和数据没有 Git 快照，无法证明同一时间点的一致性。

## 7. 关键风险

| 风险 | 等级 | 影响 |
|---|---|---|
| 合成数据被误当真实实验 | 致命/P0 | 直接损害论文、立项或答辩可信度 |
| 冒烟脚本吞异常并返回 0 | 致命/P0 | Claude Code/CI 会错误判断任务完成 |
| CTDE、8 项奖励、多机等宣称超出执行主链 | 高/P0 | 策划书技术方案与成果证据不一致 |
| 场景评估入口维度错误 | 高/P0 | 不能形成统一评估结果 |
| 真实日志表现接近 0 配送 | 高/P0 | 核心性能指标尚未实现 |
| Git 仓库为空 | 高/P0 | 无法复现、回滚、审计或关联产物 |
| 依赖只设宽泛下限，无锁文件 | 高/P1 | NumPy/Gym/PyTorch 组合易漂移；当前环境已出现 Gym 警告，且缺两个声明依赖 |
| 同名/重复模块和两套体系并存 | 中/P1 | Agent 导入路径冲突、维护和文档容易错指 |
| 检查点缺优化器和运行清单 | 中/P1 | 不能可靠续训或复现实验 |
| 软件原型/AirSim 为空 | 中/P2 | 若策划书列为交付物，将直接验收失败 |

## 8. 修复任务与逐项验收标准

### P0-1：修复五算法冒烟与“假绿”退出码

**任务**：修复 Q-learning 的列表统计、SARSA 的 `common_python` 依赖、Monte Carlo 的 `kaiwudrl` 依赖、DQN 配置导入、A2C 样本接口；删除逐算法异常吞噬或在任一失败时汇总后 `sys.exit(1)`。

**验收标准**：

1. `python drone_delivery/train_test.py --algo all --episodes 5` 退出码为 0；输出无 `FAILED`、Traceback 或缺模块信息。
2. 每个算法至少完成 5 个完整回合、执行一次有效 `learn()`，并输出非空训练统计。
3. 人为破坏一个导入或制造断言失败时，命令必须返回非 0；新增自动化回归测试验证该行为。
4. `python -m unittest discover -s tests -v` 和 `python -m pytest -q -p no:cacheprovider` 同时通过。

### P0-2：建立唯一权威的训练—评估入口

**任务**：修复 `run_scenario.py` 的多机状态/动作维度，要求评估必须显式加载匹配场景的检查点；明确旧体系、武汉体系和 `drone_delivery` 体系的生产级主入口，其余标记 legacy/reference。

**验收标准**：

1. S1 三机环境 Agent 输入为 33 维、输出为 9 维；运行 3 回合无 shape 错误。
2. 评估默认使用确定性策略且拒绝随机未训练模型冒充 PPO 结果。
3. `terminated` 与 `truncated` 在同一步不得无解释地同时为 true；增加边界测试。
4. 每个权威入口提供 `--help`、`--seed`、`--output-dir`、`--no-save`/dry-run 和明确退出码。

### P0-3：隔离合成数据并重做真实实验

**任务**：把 `generate_training_data.py` 及其产物迁移到明确的 `demo/synthetic` 命名空间，在策划书/论文/答辩图表构建中禁止读取；实现真实 baseline 和 ablation。

**验收标准**：

1. 每个数据文件包含 `data_kind=real|synthetic`；报告构建器检测到 synthetic 时直接失败。
2. 基线至少包含 Random、Greedy/A*、本文方法；消融至少覆盖 CTDE、能耗、碰撞/安全奖励。
3. 每组至少 5 个固定且公开的随机种子，报告均值、标准差和 95% CI。
4. 每个 run 生成 `run_manifest.json`，至少记录：命令、时间、Git SHA、dirty 状态、Python/依赖版本、配置完整快照、种子、环境 ID、checkpoint SHA-256、数据 SHA-256。
5. 图表只从上述真实 run 聚合；从原始日志一条命令可重建表格与图。

### P0-4：让核心算法宣称与实际执行一致

**任务**：二选一：完成 CTDE、真实能耗、形成约束和多机任务分配接入；或在策划书中降级为“目标架构/后续工作”。修复奖励一次性语义。

**验收标准**：

1. CTDE 测试证明：Actor 只接收局部观测、Critic 接收全局状态、一次 PPO 更新后集中式 Critic 参数发生有效变化。
2. `formation` 奖励在改变队形误差时非零且方向正确。
3. 已配送订单只在 `未完成→完成` 跃迁时奖励一次；连续 10 步保持完成状态不得重复发奖。
4. SOC 随能耗单调下降且能量守恒误差在预设容差内；基础环境不再返回固定 `0.8`。
5. 三机测试证明任务分配互斥、可重分配且不会给不同无人机重复计同一订单。

### P0-5：修复版本与成果追溯

**任务**：恢复有效 Git 仓库，建立数据/模型版本规范和可复现实验标签。

**验收标准**：

1. `git fsck --full`、`git status`、`git rev-parse HEAD` 正常；关键成果对应不可变 tag。
2. `.gitignore` 排除缓存和临时输出；大模型采用明确的 LFS/发布包策略。
3. 每张正式图可从 manifest 定位到数据 SHA、checkpoint SHA 和 commit SHA。
4. 第三方 Kaiwu 代码/zip 有来源、许可证与改动说明，不计入自研代码统计。

### P1：工程化与测试门禁

1. 锁定 Python 与依赖版本，生成可复现 lock/constraints；在干净环境安装并运行全部门禁。
2. 迁移旧 `gym` 链至 Gymnasium，消除 NumPy 2.0 兼容警告。
3. 增加 CTDE、奖励分量、Agent 冒烟、真实数据 schema、manifest、baseline/ablation 的测试。
4. CI 至少执行 AST/格式检查、unittest、pytest、五算法短冒烟和一条武汉短链。
5. 检查点保存 Actor、Critic、优化器、归一化统计、配置、步数与版本元数据，并验证断点续训等价性。
6. 合并/重命名同名 `task_allocator`、airspace、reward 等重复模块，建立清晰的包边界。

### P2：可选交付物补全或从范围删除

1. 若软件原型是交付物：补 `index.html`、样式、脚本、真实数据适配层、截图和启动说明，并进行浏览器冒烟测试；否则从成果清单删除。
2. 若 AirSim 是交付物：补配置、控制桥接、版本/地图说明和录屏；否则明确标注“未实施的可选扩展”。
3. 补齐图表生产入口，删除无来源图片或建立 manifest；正式图应包含真实/合成水印与数据来源。

## 9. 建议的技术验收门禁

在策划书宣称“技术部分完成”前，以下门禁必须全部为绿色：

```text
[ ] Git 仓库有效，正式结果有 tag 和 SHA
[ ] 149+ Python 文件语法检查通过
[ ] unittest 全部通过（当前基线：29）
[ ] pytest 全部通过且不靠吞异常
[ ] 五算法 --algo all 冒烟通过；任一失败返回非 0
[ ] 旧场景/武汉场景至少各一条短链可运行
[ ] CTDE/能耗/奖励/任务分配的接入测试通过，或从“已完成”宣称中删除
[ ] baseline 与 ablation 真实执行并有多种子统计
[ ] synthetic 数据与正式数据物理隔离，报告禁止读 synthetic
[ ] 数据、模型、图表、文档通过 manifest 一一追溯
[ ] 目标性能由独立评估脚本验证，不从训练日志直接推断
```

## 10. 审计限制

1. `.git` 为空，无法执行代码历史、提交归因、版本差异或文件来源时间线分析；本报告只审计当前文件快照。
2. 按任务约束未联网、未安装缺失依赖、未运行长时间完整训练；因此没有尝试重新证明 85% 完成率等性能目标。
3. 未执行会覆盖项目输出的可视化测试；pytest 排除了 `test_visualization`，但以临时目录单独验证了 15 张数据图生成并清理临时文件。
4. 对四区 2D/3D 场景只验证 CLI、清单、已有资产和相关单测，未全量重新生成 OSM/OBJ 资产。
5. 检查点“可加载”只证明文件结构有效，不证明其训练来源、策略质量或与现有代码兼容。
6. 并发变更说明：本报告完成验证后，工作区在 2026-07-11 14:29:26 新出现 `07_仿真与控制代码/Python仿真/run_full_experiments.py`（44,434 bytes）；该文件不在审计开始时的清单内，也不是本审计创建的。文中的“134 个仿真 Python 文件 / 149 个 AST 通过”均指该并发文件出现前的审计快照；新入口应在确认来源后单独执行语法、`--help`、最小无保存运行和产物追溯审查，不能自动视为已完成修复。

## 11. 最优先修复项（一句话）

**先把五算法冒烟改成真实门禁并隔离全部合成数据，再建立带 Git/配置/种子/模型哈希的统一训练—评估—图表流水线；在此之前，不应在策划书中把 CTDE、8 项奖励、基线/消融和性能指标写成“已完成成果”。**
