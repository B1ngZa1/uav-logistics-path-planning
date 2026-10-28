# 总控 Prompt 与任务清单快速复核

🎯 目的：判断两份文件能否直接驱动 Claude Code 连续、并发安全且证据可信地完成项目。

## P0

- **统一 Phase/Gate 编号。** `[from input]` 总控 Prompt 使用 Phase 0–7（Gate 2=算法闭环、Gate 3=真实实验、Gate 5=策划书），任务清单使用 Phase 0–6（Gate 2=真实实验、Gate 4=策划书）；同名 Gate 含义不同，会让跨文件依赖和 Agent 交接误执行。应选一套 ID，并让任务清单成为唯一机器可执行来源。
- **消除“多负责 Agent”与单写者冲突。** `[from input]` Prompt 规定策划书只能由 ProposalEditor 写，但任务 S4-03 把“ProposalEditor + BusinessAnalyst”共同列为市场/商业正文负责人；S0-04、S2-07 等也混合“负责人/审查人”。每项必须明确 `writer`、`reviewer`、写入路径，BusinessAnalyst 只能交研究输入。
- **给共享台账指定唯一写者。** `[inferred]` `CLAIMS.csv`、`DECISIONS.md`、`RUNS.md`、`BLOCKERS.md` 会被多个 Wave 使用，而现有“写锁”没有锁的所有者、获取、释放和冲突恢复协议。建议 Lead 独占 TASKS/BLOCKERS，ProjectAuditor 独占 CLAIMS，ExperimentScientist 独占 RUNS，其他 Agent 只提交 handoff/patch 请求。

## P1

- **补齐非人类阻塞终止语义。** `[from input]` 最终标准只允许“人类专属输入”保持 Blocked；网络、付费来源、GPU/磁盘、依赖服务或赛事规则缺失也可能形成真实阻塞。三次失败后应允许形成 `SUBMISSION_BLOCKED`，记录证据并请求用户作范围决策，禁止无限循环或擅自降级问题。
- **固定 Windows 命令基线。** `[from input]` Agent 工具统一写作 `Bash`，示例命令未声明执行壳；在当前 Windows 工作区应说明“工具名可为 Bash，但规范命令以 PowerShell 7 为准”，路径加引号，禁止 PowerShell→cmd/Bash 混合文件操作，并逐条记录 `$LASTEXITCODE`。
- **为跨平台复现保留第二套命令。** `[inferred]` 正式 `RUNS.md` 应同时记录 Windows PowerShell 命令和平台无关的 `python -m ...` 入口；不要把 `sha256sum`、`rm`、`export` 等 Bash 专用写法直接用于 Windows。
- **把 synthetic 隔离定义为非破坏性迁移。** `[from input]` Prompt 同时要求原始日志只读和把模拟数据迁至 synthetic；应明确先哈希、复制/登记、更新消费者、验证后再由用户批准归档，不移动或覆盖原始文件。
- **补 Lead skill 的确定映射。** `[from input]` 任务清单写 `team orchestration / project ops`，但总控 Prompt 未定义这一 skill slug；应映射为已安装 skill 或创建单一合法名称，并验证 Agent frontmatter 的 `skills` 字段受当前 Claude Code 版本支持。
- **Git 初始化需单列授权闸门。** `[inferred]` 修复空 `.git` 属于状态变更；应先保存基线与故障证据，在获批计划中显式授权 `git init`，没有真实身份时只初始化、不提交。

## 通过项

- `[from input]` 工作流连续且顺序正确：审计→规格→实现→测试→真实实验→文档→独立审查→修复→交付，并明确计划批准后立即进入执行。
- `[from input]` Agent 角色、skill、允许路径和禁止行为总体清楚，ProposalEditor 与 SubmissionReviewer 的职责分离设计正确。
- `[from input]` 数据诚信规则强：禁止伪造、synthetic/real 分区、真实实验多种子、原始日志只读、图表一命令复现、失败结果保留。
- `[from input]` 并发上限 3、分 Wave、不同路径或写锁才并行，方向正确；补齐共享台账锁协议后可执行。
- `[from input]` 完成条件是 P0/P1 为 0、关键失败非零退出码、Reviewer 书面通过和 `FINAL_ACCEPTANCE.md`，没有用时间或文件数量冒充完成。
- `[from input]` 人类信息集中写入 `HUMAN_INPUT.md`、等待时继续非阻塞任务，能减少不必要中断。

➡️ 结论：两份文件的证据观和总体编排合格；先修复 Gate 编号、单写者/共享台账所有权与 Windows/阻塞语义，即可作为稳定总控工作流。
