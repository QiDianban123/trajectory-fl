# S3-A-01 A0：三模式接口冻结候选

实际日期：2026-09-12（Asia/Shanghai）
执行者/模型：Codex（GPT-5）；单人串行。工作包：S3-A-01 接口准备子任务（A0）。
状态：技术候选草案；**待真实确认与 MR，非已批准接口**。

## Issue、分支与提交

- Issue：待创建（未取得平台 Issue/MR 创建权限，未伪造编号）。
- 分支：`docs/s3-interface-freeze`；起点：`origin/dev` `4c8c5e1`。
- 初始草案（D7 实质增量）：`2b6ecf1`，新增 `docs/s3_contract.md`。
- 评审修订（D8 实质增量）：`71333a3`、`29395c0`、`46eb867`、最终候选
  `b698e6a`；它们依次关闭 holdout、last-state、状态 ID、v2 schema、恢复和全失败轮次缺口。
- 本次仅改契约文档；没有数据、outputs、checkpoint、日志、个人路径或生产算法进入提交。

## 冻结候选内容与下游接口

- 六类歧义均写入 [S3 contract](../../s3_contract.md)：计数、holdout、last epoch state、公平预算、
  失败/恢复、UI 结构化字段。
- B：train-frozen interval 对 normalized holdout history 的 train-scaler inverse-transform 投影；
  不允许重分区/重拟合。C：可选 `FitResult.last_checkpoint_payload` 与 last-state adapter。
- E：Client/Round/Fairness v2 records、nullable failure 与 v1 兼容读取；D：不改 D2 聚合类型，
  wrapper 做 input/output state hash、全失败不聚合；G：只编排 D 的具名 request/result；A：受控
  三模式 CLI/config/无参 smoke；UI 在 A/F 核心准出前保持 disabled。

## 命令、退出码与结果

| 命令 | 退出码/结果 |
|---|---|
| `git diff --check origin/dev..b698e6a` | 0；无本步空白错误 |
| `.venv\\Scripts\\python.exe -m src.cli validate-config` | 0；`run=d2-smoke mode=smoke seed=42` |
| `git diff --cached --check`（每次候选提交前） | 0 |

本步为纯文档接口冻结，未重复训练；S3 preflight 的真实 Centralized/UI smoke 和全量 pytest
证据仍在 `chore/s3-preflight` 分支，不能替代本候选的后续远端 CI。

## 缺陷与评审闭环

- D/F 初审 `2b6ecf1`：P0 为 holdout 规则会拒绝正常互斥 split、last epoch state 无可用来源。
- 二审 `71333a3`：补 train-scaler anchor、v2/路径/恢复；仍发现 AggregationResult 兼容及 D→G
  签名问题。
- 三审 `29395c0`：关闭上述 P0；F 发现 resume 与已有输出、全失败权重冲突。
- 终审 `46eb867` 与 `b698e6a`：D/F 结论 P0=0、P1=0；最后仅要求 failed RoundRecord 的
  `error_message` 非空，D/F 均复核未引入 P0/P1。
- AI 评审：D/F 角色视角均为 Codex（GPT-5）只读审阅，绑定最终 `b698e6a`；不替代真人批准。
- 真人批准：待原规则要求的 D/F/相关协作者实际确认；无 MR URL、无远端 CI URL、无合并。

## 继续条件

1. 将本候选推送并创建目标 `dev` 的单一目的 MR，取得真实 D/F 评审与原规则真人确认。
2. MR 对最终候选运行远端 Quality Gate 成功；不得用本地记录替代。
3. 经平台合入 `dev` 后，B 才从最新 `origin/dev` 的 `feature/s3-b-client-dataloaders` 开始。
4. 如果评审改变任一 Proposed 签名、字段或错误语义，先更新 contract、受影响 S2 回归映射和审阅；
   草案未批准前，CLI/UI 未实现模式继续禁用。

## S3-A-01 生产增量（2026-09-13）

- Issue：待创建；分支：`feature/s3-a-unified-mode-cli`；起点：`428e3f6a2c3a4685bec8952e8cae40ec6d9d8788`。原工作树的 `docs/daily_records/S1_B/D2_B.md` 修改保留且未暂存。
- D7 实质增量及实现 SHA：`7600bbd9a82ba7b2bbd4c1f711d9122bc7ce2c75`。统一三模式 CLI/公共 runner、配置与路径预检、同一数据/seed/initial state、实际预算；Federated summary 使用聚合后 global state 指标，模式失败透传非零退出码。
- D8 实质增量：新增四份 S3 配置、无参数真实 smoke、README 命令和证据；D8 SHA 由本文件所在后续提交给出，避免自引用循环。
- 实际 smoke：`python scripts/run_three_mode_smoke.py` 退出 0；run_id `67251547124f`；默认 5 RSU，21/5/4 train/validation/test，三个 manifest 均 `completed` 且 `fairness.comparable=true`。
- 门禁：Ruff 退出 0；A/G 专项 pytest 退出 0（26 passed）；默认和四份 S3 `validate-config` 均退出 0；`git diff --check` 退出 0。全量 pytest 为 1（323 passed、1 failed）：`test_streamlit_page_health`；单独复跑仍失败，直接启动确认当前复用 `.venv` 缺少 `streamlit`。
- 缺陷闭环：修正旧 smoke holdout 越界样例、CLI 失败误报 0、矩阵结果缺 actual/exit、Local-only 总预算、Centralized 伪造实际预算、Federated 本地状态指标冒充 global 指标、旧 Centralized 配置兼容、resume/processed 路径约束和 `code_sha=unknown`。
- AI 基础评审：Codex（GPT-5）审查 `7600bbd` 及 D8 工作树，P0=0；全量 UI 环境门禁仍阻塞正式完成。真人审批：待创建 MR 后由非作者完成；AI 记录不代替真人批准。
- 下游接口：`src.experiments.mode_runner.run_mode/run_three_mode_matrix`、`ModeDispatchResult`、S3 `three_mode` schema、`CentralizedExperimentRequest.initial_state`。UI 可调用三个受控 `train --mode`，但 UI capability 启用属于后续 UI 工作包。
