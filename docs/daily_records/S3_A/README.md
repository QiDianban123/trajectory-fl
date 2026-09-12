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
