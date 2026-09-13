# S3-G-01：三模式 runner、恢复与持久化边界

日期：2026-09-13（Asia/Shanghai）

分支：`feature/s3-g-three-mode-runner`

基线：`origin/dev` `c4cc8e4`
修复基线确认：`fix/s3-review-findings` 头提交 `13bb622` 是 `origin/dev` 的祖先。

## 实现范围

- 新增 `LocalOnlyExperiment`：按客户端从同一初态独立训练，消费 `ClientDataBundle`、C 的模型/
  trainer 工厂和 D 的 `run_local_only_clients`，记录 completed/failed/skipped、训练/评价计数、
  sample visits、loss、ADE/FDE、耗时、状态哈希、客户端画像和安全相对 artifact 路径。
- 新增 `FederatedExperiment`：只通过 D 的 `run_federated_round`、
  `InMemoryFederatedServer`、`DeterministicClientSelector` 推进轮次；保存 selection、失败、FedAvg
  权重、训练计数、sample visits、输入/输出 global state ID、指标、耗时和轮次 checkpoint。
- 新增 schema v2 manifest 与 `checkpoints/recovery.json` 正式恢复入口。Local-only 仅提交完整
  客户端边界，Federated 仅提交完整聚合轮次边界；单独保存恢复边界 manifest，失败轮不会推进或
  污染可恢复的 global/loader/RNG 状态。
- 恢复包保存 config digest、八字段 identity、initial/current state、Python/NumPy/Torch 与 loader
  RNG、selector 状态、完成客户端/轮次、planned/actual budget、retry sample visits、恢复来源和
  artifact 索引。坏 checkpoint、错 identity、非连续轮次、重复或已完成 run 均在训练前拒绝。
- 扩展 E 的 `ClientResultRecord`、`RoundRecord`，新增 `FairnessRecord`，保持既有构造方式兼容。
  三模式矩阵逐字段比较身份与预算；actual 与 planned 不同会保留记录但返回非零。
- CSV 仅从结构化客户端记录导出；JSON/manifest 是事实源。未复制 Trainer、FedAvg、ADE/FDE、
  client loader 或汇总算法。

## 自动化验证

- 真实链路：2 客户端 × 1 round，使用真实 `LocalTrainerAdapter` / `TorchTrainer`。
- 确定性：客户端插入顺序互换后，客户端末态、指标和 global state hash 一致。
- 失败：部分失败保留成功记录；全失败无 summary；空 train 客户端显式 skipped。
- 防护：身份和 planned budget 每个字段扰动均在输出目录创建前退出 2；覆盖旧 run、恢复完成 run、
  坏 state checkpoint 均拒绝。D 的既有测试覆盖重复/过期 update 和重复聚合。
- 恢复：客户端及轮次中断恢复与连续运行对比状态 hash、轮次、客户端结果、sample visits、
  RNG/selector 状态和 artifact 索引；失败客户端重试成本写入 `retry_sample_visits`；全失败轮从上一
  完整边界重跑。
- 最终本地结果：新增 S3-G 集成测试 `10 passed`；全量测试在补齐仓库已声明的 Streamlit
  依赖后为 `323 passed`。Ruff、`validate-config`、`git diff --check` 均通过。

## 评审与审批（分列）

| 项目 | 状态 | 记录 |
|---|---|---|
| G 自检 | 已完成 | 已检查恢复原子边界、失败语义、身份/预算、事实源、依赖方向和版本化恢复提交；全部自动门禁通过。 |
| D 评审 | 待评审 | 尚无 D 成员正式评审结论，不记录为通过。 |
| A 评审 | 待评审 | 尚无 A 成员对 CLI/config dispatch 边界的正式评审结论，不记录为通过。 |
| 真人审批 | 待审批 | 尚未取得真人批准；Local-only、Federated、compare 的 UI capability 不应启用。 |

## 后续交接

- A：接入配置 schema、CLI dispatch、路径白名单和无参 smoke；将 `ExperimentInputError.exit_code=2`
  映射为 CLI 用法错误，将 `ModeRunResult.exit_code` 原样返回。
- F/UI：从 schema v2 manifest 只读展示 clients/rounds/fairness/error/artifacts；完成系统门禁和真人
  审批前保持 S3 capability disabled。
- D/A/真人：分别填写上表的正式评审与批准记录，不以自动化测试替代人工确认。
