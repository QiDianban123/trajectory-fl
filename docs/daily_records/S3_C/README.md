# S3-C-01：客户端模型隔离与本地训练兼容

实际日期：2026-09-13（Asia/Shanghai）。执行者/模型：Codex（GPT-5）；状态：技术候选，待 MR。

- 分支：`feature/s3-c-client-training-state`；起点 `origin/dev` `7f807b3`（已含 B / PR #33）。
- Issue：待创建；未伪造编号或审批。D7 SHA：`1f183b3`（独立 model/Trainer factory）。D8 SHA：
  `a591878`（last epoch envelope 与 adapter 隔离测试）。最终候选：`c4ea760`（契约签名对齐）。
- 复用 `clone_model_state`/`model_state_id`/`snapshot_model_state`；模型层无 federated 依赖。
  新 factory 仅构造模型，A/G 统一初始化/clone/hash；每个 client Trainer 显式带 client ID、config、
  seed、split 和 local epochs。
- `FitResult.last_checkpoint_payload` 为可选 schema-v1 envelope，TorchTrainer 仍加载 best state
  以保持 Centralized 兼容；S3 LocalTrainerAdapter 改用验证过的 last-epoch state，旧 best adapter 保留。
- 测试：定向训练/恢复/adapter/factory 通过；最终 `pytest -q` 为 `306 passed in 70.10s`；Ruff、
  `validate-config`、`git diff --check` 均退出 0。
- AI 审阅：E 与 A 视角审阅最终候选 P0=0/P1=0；真人批准、远端 CI、MR URL 待处理，不作伪造。
- 下游：D 消费 last-state ClientUpdate 与 isolated factory；G/A 消费显式 factory contract；E 保持
  Centralized best checkpoint/result 兼容。未启用 Local-only/Federated/compare UI 或实现 FedAvg。
- 继续条件：C 经 MR 合入 `dev` 后，E 从最新 dev 实现结构化 Client/Round/Fairness records；
  如签名再变更，先更新 contract 与 C/S2 回归映射并复审。
