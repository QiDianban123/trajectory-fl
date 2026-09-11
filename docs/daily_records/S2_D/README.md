# S2-D-01：本地训练适配与状态兼容交付记录

- 日期：2026-09-11
- 分支：`feature/s2-d-client-training-adapter`
- 目标分支：`dev`
- 主评审：A；协作评审：C
- 关联：F5、AT-03

## D5 交付

- 新增 `clone_model_state`，逐张量断开 autograd 并复制存储，拒绝空 state、非法 key、
  非 Tensor、非 strided Tensor 和非有限浮点值。
- 新增 `model_state_id`，按排序后的 key 对 key、dtype、shape 和原始 bytes 做长度前缀的
  SHA-256；映射插入顺序不影响 ID，任一状态语义变化都会改变 ID。
- 新增 `snapshot_model_state`，把全局 state 的隔离副本与内容派生 ID 绑定，供未来轮次
  分发使用。
- 新增 `LocalTrainerAdapter`，实现现有 `FederatedClient.local_train` 边界，只委托共享
  `Trainer.fit`，没有复制 optimizer 或训练循环。

## D6 交付

- 新增 `fit_result_to_client_update`，将最佳 checkpoint state、实际有效 `sample_count` 和
  固定训练统计映射为冻结的 `ClientUpdate`。
- 训练统计字段固定为 `best_epoch`、`epoch_count`、`train_loss` 和可选
  `validation_loss`，均符合 `Mapping[str, float]` 契约。
- 适配器在训练前校验 `global_state_id` 与分发内容一致，在训练后再次检查调用方全局
  state 未被修改；初始 state、模型 state、checkpoint state 和返回 state 之间不共享存储。
- `ClientUpdate` 已通过 `AggregationRequest` 的 key、shape、dtype、round、state ID 和
  sample-count 契约验证；没有实现 FedAvg、客户端调度或联邦 round。

## 公共接口

```python
snapshot = snapshot_model_state(model.state_dict())
dispatch = ClientTrainRequest(round_index, snapshot.state_id, snapshot.state)
client = LocalTrainerAdapter(client_id, trainer, model, train_batches, validation_batches)
update = client.local_train(dispatch)
```

正式接口为：`ModelStateSnapshot`、`clone_model_state`、`model_state_id`、
`snapshot_model_state`、`fit_result_to_client_update`、`LocalTrainerAdapter` 和
`CLIENT_TRAINING_STAT_KEYS`。以上接口由 `src.federated` 导出。

## 验证证据

- 生产提交：`474129f feat(federated): add local trainer adapter [S2-D-01][F5]`
- 测试提交：`e04f90c test(federated): verify trainer state adapter [S2-D-01][AT-03]`
- 定向测试：`26 passed`（状态、冻结联邦契约、真实 TorchTrainer 集成）
- 全量测试：`262 passed`
- Ruff：`All checks passed!`
- 配置检查：`Configuration valid: run=d2-smoke mode=smoke seed=42`
- `git diff --check`：通过（忽略 worktree 既有 S1-B CRLF 工作区噪音）

复审修复后新增真实 dropout LSTM 的客户端顺序测试，并隔离 Python、NumPy、Torch
CPU/CUDA 随机状态；训练和验证 batches 现在必须可重复迭代。DataLoader 的 collate 导入
调整为工厂内延迟导入，独立 Python 进程可以直接导入 `TorchTrainer`。修复后定向测试
`14 passed`，全量测试 `265 passed`。

## 复审修复

- 每次本地训练隔离 Python、NumPy、Torch CPU/CUDA 随机状态，并使用 Trainer 配置中的
  seed；真实启用 dropout 的 LSTM 已验证客户端执行顺序不影响各自结果。
- adapter 入口要求训练和验证 batches 可重复迭代，避免一次性 iterator 在后续轮次耗尽。
- DataLoader 的 collate 导入延迟到工厂调用时，解除 `src.data` 与 `src.training` 的包初始化
  循环；独立 Python 进程可直接导入 `TorchTrainer`。

## 完成报告

实际 Commit SHA：`474129f`、`e04f90c`（本文档提交见当前分支 HEAD）

MR：待创建，目标 `dev`

Ruff：`ruff check src tests` 通过

pytest：初始定向 `26 passed`；复审修复定向 `14 passed`；最终全量 `265 passed`

配置/冒烟检查：配置校验通过；真实 `TorchTrainer` → `ClientUpdate` →
`AggregationRequest` 集成通过

评审结论：S2-D 范围内达到可评审、可合并水平，待 A 主评审和 C 协作评审

遗留问题：当前 state ID 只支持冻结契约允许的 strided Tensor；联邦客户端失败映射、调度、
FedAvg 和轮次控制按计划留给后续工作包

下游交接：未来 server 应使用 `snapshot_model_state` 生成同一轮统一基线和 state ID；
聚合侧直接消费适配器返回的 `ClientUpdate`，不要重新推断 `sample_count` 或 stats 字段
