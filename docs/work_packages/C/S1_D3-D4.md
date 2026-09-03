# S1-C-01：NumPy 样本到 Torch Batch 桥接

- 阶段：S1（D3—D4）
- 角色：C，模型与训练开发
- 分支：`feature/s1-c-torch-batching`
- 主评审：E；协作：B
- 关联：F1、F3、AT-01

## 目标

建立数据层 NumPy `TrajectorySample` 到训练层 Torch `TrajectoryBatch` 的唯一显式转换边界。

## D3

- 生产代码：实现 sample-to-tensor/collate 接口，固定 batch-first、`float32` 和有限值。
- 测试：单样本、batch、错误 shape/dtype、空 batch 和 NaN/Inf。
- 建议提交：`feat(training): add torch batch bridge [S1-C-01][F3]`。

## D4

- 生产/联调：实现 DataLoader collate、device 前置校验和 metadata 保留。
- 测试：与 B Dataset、冻结 ModelContract 的 shape/dtype 对接。
- 建议提交：`test(training): verify data model bridge [S1-C-01][AT-01]`。

## 交付与验收

- 转换后 `history=[B,T_h,2]`、`future=[B,T_f,2]`，均为 `torch.float32`。
- 数据层不依赖模型实现，转换不改变 split/meta 或坐标语义。
- 不实现 LSTM 或优化循环；B/E 评审通过。
