# S2-C-01 交付记录：LSTM Seq2Seq 与共享 Trainer

- 日期：2026-09-10
- 分支：`feature/s2-c-lstm-trainer`
- 主评审：E
- 第二评审：A
- 关联：F3、AT-03

## 已完成

### S2-C-D5：模型

- 新增 `LSTMSeq2Seq`，将归一化绝对位置 `history [B,T_h,2]` 编码并输出
  `pred_future [B,T_f,2]`。
- 新增 `LSTMSeq2SeqConfig` 和 `from_model_config()`，模型结构全部来自冻结配置。
- 模型只负责网络与 `state_dict`，不读取文件、不创建路径、不计算 ADE/FDE。
- 在 forward 边界拒绝空 batch、错误 shape、错误 dtype、非有限值，以及模型/输入
  device 或 dtype 不一致。

### S2-C-D6：共享训练

- 新增 `TorchTrainer`，实现 MSE、Adam、训练/验证循环、设备迁移、`eval()`、
  `no_grad()` 和有限值检查。
- 新增可配置 `gradient_clip_norm`，每个 batch 在 optimizer step 前执行梯度裁剪。
- 训练从调用者提供的 `initial_state` 深拷贝开始，不修改初始状态；训练结束后模型恢复
  到验证 loss 最低的 epoch。
- checkpoint 包含 schema、CPU `model_state`、完整模型配置、seed、epoch、split ID 和
  loss；保存使用同目录临时文件原子替换，加载使用 `weights_only=True`。
- checkpoint 加载会校验 split ID、模型配置、state key、shape、dtype 和有限值，恢复后
  的预测与保存前逐元素一致。
- 一次性 batch iterator 会在 fit 开始时固化，确保多 epoch 不会因 iterator 耗尽而静默
  产生空训练。

## 公共接口

```python
model = LSTMSeq2Seq.from_model_config(model_section)
trainer_config = TorchTrainerConfig.from_config(
    model_config,
    seed=seed,
    split_id=split_id,
    device="cpu",
)
trainer = TorchTrainer(model.contract, trainer_config)
result = trainer.fit(
    model,
    train_batches,
    validation_batches,
    initial_state=initial_state,
)
trainer.save_checkpoint(result.checkpoint_payload, checkpoint_path)
trainer.load_checkpoint(model, checkpoint_path)
```

`TorchTrainer.evaluate()` 只返回样本数和归一化坐标空间的 MSE。米制 inverse-transform、
ADE/FDE 和图表仍由 E 的评价层负责。

## 验证证据

- 模型提交：`f3975a8 feat(model): implement lstm seq2seq [S2-C-01][F3]`
- Trainer 提交：`611649b feat(training): implement torch trainer [S2-C-01][AT-03]`
- 定向测试：`25 passed`（LSTM、TorchTrainer 和模型配置）
- 全量测试：`231 passed`
- Ruff：`All checks passed!`（`src tests scripts`）
- 配置检查：`Configuration valid: run=d2-smoke mode=smoke seed=42`
- `git diff --check`：通过

定向测试覆盖输出 shape、梯度和有限值、错误输入 dtype/device、小样本过拟合、验证
`no_grad`、梯度裁剪、空 batch、metadata split ID、非有限 state、checkpoint 损坏、
split/model 不匹配及保存前后预测一致。

## 完成报告

实际 Commit SHA：`f3975a8`、`611649b`

MR：待创建，目标 `dev`

Ruff：通过

pytest：定向 `29 passed`；全量 `231 passed`

配置/冒烟检查：配置校验通过；小样本 loss 降幅和 checkpoint round-trip 通过自动化测试

评审结论：待 E 主评审、A 第二评审

遗留问题：当前只完成共享模型与 Trainer；集中式 CLI、物理指标和实验产物由 A/E/G 接入

下游交接：D 可从 `FitResult.checkpoint_payload["model_state"]` 适配本地训练状态；A/G
通过上述工厂接口组装 centralized runner；B 继续提供冻结的 `TrajectoryBatch`
