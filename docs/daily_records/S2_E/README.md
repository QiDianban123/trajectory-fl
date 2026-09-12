# S2-E-01：集中式评价与训练图表交付记录

日期：2026-09-11。分支：`feature/s2-e-centralized-evaluation`。目标分支：`dev`。

## D5 交付

- 新增 `inverse_transform_batch`，严格验证浮点 `[B,T_f,2]` prediction/truth，分别通过
  train-fitted scaler 批量恢复物理坐标；拒绝 shape/dtype 不匹配、空 batch、NaN/Inf 和
  损坏的 scaler 输出。
- 新增 `evaluate_centralized`，只在反归一化成功后调用共享 `compute_metrics`，单位固定为
  `meter`，`sample_count` 由批维推导。
- 评价请求只包含已保存预测/真值、scaler、loss 日志与运行元数据；没有模型或 Trainer
  入口，不触发训练，也不复制 C 的训练循环。

## D6 交付

- 新增 `LossHistory`、集中式 loss 曲线和 prediction/truth 米制轨迹图。
- 使用同一个最终 `ResultRecord` 写出 `metrics.json` 与 `metrics.csv`；artifact 路径固定为
  相对运行目录的 `figures/loss_curve.png` 和 `figures/prediction_trajectory.png`。
- adapter 返回完整产物路径并验证四个文件均存在且非空；相同保存输入可重建同源记录和图表。

## 公共接口与边界

```text
LossHistory(train, validation=None)
inverse_transform_batch(prediction, truth, scaler) -> PhysicalTrajectoryBatch
evaluate_centralized(request) -> CentralizedEvaluationOutput
plot_loss_curve(train_losses, output_path, validation_losses=None)
plot_prediction_trajectory(future_meter, prediction_meter, output_path)
```

结果 schema 复用冻结的 `ResultRecord`、`CSV_FIELDS`、`write_json` 和 `write_csv`，没有另造
stats/result 字段。B 的 scaler/data 语义、C 的 Trainer 和联邦契约均未修改。

## 验证记录

```text
实际 Commit SHA：见本分支 S2-E 提交
MR：已合入 `dev`（PR #25）
Ruff：`ruff check src tests scripts`，All checks passed
pytest：定向 22 passed；全量 262 passed
配置/冒烟检查：`python -m src.cli validate-config` 通过；`git diff --check` 通过
评审结论：待 C 主评审、A/G 协作评审
遗留问题：当前图表选择批次首个样本；多样本挑选策略由后续 runner 配置决定
下游交接：A/G 传入已保存 prediction/truth、ProcessedDataBundle.scaler 和 loss 日志
```
