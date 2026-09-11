# S2-G-01：常速度基线与 CentralizedExperiment

- 分支：`feature/s2-g-centralized-runner`
- 目标分支：`dev`
- 范围：常速度 sanity baseline、集中式实验编排和可复现产物

## 公共接口

- `ConstantVelocityBaseline`：只接受标准 `TrajectoryBatch.history`，输出
  `[B,T_f,2]`，在归一化坐标中按最后两帧速度外推。
- `collect_predictions`：在 `no_grad` 下收集标准 batch 的 history、prediction、truth，
  不计算指标也不触发训练。
- `evaluate_prediction_arrays`：对 baseline 和 LSTM 统一执行 scaler inverse-transform、
  米制 ADE/FDE 和 sample count。
- `CentralizedExperiment`：调用 B 的 reader/loader、C 的 model/Trainer、E 的评价和
  ResultRecord、RunContext 的 manifest；自身不实现清洗、训练循环、ADE/FDE 或路径安全。

## 产物

成功运行保存 config snapshot、metadata、结构化 train.log、LSTM checkpoint、normalized
predictions、LSTM metrics JSON/CSV、constant-velocity metrics JSON/CSV、两类轨迹图、loss
曲线和完整 RunContext manifest。失败运行保留配置、日志、失败 ResultRecord、metadata 和
manifest；同一个 run_id 不会覆盖旧目录。

## 验证

- baseline 覆盖静止、匀速、batch shape、错误 dtype/shape、NaN 和米制指标对接。
- fake Trainer 覆盖 load → loaders → model → trainer → fit → checkpoint → evaluate 编排顺序。
- fake 失败路径覆盖 failed status、错误信息、重复 run 拒绝和 manifest 保留。
- 真实小样本 smoke 使用真实 LSTM、TorchTrainer、DataLoader、baseline 和 E 评价接口。
- 定向测试：`26 passed`；全量测试：`286 passed`。
- 真实 smoke：loss `12.501712322235107`、LSTM ADE `4.839854313775069 m`、
  FDE `5.3568157744401095 m`；常速度 baseline ADE/FDE 均为 `0.0 m`。
- Ruff、配置校验和 `git diff --check` 通过。
