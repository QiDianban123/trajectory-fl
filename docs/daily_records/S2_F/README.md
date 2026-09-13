# S2-F-01：模型恢复与集中式回归

- 分支：`test/s2-f-model-recovery`
- 基线：`origin/dev` `a7ce788`
- 目标：`dev`
- 人工评审：B 主评审、C/A 协作评审，当前均待 MR 记录

## 生产测试审计

验收测试直接调用 `LSTMSeq2Seq`、`TorchTrainer`、`ProcessedDatasetReader`、
`create_dataloaders`、`evaluate_centralized`、`CentralizedExperiment`、`RunContext` 和生产 CLI。
只检查常量或测试自身构造的数据不计为 AT-03 证据。

## 缺陷与修复

- P1：`TorchTrainerConfig.seed` 未在 `fit` 内生效。真实两层 dropout LSTM 使用相同初始
  state 与 seed 连续训练，修复前最终参数最大差异为 `0.109579`。
- 修复：`TorchTrainer.fit` 在隔离的 CPU/CUDA RNG scope 内应用配置 seed，并恢复调用方
  RNG；修复后 checkpoint state 与 metrics 重复一致。
- P0：未发现。

## 验证基线

- Python `3.13.7`，PyTorch `2.13.0+cpu`，CUDA `False`，Torch CPU threads `8`。
- unit：`250 passed / 6.60s`；integration：`40 passed / 41.42s`；system：
  `2 passed / 13.43s`。
- Centralized smoke：train/validation/test=`17/4/4`，模型参数 `786`，耗时 `8.546s`，
  validation loss `0.520146 -> 0.001460`，ADE `21.686918m`，FDE `24.571101m`。
- smoke 运行目录产物共 `136397 bytes`；最终全量 `292 passed / 54.55s`，Ruff、配置、
  环境与 diff Gate 全部通过。
