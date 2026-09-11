# MS3：S2 集中式训练准出报告

日期：2026-09-11。验收基线：`origin/dev` `a7ce788`。验收分支：
`test/s2-f-model-recovery`。

## 准出结论

S2 的本地技术门禁满足 AT-03 候选准出条件，未发现 P0，发现的 1 个 P1 已在责任模块修复并
增加生产回归测试。最终 MS3 状态仍须等待本分支合入 `dev`、最新远端 Quality Gate 成功，
以及 B 主评审和 C/A 协作评审在 MR 中如实记录；本报告不代填人工批准。

## AT-03 映射

| 验收项 | 结果 | 生产代码证据 |
|---|---|---|
| shape、float32、device、有限值、梯度 | 通过 | `test_lstm_seq2seq.py` |
| 固定 seed 重复性 | 通过 | `test_model_recovery.py`，真实 dropout Trainer |
| 小样本过拟合明确阈值 | 通过 | `test_torch_trainer.py`，final loss < initial 的 5% |
| eval/no_grad、梯度裁剪 | 通过 | `test_torch_trainer.py` |
| checkpoint、best epoch、预测一致 | 通过 | Trainer 与 Centralized runner round-trip |
| checkpoint 缺失、损坏、schema、state 错误 | 通过 | `test_model_recovery.py`、`test_torch_trainer.py` |
| processed cache identity 与失效 | 通过 | `test_processed_data.py` |
| 三 split DataLoader 数量、顺序、尾批、metadata | 通过 | processed/data-model bridge 集成测试 |
| inverse-transform 后米制 ADE/FDE | 通过 | `test_centralized_evaluation.py` |
| JSON/CSV、图表、artifact 路径 | 通过 | evaluation 与 CentralizedExperiment 测试 |
| 重复 run、冲突、失败恢复、日志解析 | 通过 | RunContext、CLI、CentralizedExperiment 测试 |
| 常速度 baseline 与 LSTM 同数据/指标 | 通过 | `predictions.npz` 与共享评价入口测试 |
| 一条命令 Centralized smoke | 通过 | `test_s2_centralized_cli_smoke.py` |

## 缺陷状态

| 等级 | 数量 | 状态 |
|---|---:|---|
| P0 | 0 | 无 |
| P1 | 1 | 已复现并修复：Trainer seed 未应用，修复前参数最大差异 `0.109579` |

## 测试与环境

- Python `3.13.7`；PyTorch `2.13.0+cpu`；CUDA 不可用；Torch CPU threads `8`。
- unit：`250 passed / 6.60s`。
- integration：`40 passed / 41.42s`。
- system：`2 passed / 13.43s`。
- 最终全量：`292 passed / 54.55s`；Ruff、配置校验、环境检查和 `git diff --check`
  全部通过。

## Centralized smoke

- 匿名 highD 风格样本：train `17`、validation `4`、test `4`。
- LSTM：`786` 参数，best epoch `29`。
- validation loss：`0.520146 -> 0.001460`，下降约 `99.72%`。
- ADE `21.686918m`；FDE `24.571101m`；总耗时 `8.546s`。
- 产物：`config_snapshot.json`、`metadata.json`、`train.log`、`training_history.json`、
  `checkpoints/best.pt`、`predictions.npz`、顶层及 baseline 的 `metrics.json/metrics.csv`、
  三张 loss/trajectory figure 和 `manifest.json`。
- 核心运行产物总计 `136397 bytes`。

## 后续门禁

本分支合入 `dev` 且远端 Quality Gate 成功、人工评审完成后，建议确认 MS3 技术准出并开始
`S2-UI-01`。在这些条件完成前，UI 可以准备需求和原型，不应把 MS3 标记为正式通过。
