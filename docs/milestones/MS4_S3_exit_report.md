# MS4：S3 三模式核心准出候选报告

日期：2026-09-13。验收基线：`origin/dev` `c339419`。候选分支：
`test/s3-f-federated-regression`。

## 技术门禁候选

| 验收项 | 本地证据 | 状态 |
|---|---|---|
| AT-05 FedAvg、失败集合与状态 ID | `test_s3_f_fedavg_regression.py`、`test_federated_contracts.py`、`test_three_mode_experiments.py` | PASS |
| AT-06 隔离、样本访问、公平性、恢复与三模式 | `test_training_adapter.py`、`test_three_mode_experiments.py`、`scripts/run_three_mode_smoke.py` | PASS |
| JSON/CSV 同源重建 | `test_three_mode_experiments.py` | PASS |
| Ruff、默认配置、真实 smoke | Ruff/配置退出 0；`test_s3_three_mode_smoke.py` 退出 0 | PASS（本地） |
| 全量 pytest、远端 CI | 本地 324 passed、1 UI health failed；远端 CI 未运行 | PENDING |

独立人工 FedAvg 用例使用样本数 1 与 3、参数 `[1,3]` 与 `[5,7]`，预期聚合为
`[4,6]`；整型 buffer 保留 global 值。该断言不调用生产聚合公式计算期望值。

## 缺陷与评审

| 等级 | 数量 | 状态 |
|---|---:|---|
| P0 | 0 | 本地 F 专项未发现 |
| P1 | 0 | 本地 F 专项未发现 |

- AI 评审：Codex（GPT-5）基础审阅 D7 `20fdd52` 与 D8 工作树，未发现 P0/P1；B/A 独立视角仍待，本地自检不代替独立 AI 评审。
- 真人审批：待 PR；没有真人批准或远端 CI 成功前不得标记为正式核心准出。
- UI：尚待 S3-UI-01 交付与浏览器系统验收；核心候选不能据此进入 UI 完成状态。

## 流程结论

本报告只记录 F 的技术候选证据。流程门禁仍要求最新 `dev` 基线、全量门禁、远端 CI、真实
非作者批准和 PR 合并；任一缺失时状态为 PENDING，不进入 UI 阶段验收。
