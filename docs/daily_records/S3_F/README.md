# S3-F-01 联邦回归与核心准出候选

实际日期：2026-09-13（Asia/Shanghai）。执行者：Codex（GPT-5）；单人串行。

## Issue、分支与提交

- Issue：待创建。分支：`test/s3-f-federated-regression`；起点：`origin/dev`
  `c3394199fd6f33526edb6ab53fc0d1660eed8ff3`。
- D7 实质增量：`20fdd52509e66257f8b8240b9d85ee2c7e6d645c`，独立人工 FedAvg、NaN/Inf、未选/缺失结果、真实客户端访问数和 v2 JSON/CSV 重建断言。
- D8 实质增量：本 README、MS4 候选报告、测试计划和真实无参三模式 smoke CI 回归；本文件的提交 SHA 不自引用。

## 验证与缺陷

- `pytest tests/unit/test_s3_f_fedavg_regression.py tests/unit/test_training_adapter.py tests/integration/test_three_mode_experiments.py -q`：退出 0，27 passed。
- `pytest tests/system/test_s3_three_mode_smoke.py -q`：退出 0，1 passed；该测试让无参数真实 5-RSU 三模式 smoke 进入 CI。
- `ruff check src tests scripts`、`python -m src.cli validate-config`：退出 0。全量 pytest：324 passed、1 failed；失败为既有 UI health test，当前复用 `.venv` 缺少 `streamlit`，须恢复依赖后重跑。
- 人工例：样本数 1:3，`[1,3]` 与 `[5,7]` 得 `[4,6]`；整型 buffer 保留 global 值。测试期望值为人工常量。
- P0/P1：本地专项未发现。生产缺陷应回到责任模块修复；本分支没有修改生产代码。

## 审批与下游

- AI 评审：Codex（GPT-5）基础只读审阅 `20fdd52` 与 D8 工作树，P0/P1=0；B/A 独立视角仍待。真人批准、PR 和远端 CI：待创建/待通过。
- 下游：AT-05/06 的核心候选证据；UI 仍待 S3-UI-01，未据此进入 UI 或 MS4 正式完成状态。
