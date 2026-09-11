# S2-A-01：集中式训练 CLI 与端到端集成

- 分支：`feature/s2-a-centralized-entry`
- 目标：`dev`
- 依赖：S2-G `CentralizedExperiment`；合并顺序为 G 后 A。

## 交付

- 新增 `python -m src.cli train --mode centralized`，CLI 仅解析参数、校验三份配置并调用
  `CentralizedExperiment`。
- 支持 processed 数据、输出根目录、run ID、data version、split ID 和 checkpoint resume 参数。
- 未知 mode、配置错误、缺失 processed 数据、路径/运行目录冲突和 checkpoint 错误均返回
  非零，并给出 `Train error:` 可定位消息。
- 成功输出 run ID、best epoch、sample count、ADE/FDE、checkpoint、metrics 和 figures 路径。
- `scripts/run_centralized_smoke.py` 一条命令生成匿名数据、调用 CLI 并验证核心产物与 loss
  下降；README 已加入复制命令。

## 验证

- 真实 smoke：best epoch `7`，loss `1.090091 -> 1.057099`，ADE `1197.706770 m`，
  FDE `1200.783340 m`。
- 定向测试：CLI、runner、checkpoint 共 `15 passed`。
- 最终全量 pytest：`287 passed`；Ruff、配置校验和 `git diff --check` 通过。MS3 初始记录
  只说明集成状态，不表示 MS3 已通过。
