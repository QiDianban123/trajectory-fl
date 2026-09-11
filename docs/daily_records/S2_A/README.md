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

- 复审 smoke：best epoch `29`，loss `0.520146 -> 0.001460`，下降约 `99.72%`；
  ADE `21.686918 m`，FDE `24.571101 m`。
- 复审定向测试：CLI、初始化、runner、checkpoint 和真实子进程 smoke 共 `18 passed`。
- 最终全量 pytest：`290 passed`；Ruff、配置校验和 `git diff --check` 通过。MS3 初始记录
  只说明集成状态，不表示 MS3 已通过。

## 复审修复

- 补齐配置声明的 Xavier 初始化；相同 seed 生成相同初始 state，resume checkpoint 时跳过
  重新初始化。
- checkpoint 保存后强制恢复并重新预测，逐元素验证与保存前预测一致。
- CLI 按“显式参数、experiment 引用、默认路径”解析配置，并将 processed、output、run ID
  和 checkpoint 来源写入有效配置快照。
- 缺失数据测试使用独立临时项目目录并断言具体错误；增加损坏 checkpoint 错误码测试和
  真实子进程一键 smoke 系统测试。
- smoke 门槛固定为有限 loss 且至少下降 20%；实际验证 `0.520146 -> 0.001460`，下降
  `99.72%`，ADE `21.686918 m`，FDE `24.571101 m`。
