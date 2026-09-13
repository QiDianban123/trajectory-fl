# S3-UI-01 UI-1：三模式受控入口与结构化读取

实际日期：2026-09-13（Asia/Shanghai）。执行者：Codex（GPT-5），单人串行。

## 前置与范围

- 基线：`origin/dev` `bd96d0434dd91a4c53894de5e2a0ffe69ab381c3`；用户确认远端 CI、真人非作者审批和独立评审已完成。
- 分支：`feature/s3-ui-three-mode-console`；本增量仅扩展 `src/ui/` 与 UI 测试，不修改数据、训练、FedAvg 或指标算法。
- UI-1 完成受控入口与只读结构化索引；三模式对比图、完整页面交互验收和发布能力属于 UI-2/后续，未声明 UI 完成。

## 按钮与安全边界

| 按钮 | 输入与命令 | 成功/失败行为 |
|---|---|---|
| Centralized / Local-only / Federated | allowed processed split、safe run_id；`python -m src.cli train --mode <mode> --experiment configs/experiments/s3_<mode>_smoke.yaml ...` | 预检显示冻结公平性字段；production runner 再校验。非零结果保留并显示。 |
| 三模式 smoke | 无参数；`python scripts/run_three_mode_smoke.py` | 生产脚本创建唯一匿名样例/run_id，读取三份 manifest。 |
| 恢复 | Local-only/Federated 的已存在 run_id；固定 `checkpoints/recovery.json` | 仅允许正式恢复边界；不存在、越界或已完成 run 被拒绝。 |

所有命令为参数数组，`shell=False`；路径经 output/run root 解析且符号链接越界被拒绝。重复输出、同 run_id 并发、超时和非零退出保持可见状态；`UiRunState` 存于 session，因此刷新不再次启动进程。

## 结果与测试

- `RunIndexReader` 兼容旧 S2 manifest 和 S3 schema v2，仅读取 identity/fairness/clients/rounds/summary/artifacts；`ArtifactResolver` 只接受 run root 内的相对路径。
- `ruff check src/ui tests/unit/test_ui_*.py`：退出 0。
- `pytest tests/unit/test_ui_capabilities.py tests/unit/test_ui_command_runner.py tests/unit/test_ui_run_index.py -q`：退出 0，7 passed。
- `pytest tests/system/test_ui_smoke.py -q`：退出 0，1 passed。
- AI 基础评审：待候选 SHA 后执行；真人审批/MR/远端 CI：待 UI 分支提交后进行。下游 UI-2 应消费同一结构化对象，不解析日志或复制算法。

## UI-2 可操作结果视图

- 修复 S3 v2 `summary` 使用 ResultRecord 内联 `ade`/`fde`/`total_seconds` 时的展示空值；旧 S2 `metrics.json` 格式保持兼容。
- 页面提供 mode/status 筛选、三模式结果表、ADE（m）比较图、客户端画像、联邦轮次/权重/global-state/失败表及安全 artifact 浏览。所有数值直接来自 manifest/ResultRecord。
- 浏览器真实操作：选择“三模式 smoke”并点击执行；run_id `eadbe9acf73b`，退出 0，三模式 manifest 均可从页面索引读取。刷新只读取 session/result index，不调用命令。
