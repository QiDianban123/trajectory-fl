# S3-A 步骤 0 前置检查

实际日期：2026-09-12（Asia/Shanghai）
执行者/模型：Codex（GPT-5）；单人串行；提示词：S3 步骤 0。
状态：**BLOCKED（不具备进入 S3 步骤 1 的完整条件）**。

## 范围与基线

- 工作树：`E:/SmallTerm2026/trajectory-fl-s3`（独立 detached worktree 后创建分支）。原
  worktree 的 `docs/daily_records/S1_B/D2_B.md` 未提交 CRLF 修改已保留，且从未暂存或提交。
- 起点：`origin/dev` = `4c8c5e127ac699dd653cb2671f9070a021f92032`（fetch 于本次检查）。
- Issue：待创建；本地没有 GitHub CLI/Issue 创建授权，未伪造编号。
- 本步分支：`chore/s3-preflight`；实现提交：
  `7769d3a35e802ac5f2c7e9c2fd40857a56515e5f`
  (`fix(smoke): force headless artifact rendering [S3-A-01]`)。

## S2 / S2-UI 来源核对

- S2-F：PR #28 合并提交 `ca4cfd92abbf035a9962098490339ed8ac6cbe88`；其源提交
  `00ecd0b` 被 `origin/dev` 包含。
- S2-UI：PR #29 合并提交 `4c8c5e127ac699dd653cb2671f9070a021f92032`；其源提交
  `3df96b5` 被 `origin/dev` 包含。
- `docs/milestones/MS3_S2_exit_report.md` 明确仍待远端最新 Quality Gate 与真人评审；
  `docs/daily_records/S2_UI/README.md` 明确人工评审待 MR 记录且既有证据仅 HTTP 200。
  本机未安装 `gh`；2026-09-12 对 GitHub REST API 的查询返回匿名请求 rate-limit exceeded，
  故没有把本地合并记录当作 Quality Gate 或真人批准通过。

## 发现与最小修复

无 `MPLBACKEND` 时，`scripts/run_centralized_smoke.py` 第一次运行退出码 1：Matplotlib
默认 Tk 后端因运行时无可用 `init.tcl` 失败，阻断 README 的一条命令 smoke 以及 UI 子进程。
修复在 `src/evaluation/visualization.py` 的 `pyplot` 导入前选择 `Agg`；只影响保存的图表
artifact，不改变 CLI、配置或冻结的生产接口。系统 smoke 回归测试清除了继承的
`MPLBACKEND`，防止 CI 的环境变量掩盖该缺陷。

## 实际命令与结果

| 命令 | 退出码 / 结果 |
|---|---|
| `.venv\\Scripts\\python.exe scripts/check_environment.py` | 0；Python 3.12.14，依赖完整 |
| `.venv\\Scripts\\python.exe scripts/run_s1_smoke.py --workspace outputs/s3-preflight-s1` | 0；17/4/4，5 RSU，split 互斥 |
| 修复前 `...run_centralized_smoke.py --workspace outputs/s3-preflight-centralized` | 1；Tk `init.tcl` 缺失 |
| 修复后 `...run_centralized_smoke.py --workspace outputs/s3-preflight-centralized-fixed` | 0；loss `0.520146 -> 0.001460`，ADE `21.686918m`，checkpoint/metrics/图表齐全 |
| `.venv\\Scripts\\python.exe -m ruff check src tests scripts` | 0；All checks passed |
| `.venv\\Scripts\\python.exe -m pytest -q` | 0；`298 passed in 58.94s` |
| `.venv\\Scripts\\python.exe -m src.cli validate-config` | 0；配置有效（run `d2-smoke`） |
| `git diff --check`（候选前） | 0；仅对保留的 S1 文件报告 CRLF 警告，未暂存 |

## Centralized 页面 smoke（已补齐）

- 在 `http://localhost:8503` 打开真实 Streamlit 页；页面显示 S2 Centralized 控制台、允许
  Centralized、Local-only/Federated/compare 禁用，以及“页面不会自动启动训练”。
- 通过实际“执行受控操作”按钮启动 `outputs/ui-smoke-1789207190`；随后只读核对该 run 已有
  manifest、metrics JSON/CSV、`train.log`、best checkpoint、prediction/loss/baseline 图、
  processed split 与 RSU 产物。
- 在 `http://localhost:8504` 复验实际页面，点击“执行受控操作”后页面显示“操作完成，退出码 0”，
  受控日志显示 5 个 RSU、split、best epoch `29`、ADE `21.686918m`、FDE `24.571101m`、
  checkpoint、metrics 与 figures 路径和 loss `0.520146 -> 0.001460`。运行目录为
  `outputs/ui-smoke-1789208821`，其 manifest、metrics JSON/CSV、训练日志、checkpoint、图表
  和 processed artifact 均存在。
- 刷新同一页面后，受控命令台回到“尚未执行页面操作。页面不会自动启动训练”，且 outputs 下的
  `ui-smoke-*` 目录仍仅为先前两次显式点击产生的 `ui-smoke-1789207190` 与
  `ui-smoke-1789208821`；本次刷新没有生成第三次训练运行。

## 评审与批准

- AI 自评：Codex（GPT-5）审阅候选 `7769d3a`。确认 `Agg` 在 `pyplot` 前设置，且回归测试
  显式移除继承环境变量；未发现新增 P0/P1。此为自评，**不替代** 02 要求的独立 AI R1/R2。
- 真人批准：无；未创建 MR、未推送，故无平台链接或批准记录。
- 远端 CI：未核实（本机无 `gh`；GitHub REST API 匿名请求已限流）。

## 缺陷、下游接口与继续条件

- 已修复：无头环境的 Centralized 图表渲染。接口冻结影响：无；下游仍调用原有
  visualization 函数，CLI/JSON schema/配置签名不变。
- 阻塞：需在托管平台核实 PR #28/#29 及最新 `dev` 的 Quality Gate 和真实非作者批准；
  本修复仍需独立 AI 评审、真人批准和 `chore/s3-preflight` → `dev` MR，之后才能进入
  S3-A0 接口冻结。
- 不关闭既有保护规则；没有推送 `dev`/`main`，没有自动合并。
