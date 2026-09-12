# S2-UI-01：Centralized 控制台初始交付

- 分支：`feature/s2-ui-centralized-console`
- 基线：`origin/dev=ca4cfd9`（S2-F 已合入）
- 框架：Streamlit `1.63.0`
- 人工评审：待 MR 记录

## 当前能力

- 顶部阶段、后端和运行目录状态栏。
- S2 快捷操作：Centralized smoke、Centralized train；Local-only、Federated、compare
  明确禁用并显示 S3/S4 原因。
- 参数数组命令预览，禁止任意 executable、shell 字符串、路径逃逸和重复 run。
- 受控命令台显示 stdout/stderr、退出码和时间。
- 只读扫描 outputs 下的 RunContext manifest、ResultRecord、metrics、training history、
  figures 和其他 artifact。
- ADE/FDE、best epoch、sample count、耗时、split_id、data_version、loss/trajectory 图展示。

## 验证

- UI capability/command/index 定向测试：`5 passed`（此前执行；本次平台额度限制后续
  Python 复跑被拦截）。
- Streamlit 系统 smoke 测试已加入 `tests/system/test_ui_smoke.py`，用于启动独立子进程并
  检查页面 HTTP 200；当前运行记录以非浏览器 HTTP 检查为准。
- Ruff 和 py_compile 通过；Streamlit 本地 URL 返回 HTTP `200`。
- 页面不触发训练，只有用户点击白名单操作后才启动现有 CLI。

## 限制与下游

- S2 页面不提供 Local-only/Federated/compare、删除、覆盖、Git 操作或任意终端。
- 浏览器自动化辅助在当前环境加载失败，页面以本地 HTTP 健康检查为证据。
- S3 增加三模式公平性 guard、客户端面板和联邦时间线；S4 增加正式矩阵和离线演示包。

## 当前提交

实现提交：`b9b1afc feat(ui): add centralized experiment console [S2-UI-01]`。
