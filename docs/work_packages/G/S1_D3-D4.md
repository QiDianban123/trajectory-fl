# S1-G-01：RunContext、数据 Manifest 与复现基础

- 阶段：S1（D3—D4）
- 角色：G，实验运行与复现开发
- 分支：`feature/s1-g-run-context-manifest`
- 主评审：D；协作：A、B
- 关联：F1、M2、M6、AT-01

## 目标

以生产代码创建 `src/experiments/`，为后续每次数据准备和实验保存统一上下文、配置快照、代码 SHA 与数据 manifest。G 不做纯文档角色。

## D3

- 生产代码：实现 `RunContext`、run_id、代码 SHA、配置快照和 manifest 数据结构。
- 测试：非法 run_id、缺 SHA、路径逃逸、重复 run 和配置不可序列化。
- 建议提交：`feat(experiments): add run context [S1-G-01][M2]`。

## D4

- 生产代码：实现数据文件 checksum、split/partition manifest 和数据画像 JSON 导出。
- 测试/联调：确定性排序、文件变化检测、与 A/B 输出路径对接。
- 建议提交：`feat(experiments): add data manifest [S1-G-01][AT-01]`。

## 交付与验收

- 新增 `src/experiments/` 及对应单测，生产代码与测试提交均可核查。
- manifest 记录配置、SHA、数据版本、split_id、文件校验值和产物相对路径。
- 不读取训练数据内容或实现训练；D/A/B 评审通过。
