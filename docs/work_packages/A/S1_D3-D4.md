# S1-A-01：数据准备入口与阶段集成

- 阶段：S1（D3—D4）
- 角色：A，项目负责人/架构与集成开发
- 分支：`feature/s1-a-prepare-data-cli`
- 主评审：D；协作：B、G
- 关联：F1、M1、M6、AT-01

## 目标

把冻结的数据配置接入 `prepare-data` CLI，形成可定位错误、可追踪产物的数据准备入口。不得实现 B 的 highD 解析逻辑。

## D3

- 生产代码：在 `src/cli.py` 和编排层接入 data config、输出根目录与退出码。
- 测试：配置缺失、输入目录缺失、成功/失败返回码。
- 建议提交：`feat(cli): wire prepare-data command [S1-A-01][F1]`。

## D4

- 生产/集成：串联 B 的 adapter、C 的 batching、D 的 partition 和 G 的 manifest。
- 测试/文档：增加最小数据冒烟；记录冻结接口冲突与 MS2 评审结论。
- 建议提交：`test(cli): add prepare-data smoke flow [S1-A-01][AT-01]`。

## 交付与验收

- `prepare-data` 成功返回 0，配置/数据错误返回非零且消息可定位。
- CLI 不包含清洗、滑窗或划分算法；调用公共接口完成编排。
- 产物含数据统计、split/manifest 路径；公共质量门禁通过。
- MR 目标 `dev`，附两日 Commit、测试和 D/B/G 评审证据。
