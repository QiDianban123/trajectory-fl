# S1-F-01：数据质量与集成门禁

- 阶段：S1（D3—D4）
- 角色：F，质量与工程化开发
- 分支：`test/s1-f-data-quality`
- 主评审：B；协作：A、D
- 关联：M5、M6、AT-01、AT-04

## 目标

为真实数据管线建立正常、边界、异常和防泄漏自动化门禁。

## D3

- 测试/工具代码：扩展 fixtures，覆盖字段缺失、NaN/Inf、短轨迹、重复/乱序 frame。
- 工程：配置 CodeCheck/CI 执行 Ruff、pytest 和配置校验。
- 建议提交：`test(data): cover highd input failures [S1-F-01][AT-01]`。

## D4

- 测试代码：增加 split 交集、scaler 泄漏/可逆、路径错误和 prepare-data 集成测试。
- 记录：输出 MS2 测试报告和缺陷闭环。
- 建议提交：`test(integration): gate data pipeline [S1-F-01][AT-04]`。

## 交付与验收

- CI 与 README 本地命令一致；失败能定位到数据文件/字段/配置。
- AT-01、AT-04 有自动化证据；无静默丢样本。
- B/A 评审通过。
