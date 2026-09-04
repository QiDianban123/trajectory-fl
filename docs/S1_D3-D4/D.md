# S1-D-01：5-RSU Non-IID 划分

- 阶段：S1（D3—D4）
- 角色：D，联邦学习开发
- 分支：`feature/s1-d-noniid-partition`
- 主评审：A；第二评审：F
- 关联：F4、AT-04

## 目标

根据道路空间区域构造 5 个可解释 RSU 客户端，并保留可重建 partition manifest。

## D3

- 生产代码：实现 partition 配置、区域边界、分组索引和 `rsu_<NN>` ID。
- 测试：边界值、空区域、重复 group、非法客户端数。
- 建议提交：`feat(data): add rsu partition schema [S1-D-01][F4]`。

## D4

- 生产代码：实现空间 Non-IID 分配、相邻小区域合并、样本/车辆统计。
- 测试：客户端并集完整、交集符合规则、无跨 split 泄漏和稳定重建。
- 建议提交：`feat(federated): partition noniid clients [S1-D-01][AT-04]`。

## 交付与验收

- 输出 partition manifest、每客户端样本数/车辆数/空间范围。
- 不复制样本填充客户端；train 外数据不参与客户端训练划分。
- A/F 双评审通过。
