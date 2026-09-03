# S1-B-01：highD 数据管线

- 阶段：S1（D3—D4）
- 角色：B，数据工程开发
- 分支：`feature/s1-b-highd-pipeline`
- 主评审：F；第二评审：D
- 关联：F1、F4、AT-01、AT-04

## 目标

在合法小样例上实现 highD 读取、清洗、先分组切分后滑窗、train-only scaler 与可重建中间数据。

## D3

- 生产代码：实现 `HighDAdapter.load_raw/preprocess`、字段映射、必需列、缺失/非有限/重复 frame 策略和统计。
- 测试：正常表头、缺列、乱序、重复、NaN/Inf 和短轨迹。
- 建议提交：`feat(data): implement highd adapter [S1-B-01][F1]`。

## D4

- 生产代码：实现 vehicle-first split、scaler 拟合/保存、滑窗、Dataset 持久化和缓存。
- 测试/文档：无跨 split 车辆、scaler 可逆、窗口 shape/顺序；记录字段与 scaler 格式。
- 建议提交：`feat(data): build split-safe samples [S1-B-01][AT-01]`。

## 交付与验收

- 输出标准 `TrajectorySample`、处理统计、scaler 和 split manifest。
- 同一 vehicle/scenario 不跨 split；validation/test 不参与 scaler 拟合。
- 不下载或提交受限原始大数据；B/F/D 评审通过。
