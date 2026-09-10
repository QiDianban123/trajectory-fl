# S1-B 数据管线评审修复

基线：`dev` `82b7cde`。分支已同步该基线后修复以下问题：

- 乱序 frame 不再被静默排序，整条轨迹以 `out_of_order_frame` 原因拒绝。
- `split_id` 现在包含 `data_version`、排序后的分组分配和规范化 split 配置；数据
  内容变化不会复用旧的划分 ID。
- 处理统计记录缺失/非数值、非有限、非整数、重复、乱序和短轨迹等拒绝原因。
- 保存前校验 train scaler、数据统计、三个 Dataset 的 split、split ID 与窗口规格；
  读取时校验缓存 scaler 形状、有限性与正尺度。JSON/NPZ 使用临时文件替换写入。
- 新增乱序、split ID 碰撞、缺字段、非整数 frame、短轨迹、小样例三 split 分配、
  无效/损坏 scaler、split bundle 不一致等回归测试。

本地验证：S1-B 定向测试 `25 passed`；全量 `208 passed`；Ruff 和
`python -m src.cli validate-config` 通过。远端 CI 仍是合并前门禁。
