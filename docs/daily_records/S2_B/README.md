# S2-B-01：DataLoader、缓存与数据性能交付记录

日期：2026-09-10。分支：`feature/s2-b-dataloader-cache`。目标分支：`dev`。

## D5 交付

- 新增 processed cache identity：`data_version + split_id + semantic config digest`。
- `split_manifest.json` 索引三个 split 的 manifest、samples 和 scaler SHA-256。
- 新增 `ProcessedDatasetReader`，统一验证版本、split、配置、文件完整性、样本数量、
  window spec、metadata、统计和 train-only scaler。
- 读取器拒绝 legacy identity、空 split、损坏 NPZ、错误 dtype、路径逃逸和跨 split
  不一致；文件签名变化会使进程内缓存失效。

## D6 交付

- 新增 `DataLoaderConfig.from_config_bundle`，复用已有 batch size、worker 和 seed 配置。
- 新增 `create_dataloaders`，复用 S1-C collate 输出 `TrajectoryBatch`。
- train shuffle 在固定 seed 下可重复；validation/test 保序并始终保留尾批。
- `ProcessedDataBundle.inverse_transform` 提供保存的 train scaler 反变换 hook；没有在
  数据层实现 ADE/FDE。
- 生产 `prepare-data` 已写出新 cache identity，旧 processed 数据需重新执行数据准备。

## 接口与边界

新增公共接口：

```text
semantic_data_config(data_config)
semantic_config_digest(data_config)
processed_cache_key(data_version=..., split_id=..., data_config=...)
ProcessedDatasetReader.load(processed_dir, data_config=..., expected_*=...)
DataLoaderConfig.from_config_bundle(bundle)
create_dataloaders(data, contract=..., config=...)
ProcessedDataBundle.inverse_transform(values)
```

模型、Trainer、评价算法和训练循环没有修改。B 只依赖冻结的 `TrajectoryDataset`、
`ModelContract` 和 S1-C collate 边界。

## 验证记录

```text
实际 Commit SHA：d392c10（生产代码）、076e889（测试）
MR：待创建
Ruff：python -m ruff check src tests scripts，All checks passed
pytest：定向 24 passed；全量 234 passed
配置/冒烟检查：validate-config 通过；prepare-data → processed reader 的 0/1 worker 集成通过
评审结论：待 F 主评审、C 协作评审
遗留问题：旧 processed 数据没有 cache_identity，必须重新执行 prepare-data
下游交接：C/G/A 使用 ProcessedDatasetReader 和 create_dataloaders，不直接读取 NPZ
```

测试同时确认命中进程缓存时会返回新的 metadata/scaler 视图，调用方修改 metadata
不会污染后续读取；只读 NumPy 数组保持共享，兼顾缓存安全与 DataLoader worker 可序列化。
