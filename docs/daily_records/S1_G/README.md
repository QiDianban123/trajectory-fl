# S1-G-01：RunContext、数据 Manifest 与复现基础

本分支将 RunContext 和数据 manifest 接入可复现运行目录。实现范围包括
配置快照、代码 SHA、数据文件 SHA-256、split/partition manifest、数据画像
和产物相对路径；不读取训练数据内容，也不实现训练或 highD 清洗。

## 已完成

- `RunContext` 使用唯一、可排序的 `run_id` 创建 `outputs/<run_id>/`，拒绝
  路径逃逸、重复运行、不可序列化配置和无效 Git SHA。
- 配置和 manifest 使用严格 JSON、排序键、UTF-8 和原子替换写入；返回的
  manifest 是深拷贝，调用方不能反向修改上下文。
- 数据文件路径必须位于 project root 内，记录 SHA-256；重复记录发现文件
  内容变化时失败。数据文件和 split/partition 条目稳定排序。
- manifest 记录 `data_version`、`split_id`、配置、数据文件校验值、split、
  D 的 `PartitionManifest.to_mapping()` 和运行产物相对路径。
- 将实验测试移至 `tests/unit`，并新增 D partition 到 G manifest 的集成测试。

## 验证与边界

目标分支已同步最新 `dev`。本地使用 Python 3.13 与 `MPLBACKEND=Agg` 验证：

```text
python -m pytest tests/unit/test_run_context.py tests/integration/test_run_context_manifest.py -q
22 passed
python -m pytest tests -q
200 passed
python -m ruff check src tests
All checks passed!
python -m src.cli validate-config
Configuration valid: run=d2-smoke mode=smoke seed=42
```

远端 CI 的 Ubuntu/Python 3.10 结果仍是合并前必需证据。

当前仓库仍没有 B 的 HighDAdapter 和 A 的 prepare-data 生产入口，因此本记录
不声称完成真实 highD 文件清单、split 生成或 A/B 输出路径联调。待依赖合入后，
应补充真实管线的文件 checksum、split_id、数据画像和产物路径验收。

## 合并要求

MR 目标为 `dev`，主评审 D，协作评审 A/B。必须先取得 CI 的 Ruff、pytest 和
配置校验结果，再合并；若 A/B 管线尚未合入，MR 应按 G 基础设施部分交付描述，
不能把当前契约测试当作 MS2 完成证据。
