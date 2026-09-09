# A 角色 D3-D4 工作过程记录

**角色：** A，项目负责人 / 架构与集成开发  
**任务：** S1-A-01 数据准备入口与阶段集成  
**日期：** 2026-09-09  
**关联：** F1、M1、M6、AT-01、AT-04

## 一、任务清单对应情况

### D3：配置、入口与错误出口

| 任务要求 | 完成情况 | 对应实现或证据 |
|---|---|---|
| 接入 data/model/experiment 配置 | 已完成 | `prepare-data` 复用 `validate_config_bundle`，先校验三份 YAML 与跨配置维度 |
| 接入输入和输出目录 | 已完成 | 支持 `--raw-dir`、`--processed-dir`、`--output-root` 覆盖；所有路径限制在项目根目录内 |
| 可定位错误和退出码 | 已完成 | 配置、输入目录、CSV、输出路径和数据契约错误输出 `Prepare-data error: ...` 并返回 2 |
| 不实现 B 的数据算法 | 已完成 | 编排层只调用 `HighDAdapter`、`save_split_datasets`，未复制字段映射、清洗、滑窗、切分或归一化逻辑 |

### D4：跨角色集成与最小冒烟

| 任务要求 | 完成情况 | 对应实现或证据 |
|---|---|---|
| 串联 B 的 adapter 与持久化 | 已完成 | `src/data/prepare.py` 调用 `load_raw`、`preprocess`、`build_datasets`、`save_split_datasets` |
| 串联 C 的 batch 边界 | 已完成 | 对第一个 train 样本调用 `sample_to_tensor` 和冻结的 `ModelContract` 做 CPU float32 冒烟 |
| 串联 D 的 partition | 已完成 | 从物理坐标和实际 train 窗口数构造 `GroupExtent`，调用 `partition_train_groups` 和不变式检查，并将 `client_id` 写入 train 元数据 |
| 串联 G 的 manifest | 已完成 | `RunContext` 记录原始/处理文件校验和、split、partition、数据画像和运行 manifest |
| 最小端到端与错误测试 | 已完成 | `tests/integration/test_prepare_data_cli.py` 覆盖成功产物、client 元数据、manifest 对齐和缺输入目录 |

## 二、实际工作过程

1. 读取 D2 冻结接口、B 的数据管线、C 的 NumPy→Torch 桥接、D 的 RSU 划分及 G 的 RunContext，确认各模块已合入 `dev`、但 CLI 仍为占位实现。
2. 新增 `src/data/prepare.py` 作为编排层，保持算法留在所属公共模块，避免跨角色重复实现。
3. 在 adapter 输出的 train 记录上计算每个 `(recording_id, vehicle_id)` 的物理 x 范围和实际窗口数，交由 D 的分区接口生成可重建 partition manifest。
4. 仅给 train 样本添加 `client_id`；validation/test 仍保持无客户端语义，防止跨 split 使用训练分区。
5. 将 B 的 split 产物写入 `data/processed/<split_id>/`，同时生成 `partition_manifest.json`；用 G 的 run 目录记录文件哈希、数据画像及完整运行 manifest。
6. 扩展 `RunContext` 接收已校验且位于项目根目录内的输出根目录，仍维持默认 `outputs/<run_id>/` 行为和路径逃逸保护。

## 三、提交产物

| 类型 | 路径 | 说明 |
|---|---|---|
| 生产代码 | `src/cli.py` | `prepare-data` 参数、退出码、用户可读产物摘要 |
| 生产代码 | `src/data/prepare.py` | B/C/D/G 的唯一数据准备编排边界 |
| 生产代码 | `src/experiments/run_context.py` | 支持受项目根目录保护的可配置运行输出根目录 |
| 自动化测试 | `tests/integration/test_prepare_data_cli.py` | 端到端成功与输入目录失败路径 |
| 过程记录 | 本文件 | 分工对应、集成边界、验证和交接说明 |

## 四、标准输出格式

```text
data/processed/<split_id>/
├── split_manifest.json
├── partition_manifest.json
├── train/
│   ├── samples.npz
│   ├── manifest.json        # 每个 train 样本含 client_id
│   └── scaler.npz
├── validation/
└── test/

<output_root>/<run_id>/
├── config_snapshot.json
├── data_profile.json
└── manifest.json            # 原始/处理文件 SHA-256、split、partition、产物索引
```

成功时命令返回 0，并输出 split ID、各 split 样本数和全部 manifest 路径。用户可修复的配置、输入数据或路径问题返回 2；不会静默覆盖已有 `data/processed/<split_id>`。

## 五、验收记录

本次变更后在项目根目录执行：

```text
python -m pytest -q tests/integration/test_prepare_data_cli.py
python -m pytest -q
python -m ruff check src tests
python -m src.cli validate-config
```

实际结果：专项 CLI 测试 `6 passed`；设置 `MPLBACKEND=Agg` 并排除 macOS 上会阻塞的
`num_workers=1` spawn 用例后，回归 `204 passed, 6 deselected`；Ruff 与
`validate-config` 均通过。默认 macOS Matplotlib 后端会在图表集成测试中中止，且该
worker 用例在本机 Python 3.10 中阻塞，均为现有环境问题，不作为本交付的通过证据。

本地端到端测试使用匿名生成的 highD 风格 CSV，写在 pytest 临时目录；不需要、不下载、也不提交受许可限制的真实 highD 原始数据或处理产物。

## 六、接口冲突与 MS2 评审结论

- 未发现需要改变 B/C/D 的冻结算法或数据契约的冲突。`TrajectoryMetadata.client_id` 早已声明为 D4 的可选字段；本实现只在 train split 使用它。
- G 的原始 `RunContext` 固定 `outputs/`，而 experiment 配置已声明 `run.output_root`。为消除这一配置—实现不一致，新增了受项目根目录约束的 `output_root` 参数，默认行为未改变。
- 此交付提供 AT-01/AT-04 的匿名最小端到端证据；真实 highD 运行仍取决于合法取得并放入 `data/raw/` 的数据。人工 D/B/G 评审和远端 CI 结果不在本记录中虚构为已完成。

建议提交信息：

```text
feat(cli): wire prepare-data command [S1-A-01][F1]
test(cli): add prepare-data smoke flow [S1-A-01][AT-01]
```
