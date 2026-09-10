# highD 数据管线说明

**负责人：** B（数据工程开发）
**阶段：** S1 / D3-D4
**状态：** 已完成最小可验证实现

## 处理流程

1. `HighDAdapter.load_raw` 读取单个 CSV 或目录下的 `*_tracks.csv`，并自动补充 `recording_id`。
2. 字段映射支持 highD 常见表头及其下划线/空格变体：`id`、`frame`、`x`、`y`。
3. `preprocess` 按 `(recording_id, vehicle_id)` 检查缺失、非有限值、非整数 ID/frame、重复/乱序 frame 和最短轨迹长度。坏轨迹整条拒绝，并在 `stats` 中记录输入行数、拒绝轨迹/行数、拒绝原因和 split 计数。
4. 使用固定 seed 对车辆组先分配 train/validation/test，再构造窗口；同一车辆不会跨 split。`split_id` 由数据版本、规范化 split 配置和分组分配共同生成。
5. 只用 train split 的物理坐标拟合 `TrainingCoordinateScaler`。validation/test 只调用 `transform`。
6. `build_samples` 生成标准 `TrajectorySample`：`history[T_h, 2]`、`future[T_f, 2]`，均为有限 `float32` 标准化坐标，并携带数据版本、车辆、frame 范围和 split 元数据。
7. `build_datasets` 生成三个经过 `TrajectoryDataset` 校验的 split；`save_split_datasets` 保存可重建产物。

## 持久化格式

```text
processed/<split_id>/
├── split_manifest.json
├── train/
│   ├── samples.npz       # history[N, T_h, 2]、future[N, T_f, 2]
│   ├── manifest.json     # 元数据、窗口、统计、split_id
│   └── scaler.npz        # mean、scale；由 train 拟合
├── validation/
│   ├── samples.npz
│   ├── manifest.json
│   └── scaler.npz
└── test/
    ├── samples.npz
    ├── manifest.json
    └── scaler.npz
```

`split_manifest.json` 是全量划分索引，记录 `data_version`、`split_id`、三个 split 的样本数、处理统计和 scaler 位置。保存前会校验三个 Dataset 的 split、split_id 与窗口规格一致；单 split 的 scaler 也必须是有限、正尺度的 train scaler。`manifest.json` 记录单 split 的样本元数据，`load_dataset` 会重新校验缓存中的 scaler。NPZ 不使用 pickle。

## S2 processed reader 与 DataLoader

`ProcessedDatasetReader.load` 是训练侧读取完整 processed 数据的入口。调用者必须提供
当前已校验的 data config；读取器使用 `data_version`、`split_id` 和数据语义配置摘要
核对 `cache_identity`。语义配置包含数据集坐标契约、sequence、split、partition、
normalization 和 preprocessing，但不包含 raw/processed 等机器路径。

写入的 `split_manifest.json` 同时记录每个 split 的 `manifest.json`、`samples.npz` 和
`scaler.npz` 的 SHA-256。读取时会拒绝以下情况：

- S1 旧格式或缺少 cache identity；
- 期望的 data version / split ID 与产物不符；
- 配置摘要不符、文件缺失或校验值不符；
- 空 split、错误 dtype/shape、metadata 身份不一致；
- 三个 split 的窗口、统计或 train scaler 不一致；
- manifest 中的绝对路径或目录逃逸。

读取器只在当前进程缓存已验证且只读的样本数组。每次命中前比较父 manifest 和全部
关键文件的大小及修改时间，文件变化后重新校验，不复用旧对象。

`create_dataloaders` 使用现有 `collate_trajectory_samples` 构造 CPU float32
`TrajectoryBatch`。train 可按运行 seed 确定性 shuffle；validation/test 始终保持
顺序且绝不丢弃尾批。batch size 来自 model config，worker 数和 seed 来自 experiment
config。`ProcessedDataBundle.inverse_transform` 提供 train-only scaler 的只读反变换入口，
供后续评价适配器在米制坐标计算指标。

## 最小调用示例

```python
from src.data.adapters import HighDAdapter
from src.data.dataset import save_split_datasets

adapter = HighDAdapter()
raw = adapter.load_raw("data/raw")
config = ...  # validated configs/data.yaml mapping
cleaned = adapter.preprocess(raw, config)
datasets = adapter.build_datasets(cleaned, config)
save_split_datasets(
    datasets,
    "data/processed/<split_id>",
    scaler=cleaned["scaler"],
    stats=cleaned["stats"],
    data_version=cleaned["data_version"],
)
```

真实 highD 原始文件需要使用者按许可自行放入 `data/raw/`，不进入 Git；处理产物和缓存也不提交。当前仓库的专项测试使用匿名内存 DataFrame，不依赖受限原始数据。

## 验收命令

```powershell
.\\.venv\\Scripts\\python.exe -m pytest -q tests/unit/test_data_contracts.py tests/unit/test_highd_adapter.py
.\\.venv\\Scripts\\python.exe -m ruff check src/data/adapters.py src/data/dataset.py tests/unit/test_highd_adapter.py
```
