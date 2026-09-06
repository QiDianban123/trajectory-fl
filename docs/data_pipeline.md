# highD 数据管线说明

**负责人：** B（数据工程开发）  
**阶段：** S1 / D3-D4  
**状态：** 已完成最小可验证实现

## 处理流程

1. `HighDAdapter.load_raw` 读取单个 CSV 或目录下的 `*_tracks.csv`，并自动补充 `recording_id`。
2. 字段映射支持 highD 常见表头及其下划线/空格变体：`id`、`frame`、`x`、`y`。
3. `preprocess` 按 `(recording_id, vehicle_id)` 检查缺失、非有限值、重复 frame 和最短轨迹长度。坏轨迹整条拒绝，并在 `stats` 中记录输入行数、拒绝轨迹/行数和 split 计数。
4. 使用固定 seed 对车辆组先分配 train/validation/test，再构造窗口；同一车辆不会跨 split。
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

`split_manifest.json` 是全量划分索引，记录 `data_version`、`split_id`、三个 split 的样本数、处理统计和 scaler 位置。`manifest.json` 记录单 split 的样本元数据，`load_dataset` 可据此重建 Dataset、scaler 和统计信息。NPZ 不使用 pickle。

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
