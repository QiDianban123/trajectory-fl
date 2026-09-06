# B 角色 D3-D4 工作过程记录

**角色：** B，数据工程开发  
**任务：** S1-B-01 highD 数据管线  
**日期：** 2026-09-06  
**关联：** F1、F4、AT-01、AT-04

## 一、任务清单对应情况

### D3：读取与清洗

| B.md 要求 | 完成情况 | 对应实现或证据 |
|---|---|---|
| 实现 `HighDAdapter.load_raw` | 已完成 | [src/data/adapters.py](../../src/data/adapters.py)；支持单文件、目录和 `*_tracks.csv` |
| highD 字段映射 | 已完成 | 支持 `id`、`Track ID`、`frame`、`Frame ID`、`x Position`、`y Position` 等变体 |
| 必需列检查 | 已完成 | 缺少 `id`、`frame`、`x`、`y` 时明确报错 |
| 缺失值、非有限值处理 | 已完成 | 整条轨迹拒绝，并记录拒绝轨迹数和行数 |
| 重复 frame 处理 | 已完成 | 整条轨迹拒绝 |
| 乱序 frame 处理 | 已完成 | 轨迹按 frame 排序后再生成窗口 |
| 统计信息 | 已完成 | `input_rows`、`valid_rows`、`rejected_tracks`、`rejected_rows`、split 计数 |
| 正常表头、缺列、乱序、重复、NaN/Inf、短轨迹测试 | 基础专项测试已完成 | [tests/unit/test_highd_adapter.py](../../tests/unit/test_highd_adapter.py) |

### D4：切分、归一化与持久化

| B.md 要求 | 完成情况 | 对应实现或证据 |
|---|---|---|
| vehicle-first split | 已完成 | 以 `(recording_id, vehicle_id)` 为 group，先切分后滑窗 |
| 防止车辆跨 split | 已完成 | 使用 `validate_split_assignments` 校验 group 独立性 |
| train-only scaler | 已完成 | 仅 train 坐标拟合 `TrainingCoordinateScaler` |
| scaler 保存与逆变换 | 已完成 | 保存 `mean`、`scale`，支持 `inverse_transform` |
| 固定长度滑窗 | 已完成 | 生成 `TrajectorySample`，满足 history/future shape 和时间顺序 |
| Dataset 持久化与重建 | 已完成 | `save_dataset`、`load_dataset` |
| split manifest | 已完成 | `save_split_datasets` 生成 `split_manifest.json` |
| 字段与 scaler 格式记录 | 已完成 | [docs/data_pipeline.md](../data_pipeline.md) |

## 二、实际工作过程

1. 阅读 D2 冻结的数据接口，确认不能改变 `DatasetAdapter`、`TrajectorySample`、`TrajectoryDataset` 和 `WindowSpec` 的公共约定。
2. 检查数据目录，确认仓库不包含受限 highD 原始数据，因此使用匿名小样例 DataFrame 验证，不下载或复制原始大数据。
3. 在 `HighDAdapter.load_raw` 中实现 CSV 文件发现、读取、recording 标识补充和字段别名映射。
4. 在 `HighDAdapter.preprocess` 中执行数值转换、缺失/非有限检查、重复 frame 检查、轨迹排序和最短长度检查。
5. 使用配置中的 seed 对车辆 group 进行 train/validation/test 划分，并在滑窗之前写入 split 信息。
6. 用 train split 的物理坐标拟合 scaler；validation/test 仅使用同一 scaler 转换，避免数据泄漏。
7. 在 `build_samples` 中生成标准样本，并将数据版本、split_id、车辆 ID、frame 范围等信息写入 meta。
8. 在 `build_datasets` 中构造三个 split 的 `TrajectoryDataset`，在 `save_dataset` 和 `save_split_datasets` 中保存样本、scaler、统计和 manifest。
9. 使用专项测试验证样本 shape、时间顺序、split、scaler 可逆性和持久化重建，再执行全量回归。


## 三、提交产物

| 类型 | 路径 | 说明 |
|---|---|---|
| 生产代码 | [src/data/adapters.py](../../src/data/adapters.py) | highD 读取、字段映射、清洗、split、窗口 |
| 生产代码 | [src/data/dataset.py](../../src/data/dataset.py) | Dataset 和 NPZ/JSON 持久化/重建 |
| 自动化测试 | [tests/unit/test_highd_adapter.py](../../tests/unit/test_highd_adapter.py) | B 角色 D3-D4 专项验收测试 |
| 技术文档 | [docs/data_pipeline.md](../data_pipeline.md) | 流程、字段、产物格式和验收命令 |
| 过程记录 | 本文件 | 任务对应、工作过程、验收和提交边界 |
| 状态更新 | [docs/data_selection.md](../data_selection.md) | 记录字段探查和实现状态 |

## 四、标准输出格式

```text
data/processed/<split_id>/
├── split_manifest.json
├── train/
│   ├── samples.npz
│   ├── manifest.json
│   └── scaler.npz
├── validation/
│   ├── samples.npz
│   ├── manifest.json
│   └── scaler.npz
└── test/
	├── samples.npz
	├── manifest.json
	└── scaler.npz
```

- `samples.npz`：`history[N, T_h, 2]` 和 `future[N, T_f, 2]`。
- `manifest.json`：单个 split 的样本 meta、窗口参数、统计和 split_id。
- `scaler.npz`：train-only 的 `mean`、`scale`。
- `split_manifest.json`：全量 split 的 data_version、split_id、样本数、处理统计和 scaler 路径。

## 五、验收记录

在项目根目录 `trajectory-fl-main` 执行：

```text
python -m pytest -q tests/unit/test_data_contracts.py tests/unit/test_highd_adapter.py
结果：8 passed

python -m ruff check src/data/adapters.py src/data/dataset.py tests/unit/test_highd_adapter.py
结果：All checks passed
```

完整回归还应执行：

```text
python -m pytest -q
python -m ruff check src tests
python -m src.cli validate-config
```

## 六、提交边界与后续事项

应提交生产代码、自动化测试、技术说明、工作记录和状态记录。以下内容不得提交：

- `data/raw/` 中的原始 highD 文件；
- `data/processed/` 中的生成数据；
- `.venv/`、`.pytest-tmp/`、`.pytest_cache/`、`__pycache__/`；
- 模型检查点和个人设备信息。

当前验证使用匿名小样例，仓库没有真实 highD 文件，因此尚未生成真实数据对应的 processed 产物。取得合法数据后，使用相同 API 运行即可生成上述目录结构；真实数据探查命令为 `python scripts/probe_highd.py data/raw`。

建议提交信息：

```text
feat(data): build split-safe highd samples [S1-B-01][AT-01]
```
