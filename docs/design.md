# 系统设计与 Day 2 接口基线

**版本：** V1.0-D2-A<br>
**日期：** 2026-09-01<br>
**状态：** A 已建立架构基线；标记为“待确认”的模块细节由 B—F 评审后冻结。

## 1. 设计目标与范围

系统在同一数据划分、模型结构、初始权重和评价实现下支持 Centralized、Local-only、Federated 三种实验模式。当前基线只确定模块边界、数据契约、配置和产物格式，不实现真实 highD 读取、LSTM 训练或 FedAvg。

## 2. 分层架构与依赖方向

```text
CLI / scripts
    -> experiments (统一编排与结果保存)
        -> training (Trainer、集中式、本地独立)
        -> federated (Client、Server、Aggregator)
        -> evaluation (指标、图表、汇总)
    -> data (适配、清洗、切分、归一化、Dataset、客户端划分)

models  <- training / federated
utils   <- 所有层（配置、seed、日志、路径；不得反向依赖业务层）
```

依赖必须单向：模型层不得读取文件或依赖联邦层；指标层不得触发训练；数据层不得依赖模型；联邦层复用通用 Trainer，不复制训练循环。CLI 只解析参数并调用编排层，不能承载业务逻辑。

## 3. 标准数据与 Tensor 契约

### 3.1 单样本与批数据

| 字段 | 单样本 | 批数据 | dtype | 语义 |
|---|---|---|---|---|
| `history` | `[T_h, 2]` | `[B, T_h, 2]` | `float32` | 过去二维绝对位置 |
| `future` | `[T_f, 2]` | `[B, T_f, 2]` | `float32` | 未来二维绝对位置真值 |
| `meta` | `dict` | `list[dict]` 或等价 collate 结构 | — | 数据版本、recording/vehicle ID、frame 范围、split、client ID |

P0 不使用变长序列或 mask；样本必须满足固定 `T_h/T_f`、有限值和时间严格递增。默认 highD 采样率 25 Hz，`T_h=75`（3 秒）、`T_f=125`（5 秒）。B/C 需确认内存与训练预算后才能修改，修改必须同步数据和模型配置。

### 3.2 坐标、切分与归一化边界

- 原始和评价坐标使用 highD 道路局部二维物理坐标，单位米。
- 数据必须先按 `vehicle_id`（备选 `scenario_id`）切分 train/validation/test，再对各 split 构造滑窗。
- 标准化统计量只允许由 train split 拟合，并按 x/y 轴保存；validation/test 只调用 transform。
- 模型输入、标签和输出处于同一标准化坐标系；评价层必须先 inverse-transform，再在米制坐标计算 ADE/FDE。
- scaler 及其拟合数据版本属于运行产物；任何从 validation/test 拟合的配置均视为 P1 严重缺陷。

### 3.3 数据层接口与 metadata（D2-B 确认）

`src.data.adapters.DatasetAdapter` 是唯一的原始数据格式边界，冻结以下三个方法：

| 方法 | 输入 | 输出/约束 |
|---|---|---|
| `load_raw(source)` | 原始数据目录或文件路径 | 仅读取原始数据，不执行项目级清洗 |
| `preprocess(raw, config)` | 原始表与 data config | 应用字段检查、异常策略、切分和训练集 scaler |
| `build_samples(cleaned, config)` | 已清洗且已切分的数据 | 只从单一 split 构造固定长度 `TrajectorySample` |

`TrajectorySample` 固定为 `history`、`future`、`meta` 三字段。前两个字段必须是有限的 `float32 [T, 2]` 数组；`meta` 必须包含 `dataset_name`、`data_version`、`recording_id`、`vehicle_id`、history/future 首尾 frame、`split_id` 和 `split`。D4 才增加可选 `client_id`。

`TrajectoryDataset` 只容纳同一 `split`、`split_id` 和 `WindowSpec` 的样本，并在构造时验证 history/future 的长度。该容器不依赖 PyTorch；D4 可以在其外建立 DataLoader 包装，而不能放宽这些数据不变量。

### 3.4 数据防泄漏与异常契约（D2-B 确认）

- 先为每个 `vehicle_id`（备选 `scenario_id`）分配一个 split，再生成窗口；同一 group 出现在多个 split 时必须立即报错。
- frame 必须严格递增，重复/乱序 frame 按 `reject_track` 拒绝；history 最后一帧必须早于 future 第一帧。
- 缺失必需字段或非有限坐标按 `reject_sample` 拒绝，最短轨迹长度不足 `minimum_track_frames` 的记录不参与样本构造。
- `TrainingCoordinateScaler.fit(..., split="train")` 是唯一允许拟合统计量的入口；validation/test 只能调用 transform，评价层调用 inverse-transform。
- zero-variance 坐标轴使用缩放值 1，防止 NaN/Inf；原始 mean/scale 与 `split_id` 一起保存。

### 3.5 模型与通用训练契约（D2-C 确认）

P0 预测模型的 `ModelContract` 固定为：输入 `history: torch.float32 [B, 75, 2]`、输出 `pred_future: torch.float32 [B, 125, 2]`，均为归一化后的二维绝对位置。模型必须保持输出与输入在同一 device，拒绝空 batch、错误 shape、错误 dtype 和 NaN/Inf；没有安装 PyTorch 时，模型/训练入口必须提示安装 `requirements.txt`，而不是在导入阶段产生难以定位的错误。

模型只定义 `forward(history)` 与 `state_dict`/`load_state_dict` 序列化边界：不得读取文件、创建优化器、移动 batch 或计算指标。Trainer 是唯一可以创建优化器、设置 `train/eval`、移动 model/batch 和调用反向传播的组件。

所有模式共享 `Trainer.fit(model, train_batches, validation_batches, initial_state=...)` 与 `Trainer.evaluate(model, batches)`。Centralized、Local-only 和未来 Federated 均只能委托该接口，不能复制训练循环。实验编排器以运行 seed 生成唯一初始 `state_dict`，保存该快照并向每种模式传入独立副本；Local-only 客户端之间不得共享训练后的参数。

`FitResult` 返回每个 epoch 的 sample count、train/validation loss、best epoch 和 checkpoint payload；`EvaluationResult` 返回 sample count 与 loss。Checkpoint 必须包含 `schema_version`、`model_state`、`model_config`、`seed`、`epoch`、`split_id` 和 `metrics`，并由结果层与运行配置共同保存。

## 4. 配置层级与校验

配置文件分为三层，均包含 `schema_version: 1`：

1. `configs/data.yaml`：数据来源、列契约、坐标单位、序列长度、先分组切分策略、归一化与异常处理边界。
2. `configs/model.yaml`：模型结构、Tensor dtype/设备责任、统一初始化和 checkpoint 默认约束。
3. `configs/experiments/*.yaml`：run 名称、模式、seed、输出根目录、对数据/模型配置的引用和运行设备。

加载顺序为 data -> model -> experiment。`src.utils.config.validate_config_bundle()` 必须拒绝缺文件、空 YAML、根节点非 mapping、未知 schema、非法范围，以及 data/model 的序列长度或坐标维度不一致。正式实验不得在代码中覆盖已保存的配置副本。

## 5. CLI 契约

| 子命令 | D2 行为 | 后续责任 |
|---|---|---|
| `status` | 输出当前基线状态 | A/F 维护 |
| `validate-config` | 加载并校验 data/model/experiment 配置；失败返回 2 | A/F |
| `prepare-data` | 明确返回“尚未实现”和退出码 2 | B/A，D3 |
| `train` | 明确返回“尚未实现”和退出码 2 | A/C/D，D6—D10 |
| `compare` | 明确返回“尚未实现”和退出码 2 | E/A，D11—D13 |

CLI 成功返回 0，可定位的用户配置/未实现错误返回 2；内部不可恢复异常不得伪装为成功。

## 6. 输出目录与可复现约定

每次实际运行写入 `outputs/<run_id>/`，其中 `run_id` 必须唯一且可排序。最低产物如下：

```text
outputs/<run_id>/
├── config.yaml          # 完整解析后的配置副本
├── metadata.json        # run_id、代码 SHA、数据版本、seed、split_id、环境
├── train.log
├── metrics.json
├── metrics.csv
├── checkpoints/
│   └── best.pt
└── figures/
```

失败运行保留配置、元数据、日志和失败原因，不覆盖旧目录。原始数据、processed 数据、outputs 和检查点均不得提交 Git。

## 7. 统一结果 schema

### 7.1 `metrics.json`

```json
{
  "schema_version": 1,
  "run_id": "20260901T120000Z-centralized-seed42",
  "status": "completed",
  "mode": "centralized",
  "seed": 42,
  "split_id": "highd-<digest>",
  "code_sha": "<git-sha>",
  "dataset": "highd",
  "model": "lstm_encoder_decoder",
  "coordinate_unit": "meter",
  "sample_count": 0,
  "metrics": {"ade": 0.0, "fde": 0.0},
  "timing_seconds": {"total": 0.0},
  "artifacts": {"checkpoint": null, "figures": []},
  "error": null
}
```

Local-only 可在顶层增加 `clients` 列表，但全局/宏平均与加权平均仍使用相同字段名并明确 `aggregation`。Federated 可增加 `rounds` 列表，但顶层最终指标格式不变。

### 7.2 `metrics.csv`

每行代表一个可比较评价结果，固定列为：

```text
schema_version,run_id,status,mode,seed,split_id,code_sha,dataset,model,
scope,client_id,aggregation,sample_count,coordinate_unit,ade,fde,total_seconds,error
```

`scope` 取 `global` 或 `client`；非客户端行的 `client_id` 为空。JSON 是完整事实源，CSV 是扁平汇总视图，二者必须由同一记录对象生成。

## 8. 待 B—F 确认的接口

- B：highD 字段映射和 scaler 的最终落盘格式仍待 D3 实际数据探查；`meta`、split 安全和训练集 scaler 契约已确认。
- C：绝对位置预测、Trainer 返回结构、float32/device 责任和 checkpoint envelope 已确认；D5 再实现 LSTM。
- D：客户端更新校验、非浮点 buffer 策略、联邦轮次统计结构。
- E：Local-only 汇总字段、图表函数签名和 JSON/CSV 一致性实现。
- F：`run_id` 生成、日志格式、路径安全、seed 覆盖范围和共享测试夹具。
