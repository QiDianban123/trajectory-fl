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

## 4. 配置层级与校验

配置文件分为三层，均包含 `schema_version: 1`：

1. `configs/data.yaml`：数据来源、列契约、序列长度、先分组切分策略、归一化边界。
2. `configs/model.yaml`：模型结构、Tensor 维度和最小训练默认值。
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

- B：highD 字段映射、`meta` 必需键、split 清单和 scaler 序列化格式。
- C：绝对位置或位移预测的最终选择、Trainer 返回结构、device/dtype 细节。
- D：客户端更新校验、非浮点 buffer 策略、联邦轮次统计结构。
- E：Local-only 汇总字段、图表函数签名和 JSON/CSV 一致性实现。
- F：`run_id` 生成、日志格式、路径安全、seed 覆盖范围和共享测试夹具。
