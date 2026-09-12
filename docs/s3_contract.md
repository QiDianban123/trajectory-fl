# S3 三模式增量接口冻结草案（S3-A-01 / A0）

状态：**候选草案，未批准、未实现**。日期：2026-09-12（Asia/Shanghai）。
基线：`origin/dev` `4c8c5e127ac699dd653cb2671f9070a021f92032`。
本文件为 B → C → E → D → G → A 的唯一增量边界；不替代 D2 已冻结接口，且不能据此
启用 UI 的 Local-only、Federated 或 compare 功能。D 主审、F 测试视角、C/G 协作和原规则的
真人确认完成前，所有 `Proposed` 签名均不得视为生产承诺。

## 1. 版本、兼容与不变条件

现有 D2/S2 的 `TrajectoryDataset`、`ProcessedDatasetReader`、`TorchTrainer`、
`ClientUpdate`、`AggregationRequest`、`ResultRecord(schema_version=1)`、
`CentralizedExperiment`、`src.cli train --mode centralized` 保持兼容。S3 不重写训练循环、
FedAvg 或 ADE/FDE；旧 Centralized JSON/CSV 继续可读。

S3 新建的三模式 manifest 采用 `schema_version: 2`，嵌入现有 ResultRecord v1 作为最终
`summary`；读取端必须同时接受旧 v1 单模式运行和 v2 manifest。新增字段缺失时，旧 v1 运行
只可展示为 Centralized 历史记录，不能参与 S3 比较。任何迁移只新增兼容导出/读取器，禁止
让 `models` 依赖 `federated`，禁止静默改写旧 artifact。

所有三模式比较必须严格相等：`data_version`、`split_id`、`partition_id`、train-scaler
identity、`model_config_digest`、`seed`、`initial_state_id`、`metric_schema`，以及下列
预算策略。任一字段不等、缺失或无法重建，runner/CLI 返回 2、写 failed manifest，UI 禁用
执行；不以“配置名字相同”代替摘要核验。

## 2. 数据、holdout 与计数口径（B 责任）

`partition_manifest.json` 仍由 train groups 建立；其稳定摘要为 `partition_id`。B 必须增加
`client_split_manifest.json`（Proposed，schema 1），每个 `client_id` 固定列出
`train_sample_count`、`validation_sample_count`、`test_sample_count`、`group_ids`、空间范围和
画像。`group_id=(recording_id, vehicle_id)` 的归属只由 **train** partition 的 group → client
映射确定；同一 group 的 validation/test window 跟随该归属，绝不使用 holdout 坐标再次分区，
绝不拟合或替换 train scaler。

若 holdout group 在 train 映射中不存在，或 manifest 有重复/未知 group，B 必须在构建 loader
前以 `ClientDataError` 拒绝整个三模式请求（CLI 退出 2），而非临时把它放到邻近 RSU。没有
train window 的 client 是 `empty_train`：不创建 ClientUpdate、不参加 Local-only 成功汇总或
FedAvg；其 holdout 数量及排除原因仍写 manifest。默认范围不支持重采样、`drop_last=True`、
提前停止改变训练计数；这些配置必须拒绝，待后续有明确的访问计数策略才可开放。

计数不可混用：

| 名称 | 精确定义 | 用途 |
|---|---|---|
| `train_sample_count` / `ClientUpdate.sample_count` | 一个完整本地 epoch 实际处理的唯一 train windows，必须每 epoch 相同 | FedAvg 权重 |
| `sample_visits` | `train_sample_count × completed_local_epochs`，仅限无重采样、无丢尾、无提前停止 | 公平预算/审计 |
| `evaluation_sample_count` | 指定 evaluation split 实际预测的 windows | ADE/FDE 加权 |
| `batch_count` | 实际 DataLoader batch 数 | 诊断，禁止作权重或指标分母 |

例如 N=5、batch=2、local_epochs=3 时，权重数为 5、访问量为 15、batch 数为 3；三者必须同时
记录但不互换。评价统一为每客户端 test；若 test 为空则记录 `empty_evaluation` failure，不将
validation 替代 test。汇总必须同时保存 macro（成功且有评价的客户端等权）和 weighted
（以 `evaluation_sample_count` 加权）；三模式总表默认展示 weighted，并标注分母和成功集合。

## 3. 状态、本地训练与 FedAvg（C/D 责任）

`initial_state_id` 是 `model_state_id(snapshot_model_state(initial_state).state)`；初始 state
由 A/G 在三模式 dispatch 前用同一 model config、seed 生成一次并以深拷贝交给每个模式。C 的
Proposed 公共工厂为：

```text
create_model_from_config(model_config) -> TrajectoryPredictor
build_isolated_client_trainer(client_id, *, seed, split_id, local_epochs) -> TorchTrainer
```

它每次返回新模型、新 optimizer（由 Trainer.fit 内创建）和独立 RNG 作用域；不得共享 parameter
storage、optimizer 或训练后 state。`local_epochs` 仅允许正整数，映射为该 client 的
`TorchTrainerConfig.epochs`，不会修改共享 `configs/model.yaml`。

本阶段 Local-only 和 Federated 的 `ClientUpdate.state` **冻结为最后完成 epoch 的 model state**，
不采用现有 `fit_result_to_client_update()` 的 best checkpoint 行为；理由是 FedAvg 的本地步数
必须代表实际完成的 local epochs。Centralized 保持现有 best checkpoint/恢复行为不变。C/D 必须
新增显式 adapter（建议 `fit_result_to_last_epoch_client_update`），保留现有函数及其 best-state
测试以保持 S2 兼容；不得悄悄改变它的语义。

浮点 state 的 r 轮聚合为

```text
w[r+1,k] = Σ(i∈successful_updates) (n_i / Σj n_j) × w[r+1,i,k]
```

其中 `n_i = ClientUpdate.sample_count`，且成功集合非空。key/shape/dtype/finite、round、
`global_state_id`、client 唯一性仍复用现有 `validate_client_update`/
`AggregationRequest`。非浮点 tensor 一律 `preserve_global`，即复制输入 global state，不能平均
客户端 buffer。任何重复/过期/NaN/Inf/空权重、未选 client 上传、选择 client 缺显式结果均拒绝；
失败写 `ClientFailure`，不伪装为 0 指标成功。

## 4. 结果、manifest 与失败语义（E/G 责任）

E 新增以下 **Proposed、待批准** 的结构化对象；实现时应位于 evaluation/experiments，不能让
UI 解析 `train.log`：

```text
ClientResultRecord(
  run_id, mode, client_id, status, error, train_sample_count, sample_visits,
  evaluation_sample_count, ade, fde, total_seconds, initial_state_id,
  final_state_id, artifact_paths
)
RoundRecord(
  round_index, selected_client_ids, successful_client_ids, failures,
  total_train_sample_count, aggregation_weights, input_global_state_id,
  output_global_state_id, sample_visits, metrics, total_seconds
)
FairnessRecord(
  data_version, split_id, partition_id, scaler_id, model_config_digest, seed,
  initial_state_id, metric_schema, planned_budget, actual_budget, comparable, reason
)
```

`ClientResultRecord.status ∈ {completed, failed, skipped}`；failed/skipped 必须有非空 reason，
`ade/fde` 对失败不可伪填 0（JSON 使用 `null`，CSV 保留空值和 status/error）。`RoundRecord` 必须
保留 selection、每个成功 client 的标准化权重、失败及输入/输出 state ID，即便最后整次运行失败。
E 从这些结构化记录生成 Client/macro/weighted summary、round 曲线和 comparison figure；JSON 为
事实源，CSV/图表只从同一对象导出。

任一模式异常：G 先原子写 config snapshot、身份、失败 ResultRecord/manifest 与可定位 error，
再以非零退出。三模式批量运行允许已成功的其他模式保留，但顶层 `status=failed` 且 exit 非零；
UI 必须同时显示成功、失败和不可比较原因。

## 5. 公平预算与恢复边界（G/A 责任）

默认公平 smoke 是全客户端、无失败、无提前停止：设每 client 有 N_c 个 train windows、
`L=local_epochs`、`R=rounds`。Centralized epochs、Local-only epochs 与 Federated 总本地 epoch
均为 `R × L`。三模式计划访问量均为 `Σ_c N_c × R × L`；运行后以各 ClientResult/RoundRecord 的
`sample_visits` 求和核对。Centralized 的 train loader 必须覆盖全部 client train windows；若其
实际访问量不同，`FairnessRecord.comparable=false` 且 compare 拒绝。

恢复只允许两个原子边界：Local-only 的“某 client 完整训练、评价与 artifact 已提交”边界，和
Federated 的“完整 round 已聚合且 RoundRecord/全局 state/checkpoint 已提交”边界。恢复包必须含
config/identity digests、initial/current state、model/optimizer（若需续训）、Python/NumPy/Torch RNG、
selector 状态、completed clients/rounds、planned/actual budget 和 artifact 相对路径。损坏、身份
不符、半个 round 或覆盖既有 `run_id` 一律拒绝；选择从上一个完整边界重跑时，重试访问量单列
`retry_sample_visits`，不能宣称等价且不计成本。

## 6. CLI、配置与 UI 的精确后续设计（A/G/UI）

在 G/A 合入前，现有 CLI 仍只支持 `--mode centralized`，UI 的 `local_only`、`federated`、
`compare` 保持 disabled。完成并批准后，唯一生产入口为：

```text
python -m src.cli train --mode {centralized,local_only,federated} \
  --data <configs/data.yaml> --model <configs/model.yaml> \
  --experiment <configs/experiments/three_mode_smoke.yaml> \
  --processed-dir <relative processed split> --output-root <relative outputs> \
  --run-id <safe id> [--seed <nonnegative>] [--resume-checkpoint <relative path>]

python scripts/run_three_mode_smoke.py
```

`three_mode_smoke.yaml` 必须含 `mode_matrix`（三模式）、`rounds`、`local_epochs`、
`clients_per_round`、公平性 identity 和恢复策略；A 的 `validate-config` 必须先校验 schema 与
identity，再创建模型/loader/输出目录。`run_three_mode_smoke.py` 不接收参数：生成匿名小样例、
校验三份配置、调用上述生产 runner、核验 manifest/ResultRecord/图表/预算；失败非零。UI 仅用
白名单参数数组调用这些入口，并从 v2 manifest/JSON/CSV 读取 Client/Round/FairnessRecord。

## 7. 责任与依赖矩阵

| 顺序/责任人 | 可开始条件 | 必须实现/消费 | 交接给 |
|---|---|---|---|
| B | 本草案获批 | client split manifest、train-owned holdout、计数/画像/空 client | C、E、D、G |
| C | B loaders/identity | 隔离 model/trainer、last-epoch adapter、state factory | D、G |
| E | B records 字段 | Client/Round/Fairness records、汇总/图表/JSON-CSV | D、G、UI |
| D | B+C | Local-only、Client/Server、数值 FedAvg、failure/round input | G、F |
| G | B+C+E+D | 三模式 runner、budget guard、恢复/manifest | A、UI、F |
| A | G 接口稳定 | CLI/config dispatch、无参 smoke、非零错误 | F、UI |
| UI | A/F 核心准出 | capability/白名单/只读记录展示 | 最终验收 |

## 8. S3 验收清单与测试样例

- B：train 并集完整/交集空；group 不跨 split；holdout 追随 train owner；空 client 显式 skipped；
  N=5,batch=2,L=3 断言 weight=5、visits=15、batches=3。
- C：两个 client 顺序互换仍有同一 initial_state_id；参数/optimizer/RNG 不共享；last epoch 与
  best checkpoint 差异时 ClientUpdate 取 last，Centralized 回归仍取 best。
- E：不等 evaluation count（如 2 与 8）精确验证 macro 与 weighted；failed client 指标为 null；
  JSON/CSV/figure 可从同一 records 重建。
- D：两 state 的人工浮点 FedAvg（n=1,3）逐元素正确；整数/布尔 buffer 保留 global；重复、
  过期、NaN、shape/dtype 错、缺结果和全失败均拒绝/记录。
- G：2 clients × 1 round 真 Trainer；逐字段扰动 split/model/seed/init/metric/budget 均拒绝；
  中断于完整 client/round 后恢复与连续运行对照；损坏/错 identity checkpoint 拒绝。
- A/F/UI：子进程验证三模式失败非零且其他 artifact 保留；无参 smoke 检查三份结果、预算、图表；
  页面检查 guard、失败面板、日志/产物、刷新不重训、路径/命令/重复 run/并发拒绝。

## 9. 本次变更影响、评审与批准记录

相较 D2：明确冻结了 train-owned holdout、`sample_count`/`sample_visits` 分离、S3 客户端采用
last-epoch state、v2 manifest、预算和恢复边界。现有 `fit_result_to_client_update` 的 best-state
语义不得改动；S2 Centralized、checkpoint、ResultRecord v1、CLI 与 UI 必须运行既有回归。
本草案未改代码、配置或现有公共签名；实现前须由 D/F 评审上述改变并记录对 S2 回归的结论，
并取得原规则要求的真人确认。未完成前状态始终为“草案待批准”。
