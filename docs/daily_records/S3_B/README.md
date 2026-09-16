# S3-B-01：客户端 DataLoader 与 Non-IID 数据核验

实际日期：2026-09-12（Asia/Shanghai）
执行者/模型：Codex（GPT-5）；单人串行。状态：技术候选，待 MR/真人确认。

## Issue、分支与依赖

- Issue：待创建；未取得平台 Issue/MR 创建权限，未伪造编号。
- 分支：`feature/s3-b-client-dataloaders`；本地起点为 A0 候选 `bc9b2e8`（尚未合入 dev），
  此例外由操作者授权继续本地开发；远端 `origin/dev` 基线仍为 `4c8c5e1`。
- D7：`dc89d1a`，新增隔离 client loader、冻结 holdout 投影和正常/边界测试。
- D8：`c00c281`，增加画像导出、极端 Non-IID、empty client、尾批与 sample-visits 测试。
- 修复：`f0693e9`（延迟 training import）；`abfffc0`（manifest/scaler identity/逐 window 验证）；
  `b7741b3`（prepare 与 loader 共享 canonical group ID）。最终代码候选：`b7741b3`。

## 实现与下游接口

- `src.data.client_loading` 只消费 `ProcessedDataBundle`、`PartitionManifest`、共享 collate 和
  `DataLoaderConfig`，产生稳定的 per-client train/validation/test loader、profile、partition/scaler
  identity、`trainable_client_ids` 与原子 `client_split_manifest.json`。
- train 逐 window 验证 canonical `(recording_id, vehicle_id)` → frozen client；validation/test 使用
  train-fitted scaler inverse-transform 的最后 history x 投影至冻结 contiguous intervals。无重分区、
  无复制补样本、无 scaler refit；`drop_last`、重复 window、标签交换、越界均拒绝。
- C 消费 loader/identity；E 消费 profile/manifest；D 消费 `train_sample_count`（FedAvg 权重）、
  `sample_visits(local_epochs)` 与 `trainable_client_ids`；G/A/UI 仅在后续接口批准后消费 manifest。
  未实现的 Client/FedAvg/CLI/UI mode 没有启用。

## 命令、退出码与结果

| 命令 | 退出码/结果 |
|---|---|
| 定向 Ruff + client/processed/partition/prepare/import 测试 | 0；最终相关集通过 |
| `.venv\\Scripts\\python.exe -m pytest -q`（未设无头后端） | 1；基线缺少未合入 preflight 的 Tk/Tcl 图表修复，14 个非 B 图表测试失败，记录为前置缺口 |
| `MPLBACKEND=Agg .venv\\Scripts\\python.exe -m pytest -q` | 0；`304 passed in 94.70s`，运行 SHA `b7741b3` |
| `.venv\\Scripts\\python.exe -m ruff check src tests scripts` | 0；All checks passed |
| `.venv\\Scripts\\python.exe -m src.cli validate-config` | 0；`run=d2-smoke mode=smoke seed=42` |
| `git diff --check` | 0；本步候选差异无空白错误 |

## 缺陷与评审闭环

- F/D 初审：缺完整 client manifest/scaler identity、逐 window ownership、N=5/batch=2/L=3 证据；已在
  `abfffc0` 修复并补测试。
- F/D 复审：发现 group ID 未按 prepare 的 `str(recording_id), int(vehicle_id)` 规范编码；已提取
  `partition.client_group_id` 并在 prepare/client loader 复用，`b7741b3` 回归 numeric/string recording ID。
- AI 审评：F、D 角色视角均为 Codex（GPT-5）只读复核 `b7741b3`，结论 P0=0、P1=0；不替代真人审批。
- 真人批准、远端 Quality Gate、MR URL：待处理。没有推送、合并、修改 dev/main 或伪造批准。

## 继续条件

1. 将 B 候选以单一目的 MR 目标 `dev`，取得真实 F/D 及原规则真人确认和最终候选远端 CI。
2. 平台合入 `dev` 后，C 从最新 `origin/dev` 实现隔离 model/trainer 与 last-epoch state；D 不得在 B
   分支实现 Client/FedAvg。
3. 若 A0 未先按流程合入，MR 必须显式说明 B 依赖其 `s3_contract.md` 候选，不能将该依赖写作已批准。
