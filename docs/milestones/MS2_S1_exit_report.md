# MS2：S1 数据闭环准出报告

日期：2026-09-10。评审起始基线：`dev` `d42fe17`。

## 准出结论

S1 代码、自动化测试和匿名可重建样例满足技术准出条件。本准出修复合入 `dev`
并取得最新 Quality Gate 成功后，可以申请 `dev` 合并到 `main`。正式合并仍须
按照 Git 规范记录非作者人工评审；下表不代填人工批准。

## MS2 门禁

| 门禁 | 结果 | 自动化证据 |
|---|---|---|
| AT-01 highD 读取、清洗、split、scaler、滑窗和持久化 | 通过 | `test_highd_adapter.py`、`test_prepare_data_cli.py` |
| AT-01 Dataset → DataLoader → Torch Batch | 通过 | `test_data_model_bridge.py`、`test_batching.py` |
| AT-02 数据诊断与物理坐标图 | 通过 | `test_data_diagnostics.py`、`test_scaler_visualization.py`、重建脚本 |
| AT-04 默认 5-RSU Non-IID、并集/交集和稳定重建 | 通过 | `test_partition.py`、`test_s1_exit_smoke.py` |
| 同一车辆不跨 split | 通过 | highD adapter 测试及系统 smoke 的三组交集检查 |
| scaler 只由 train 拟合且可逆 | 通过 | data pipeline gate、highD adapter 和可视化集成测试 |
| 一条命令重建匿名合法小样例 | 通过 | `python scripts/run_s1_smoke.py` |
| 配置/路径/数据错误可定位，失败不留半成品 | 通过 | CLI process、配置路径、事务回滚测试 |
| manifest 可追到配置、SHA、data version、split、partition 和产物 | 通过 | RunContext、processed index 和 prepare-data 集成测试 |

## 一键 smoke 证据

smoke 使用 25 条匿名车辆轨迹、每条 200 帧和默认 `num_clients: 5`：

```text
Clients: rsu_01, rsu_02, rsu_03, rsu_04, rsu_05
Samples: {'train': 17, 'validation': 4, 'test': 4}
Split groups: disjoint
```

综合验证命令：

```powershell
$env:MPLBACKEND = "Agg"
python scripts\run_s1_smoke.py
python scripts\plot_data_diagnostics.py
python -m pytest -q
python -m ruff check src tests scripts
python -m src.cli validate-config
```

起始基线 `d42fe17` 的 GitHub Quality Gate 成功。加入本报告和系统 smoke 后，
本地全量测试为 `214 passed`，`python -m ruff check src tests scripts` 与配置校验
通过；环境、highD 探查、指标示例和数据诊断脚本均执行成功，诊断脚本生成四张
PNG。新提交的远端 Quality Gate 仍是最终事实源。

## 人工评审与合并门禁

| 范围 | 要求评审 | 状态 |
|---|---|---|
| A 编排、CLI、事务恢复 | D 主评审，B/G 协作 | 待在合并请求记录 |
| B highD、split/scaler | F 主评审，D 第二评审 | 待在合并请求记录 |
| C batch bridge | E 主评审，B 协作 | 待在合并请求记录 |
| D 5-RSU partition | A 主评审，F 第二评审 | 待在合并请求记录 |
| E 数据图 | C 主评审，B 协作 | 待在合并请求记录 |
| F 质量门禁 | B 主评审，A/D 协作 | 待在合并请求记录 |
| G RunContext/manifest | D 主评审，A/B 协作 | 待在合并请求记录 |

合并顺序：本准出修复分支 → `dev`；最新 `dev` Quality Gate 成功且至少一名
非作者完成批准后，再创建 `dev` → `main` 的 MS2 合并请求。P0 数据防泄漏、
partition 和公共接口建议取得两名评审。使用普通 merge commit 保留准出证据。
