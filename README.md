# 基于联邦学习的车辆轨迹预测系统

本仓库是“基于联邦学习的车辆轨迹预测系统”的独立项目根目录。D1 已完成需求基线和数据选型，D2 已完成架构与公共接口冻结，S1 已形成可验证的数据闭环。

## D1 已确定的范围

- 主数据集：highD（单一主数据集；原始数据不提交）。
- P0：F1—F7 与 M1—M6，详见 [需求基线](docs/requirements.md)。
- 技术栈：Python 3.10+、PyTorch、NumPy/Pandas、Matplotlib、PyYAML、pytest。
- D1 不实现训练或联邦流程，避免在接口冻结前产生返工。

## 快速开始

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
python -m src.cli status
python -m src.cli validate-config
python scripts\check_environment.py
python scripts\run_s1_smoke.py
```

若尚未安装 Python，请先安装 Python 3.10 或更新版本，并重新打开终端。准备数据时，原始 highD CSV 放入 `data/raw/`；不要提交原始数据。

`run_s1_smoke.py` 会生成匿名 highD 风格小样例，调用生产 `prepare-data`，并验证
train/validation/test 互斥、默认 5-RSU 划分及完整 manifest。每次运行使用新的
`outputs/s1-smoke-<UTC time>/`，不会覆盖历史结果。

## S1 数据诊断图

运行以下命令可从确定性的合法小样例重建原始/清洗轨迹、异常计数、真值轨迹和
scaler inverse-transform 抽检图：

```powershell
python scripts\plot_data_diagnostics.py
```

默认输出位于 `outputs/s1-e-data-plots/figures/data/`。可使用 `--output-dir <目录>`
覆盖输出位置；脚本只生成可重建产物，不向仓库提交运行图片。

## 再次进入虚拟环境（PowerShell）

每次打开新的终端后，先进入仓库根目录，再激活已创建的 `.venv`：

```powershell
cd <仓库目录>
.\.venv\Scripts\Activate.ps1
python --version
python -m pytest -q
```

命令提示符出现 `(.venv)` 即表示激活成功。若 PowerShell 阻止执行激活脚本，仅对当前终端临时放行后再激活：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

完成工作后可运行 `deactivate` 退出虚拟环境。始终使用 `python -m pytest`，以确保 pytest 使用当前 `.venv` 的解释器和依赖。

## 配置与命令契约

- `configs/data.yaml`、`configs/model.yaml` 和 `configs/experiments/smoke.yaml` 是最小可校验配置。
- `python -m src.cli validate-config` 校验 YAML schema 以及数据/模型的序列维度一致性。
- `python -m src.cli prepare-data` 执行 highD 数据准备；支持输入、processed、运行输出和 run ID 覆盖，成功返回 0，用户可修复错误返回 2。
- `python -m src.cli train --mode centralized` 运行已准备的 processed 数据；CLI 只解析参数、
  校验配置并调用共享 `CentralizedExperiment`。
- 架构、Tensor、配置、输出目录和结果格式见 [D2 设计基线](docs/design.md)。
- 运行产物包括 `metrics.json`、`metrics.csv`、`figures/` 和 `checkpoints/`；JSON 是结果事实源，CSV 是自动生成的扁平视图。

## 质量与可复现性

每次功能变更至少执行以下命令：

```powershell
python -m pytest -q
python -m ruff check src tests scripts
python -m src.cli validate-config
```

GitHub Actions 的 Quality Gate 使用 Ubuntu / Python 3.10，并保存 JUnit 报告。
本地可设置 `MPLBACKEND=Agg` 使用无界面绘图，再运行上述相同检查；需要报告时
运行 `python -m pytest -q --junitxml=outputs/quality-gate.xml`。
S1 的准出范围、自动化证据和人工审批状态见
[MS2 S1 准出报告](docs/milestones/MS2_S1_exit_report.md)。
S2 集中式训练的 AT-03 映射、缺陷状态和候选准出条件见
[MS3 S2 准出报告](docs/milestones/MS3_S2_exit_report.md)。

运行入口应在创建模型、数据划分或训练前调用 `set_global_seed(seed)`。每个 run 使用唯一 `run_id`，其配置、元数据、JSON 日志、指标、检查点和图表保存在 `outputs/<run_id>/`；已存在的 run ID 会被拒绝，避免覆盖结果。

## D1 交付物映射

| 成员 | D1 交付物 |
|---|---|
| A | 项目/CLI/Issue 骨架、需求和任务基线 |
| B | 数据选型记录、可安全运行的数据可用性探查脚本 |
| C | 模型输入输出接口草案 |
| D | Non-IID 和 FedAvg 方案、联邦契约草案 |
| E | ADE/FDE 实现与图表样例脚本 |
| F | 依赖、环境检查、pytest 骨架、测试计划 |

## 目录

`docs/` 保存 D1 的基线与记录，`scripts/` 保存人工执行入口，`src/` 保存可演进的模块接口和已完成的轻量工具，`tests/` 保存自动化测试。开发任务、风险和当日决策见 `docs/`。

团队协作请遵循 [Git 协作操作手册](docs/git_workflow.md)。

## 集中式 smoke

一条命令生成匿名 highD 风格数据、执行集中式训练并验证核心产物：

```powershell
python scripts\run_centralized_smoke.py
```

对已准备的数据，正式入口示例为：

```powershell
python -m src.cli train --mode centralized --processed-dir data/processed/<split_id> --run-id centralized-001
```

## S2 本地交互控制台

S2-UI-01 提供受控的 Streamlit 页面：

```powershell
python scripts\run_ui.py
```

页面只开放已验收的 Centralized smoke 和 Centralized train，S3 的 Local-only/Federated
操作会显示为禁用状态。命令预览使用参数数组生成，页面不接受任意 shell 命令；已保存的
metrics、manifest、日志、checkpoint 和图表从 run 目录只读加载。
