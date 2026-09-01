# 基于联邦学习的车辆轨迹预测系统

本仓库是“基于联邦学习的车辆轨迹预测系统”的独立项目根目录。D1 已完成需求基线和数据选型；D2 正在冻结架构、配置与公共接口，之后再开始功能实现。

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
```

若尚未安装 Python，请先安装 Python 3.10 或更新版本，并重新打开终端。准备数据时，原始 highD CSV 放入 `data/raw/`；不要提交原始数据。

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

## D2 配置与命令契约

- `configs/data.yaml`、`configs/model.yaml` 和 `configs/experiments/smoke.yaml` 是最小可校验配置。
- `python -m src.cli validate-config` 校验 YAML schema 以及数据/模型的序列维度一致性。
- `prepare-data`、`train`、`compare` 已保留为后续实现入口；当前调用会明确提示未实现并返回非零退出码。
- 架构、Tensor、配置、输出目录和结果格式见 [D2 设计基线](docs/design.md)。

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
