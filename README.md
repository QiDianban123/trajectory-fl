# 基于联邦学习的车辆轨迹预测系统

本仓库是“基于联邦学习的车辆轨迹预测系统”的独立项目根目录。D1 已完成需求基线、数据选型、工程骨架、接口草案和测试入口；D2 冻结接口后再开始功能实现。

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
pytest
python -m src.cli status
python scripts\check_environment.py
```

若尚未安装 Python，请先安装 Python 3.10 或更新版本，并重新打开终端。准备数据时，原始 highD CSV 放入 `data/raw/`；不要提交原始数据。

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
