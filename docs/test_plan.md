# D2 测试计划

**负责人：** F；**版本：** V1.0-D2。

## 当前可执行测试

- `tests/unit/test_config.py`、`test_cli.py`：YAML schema、缺文件和 CLI 错误返回。
- `tests/unit/test_data_contracts.py`：样本格式、先切分后滑窗、时间顺序与训练集 scaler。
- `tests/unit/test_model_training_contracts.py`：CPU 模型 shape、checkpoint、共享 Trainer 与 Local-only 隔离。
- `tests/unit/test_federated_contracts.py`：ClientUpdate、失败记录、state_dict、样本数与非浮点 buffer 策略。
- `tests/unit/test_metrics.py`、`test_evaluation_contracts.py`：米制 ADE/FDE、JSON/CSV schema 与图表输出路径。
- `tests/unit/test_utils.py`：固定随机源、路径逃逸拒绝、run 输出目录、结构化日志和共享夹具。

运行质量门禁：

```powershell
python -m pytest -q
python -m ruff check src tests
python -m src.cli validate-config
```

依赖版本由 `requirements.txt` 管理。测试报告只记录命令与结果，不记录设备路径、账户或其他机器专属信息。

本次 D2 基线已验证 `python -m pytest -q` 与 `python -m src.cli validate-config`；Ruff 检查在安装 `requirements.txt` 中声明的 Ruff 后执行。该前置条件不影响 pytest 的结果记录。

## 后续测试挂钩

| 阶段 | 测试 | 通过条件 |
|---|---|---|
| D3-D4 | 字段映射、缺失/异常、时间顺序、可逆归一化、车辆/场景无交集 | AT-01、AT-04 的数据部分 |
| D5-D6 | 模型形状/梯度/过拟合、训练—检查点集成 | AT-03 |
| D7-D8（S3） | 客户端隔离、FedAvg 人工数学用例、两客户端一轮集成、三模式冒烟 | AT-05、AT-06 |
| D9-D10（S4） | 小样本端到端、配置/缺文件错误、结果汇总重建、干净环境验收 | AT-02、AT-07、AT-08 |

每个新增核心函数必须包含正常、边界和异常用例；所有指标测试均使用反归一化物理坐标构造数据。pytest 的临时目录固定为仓库内 `.pytest-tmp/`，避免依赖系统临时目录权限。
