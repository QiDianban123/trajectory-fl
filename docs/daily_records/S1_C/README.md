# S1-C-01 交付与评审准备记录

本目录集中保存本次 S1-C 文档：[任务卡及完成记录](task_card.md)。
后续 C/D 评审发现的三项缺陷修复及最新 128 项回归结果见 [评审修复记录](review_fixes.md)。
公共接口说明与测试计划分别保留在 [设计文档](../../design.md) 和 [测试计划](../../test_plan.md)。

- 实际执行日期：2026-09-05。D3/D4 为项目子任务标签，本次同日完成，不倒签提交。
- 需求/验收：F1、F3、AT-01（数据到模型桥接部分）。
- 分支：`feature/s1-c-torch-batching`；基线：同步后的 `origin/dev`，`0a45fc1`。
- 实际实现及测试 Commit SHA：`2d6e680edb64563160871fdab2eff764b245d40a`。
- Issue：未创建；MR：未创建，下方提供可直接使用的草稿。
- 状态：本地实现及联调通过，人工评审与合入待完成。

## D3 / D4 实际成果

| 子任务 | 实现 | 验证 |
|---|---|---|
| S1-C-D3 | `src/training/batching.py` 的单样本和 batch 显式转换 | 单样本、多样本、空 batch、错误类型/shape/dtype、NaN/Inf |
| S1-C-D4 | 可绑定的 DataLoader collate、metadata 复制、batch device 前置校验 | 现有 B Dataset、冻结 75/125 ModelContract、三种 split、spawn worker、尾批 |

`TrajectoryBatch.meta` 是兼容增量，默认空 tuple 保留 D2 二参数构造；桥接始终填充逐样本 metadata。
所有桥接输出都在 CPU，严格保持 float32，坐标数值不变；metadata 含 split/client_id 和扩展嵌套字段。
数据层没有新增 Torch 或模型依赖，没有实现 LSTM、优化器或训练循环。

## 使用与下游交接

```python
from functools import partial

from torch.utils.data import DataLoader

from src.models.base import ModelContract
from src.training import collate_trajectory_samples, sample_to_tensor

# model_config 为已校验的 configs/model.yaml 的 model 节。
contract = ModelContract.from_model_config(model_config)
batch = sample_to_tensor(dataset[0], contract=contract)
loader = DataLoader(
    dataset,
    batch_size=32,
    collate_fn=partial(collate_trajectory_samples, contract=contract),
)
for batch in loader:
    batch.validate(contract)
    # 后续 Trainer 负责移动 history/future 和模型，保留 batch.meta。
```

调用者持有 DataLoader；Windows 下启用 worker 时，入口遵守 `if __name__ == "__main__":`。
本次自动化测试使用 spawn 子进程验证 collate 与 metadata 可传输。
混合 split/split_id 的 batch 会报错，跨客户端的同一 split 允许合批。
数组和嵌套 metadata 不与原样本共享可变存储。

## 质量证据

使用项目已有 `.venv` 解释器，未变更依赖。

| 命令 | 实际结果 |
|---|---|
| `python -m pytest -q`（同步后的基线） | 71 passed |
| `python -m ruff check src tests` | All checks passed |
| `python -m pytest -q`（实现后全量） | 117 passed in 11.07s，新增 46 项 |
| `python -m src.cli validate-config` | Configuration valid: run=d2-smoke mode=smoke seed=42 |
| `git diff --check` | 通过 |

首次检查发现测试模块重名导致收集冲突，以及一处行宽超限；修正后上述门禁全部通过。
测试包括有限值重新校验、read-only/负 stride NumPy 数组、嵌套 metadata 独立性、
错误 device 在数值操作前被拒绝，以及缺少 PyTorch 时的安装指引。

## 评审结论与遗留问题

- 自检：满足 S1-C 本地代码和测试交付要求。
- 人工评审：E 主评审、B 数据对接确认待完成；按 Git 规范，公共接口变更还需 A 第二评审。
- 未推送、未创建 Issue/MR、未合入 dev，不将自动化测试或自检记作团队批准。
- 联调使用现有 `TrajectoryDataset` 和合成标准样本；最新 dev 尚无 B 的真实 highD adapter/pipeline，
  因而不声称已完成真实 highD 端到端或 AT-01 的全部验收。
- GPU 实际设备测试未执行；已验证 CPU 输出以及 CPU/meta 设备不一致的前置拒绝。
- 后续 C Trainer 必须保留 metadata，并在统一设备迁移后继续调用 batch 校验。

## MR 草稿

标题：`[S1-C-01][F3][AT-01] Implement NumPy to Torch batch bridge`

目标分支：`dev`；关联 Issue：待创建并补入编号。

NumPy `TrajectorySample` 原先不能直接供 Torch 训练使用，`TrajectoryBatch` 也不保留样本来源。
新增 `sample_to_tensor` 和 `collate_trajectory_samples`，严格校验输入，输出 batch-first CPU
float32 Tensor 并复制逐样本 metadata；修正 batch 校验顺序，使非法标签和设备不一致有明确错误。

- 范围：训练层转换、兼容 metadata 字段、单元/集成测试及说明。
- 非范围：LSTM、训练循环、highD adapter、依赖和配置变更。
- 验证：Ruff 通过；pytest 117 项通过；validate-config、diff --check 通过。
- 接口影响：新增训练层导出及默认空 tuple 的 `TrajectoryBatch.meta`；现有构造与 validate 调用兼容。
- 风险：复制坐标和嵌套 metadata 有内存开销；真实 highD 管线联调待 B 实现集成。
- 回滚：由工作分支 MR 撤销本工作包改动，恢复原训练契约；不涉及数据迁移。
- 重点评审：E/A 检查公共接口及设备责任；B 确认 split、metadata 和归一化坐标保持。
