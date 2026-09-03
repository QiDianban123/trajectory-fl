# 工作包命名、Git 与证据规范

## 1. 工作包和文件

- 工作包：`S<1-4>-<A-G>-<两位序号>`，例如 `S2-C-01`。
- 每日子任务：`S2-C-D5`、`S2-C-D6`。
- 文件固定为 `S1_D3-D4.md`、`S2_D5-D6.md`、`S3_D7-D8.md`、`S4_D9-D10.md`。
- Python 模块/函数/配置键用 `snake_case`，类用 `PascalCase`，常量用 `UPPER_SNAKE_CASE`。
- 测试文件用 `test_<module>.py`，测试函数用 `test_<condition>_<expected>()`。
- 客户端 ID 用 `rsu_<NN>`；实验配置用 `configs/experiments/s<阶段>_<mode>_<purpose>.yaml`。
- `run_id` 用 `YYYYMMDDTHHMMSSZ-<mode>-seed<seed>`，图表和检查点使用相对路径。

## 2. Git

- `main`：受保护发布分支；`dev`：受保护集成分支；工作分支一律从最新 `dev` 创建。
- 分支：`feature/s<阶段>-<角色>-<主题>`；缺陷/测试/文档使用 `fix/`、`test/`、`docs/`。
- Commit：`<type>(<scope>): <英文动词短句> [工作包][需求/验收]`。
- MR：`[工作包][需求/验收] 简短标题`，目标为 `dev`；阶段验收后再由 `dev` 向 `main` 发 MR。

示例：

```text
feature/s1-b-highd-adapter
feat(data): implement highd field mapping [S1-B-01][F1]
[S1-B-01][F1] Implement highD adapter and preprocessing
```

## 3. 公共质量门禁

```powershell
python -m ruff check src tests
python -m pytest -q
python -m src.cli validate-config
git diff --check
```

## 4. 每日证据

- 两个项目日均须有本人真实、可审阅的代码/测试提交或 MR 更新；禁止空提交、倒签和伪造作者。
- 记录 Issue、分支、Commit SHA、MR、测试命令/结果、评审者、阻塞和次日计划。
- 不提交 `data/raw`、`data/processed`、`outputs`、`.venv`、`.pt`、日志、密钥或个人绝对路径。
- 公共接口、数据切分、指标和 FedAvg 变更须两人确认；其他 MR 至少一名非作者批准。

## 5. 工作包完成报告字段

```text
实际 Commit SHA：
MR：
Ruff：
pytest：
配置/冒烟检查：
评审结论：
遗留问题：
下游交接：
```
