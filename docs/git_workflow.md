# Git 协作操作手册

**适用对象：** A—G 全体成员<br>
**适用仓库：** `trajectory-fl`<br>
**计划基线：** D1—D10、四个后续阶段（S1—S4）<br>
**目标：** 保护发布版本，保持集成分支可运行，让每项工作、测试、评审和阶段证据可追溯。

## 1. 分支与权限策略

```text
main  ←  仅接收已验收的 dev 里程碑 MR，保存发布标签
  ↑
dev   ←  受保护集成分支，接收已评审的工作分支 MR
  ↑
feature/fix/test/docs/chore  ←  从最新 dev 创建的单一工作包分支
```

| 分支 | 用途 | 允许合入来源 | 禁止事项 |
|---|---|---|---|
| `main` | 已验收的发布版本 | 仅 `dev` 的里程碑 MR | 直接开发、直接推送、强推 |
| `dev` | 每日集成、始终可运行 | 已评审工作分支 MR | 直接开发、绕过评审、失败测试合入 |
| 工作分支 | 一个工作包/Issue 的明确增量 | 从 `dev` 创建 | 混入无关重构或多个目标 |

进入 D3 前，A/F 负责将经评审的 D2 基线建立为 `dev` 并在托管平台设置保护规则。建议权限如下：

| 角色 | 推荐权限 | 责任 |
|---|---|---|
| A | `main`/`dev` 合并、发布标签、保护规则管理 | 阶段集成与发布 |
| F | CI/CodeCheck、MR 模板、保护规则协管 | 质量门禁与回归证据 |
| B—E、G | 推送自身工作分支、创建/评论/评审 MR | 模块代码、测试和文档交付 |

权限遵循最小授权；任何成员都不得直接推送 `main` 或 `dev`。

## 2. 命名规范

工作包、分支和提交须关联阶段与需求：

```text
工作包：S<阶段>-<角色>-<序号>            例如 S1-B-01
功能：feature/s1-b-highd-adapter
修复：fix/s3-d-fedavg-weight
测试：test/s4-f-system-acceptance
文档：docs/s4-e-result-analysis
工程：chore/s1-f-quality-gate
```

提交格式：

```text
<type>(<scope>): <英文动词短句> [工作包][需求/验收]
```

```text
feat(data): implement highd field mapping [S1-B-01][F1]
test(federated): verify weighted aggregation [S3-F-01][AT-05]
feat(experiments): add reproducible run matrix [S4-G-01][F6]
```

可用 type：`feat`、`fix`、`test`、`docs`、`refactor`、`chore`。禁止空提交、倒签、伪造作者或为了每日记录拆分无意义改动。

## 3. 开始工作与每日开发流程

首次加入或开始新工作包：

```powershell
git clone <仓库地址>
cd trajectory-fl
git switch dev
git pull --ff-only origin dev
git switch -c feature/s1-b-highd-adapter
```

开发中保持小步提交：

```powershell
git status
git diff
python -m ruff check src tests
python -m pytest -q
python -m src.cli validate-config
git add <本工作包文件>
git commit -m "feat(data): implement highd field mapping [S1-B-01][F1]"
git push -u origin feature/s1-b-highd-adapter
```

每个项目日，A—G 均须产生本人真实完成、可审阅的代码/测试提交，或在 MR 中留存可审计的评审记录。日报必须记录工作包、Issue、分支、Commit SHA、测试、MR、评审、阻塞和下一日计划。

## 4. 提交前质量门禁

```powershell
git status
git diff --check
python -m ruff check src tests
python -m pytest -q
python -m src.cli validate-config
git diff --cached --name-only
```

不得提交 `data/raw/`、`data/processed/`、`outputs/`、`.venv/`、checkpoint、日志、密钥、个人配置或绝对个人路径。若误暂存，使用：

```powershell
git restore --staged <文件路径>
```

## 5. MR 流程与评审

每个工作包对应一个 Issue、一个工作分支和一个目的明确的 MR。工作分支 MR 的目标必须是 `dev`；只有阶段验收通过后，A/F 才能创建 `dev`→`main` 的里程碑 MR。

MR 标题：`[S1-B-01][F1] Implement highD adapter and preprocessing`。

MR 描述最少包含：关联 Issue、需求/验收编号、改动范围、非范围、验证命令与结果、接口/配置影响、风险/回滚方式和需要重点评审的内容。

| 作者模块 | 主评审者 | 关键改动第二评审 |
|---|---|---|
| A：架构、CLI、发布 | D | F |
| B：数据管线 | F | D |
| C：模型、Trainer | E | A |
| D：Non-IID、联邦 | A | F |
| E：评价、可视化、结果 | C | B |
| F：质量、CI、工程化 | B | A |
| G：实验 runner、manifest、复现脚本 | D | A |

数据切分、指标、FedAvg、公共接口、依赖、保护规则和 `dev`→`main` MR 必须至少两人批准。其他 MR 至少一名非作者批准。

## 6. 同步、冲突与紧急修复

工作分支落后时：

```powershell
git fetch origin
git rebase origin/dev
python -m ruff check src tests
python -m pytest -q
git push --force-with-lease
```

`--force-with-lease` 仅允许用于自己已推送的工作分支，禁止用于 `main`/`dev`。冲突无法判断时停止 rebase，联系模块负责人；可用 `git rebase --abort` 恢复。

紧急修复从 `main` 创建 `fix/<阶段>-<缺陷>`，验证后分别向 `main` 和 `dev` 创建 MR，避免修复遗漏在后续开发线。

## 7. 阶段标签与每日收尾

| 时点 | 标签 | 条件 |
|---|---|---|
| D4 / MS2 | `ms2-data-pipeline` | 数据闭环、AT-01/04 通过 |
| D6 / MS3 | `ms3-centralized` | 集中式端到端、AT-03 通过 |
| D8 / MS4 | `ms4-three-modes` | 三模式小样本、AT-05/06 通过 |
| D10 / MS5 | `v1.0.0` | AT-01—AT-08、发布验收通过 |

每日收尾清单：

- [ ] 工作分支、Issue、工作包和 Commit 信息一致。
- [ ] 代码、测试、配置、文档和每日记录同步。
- [ ] 质量门禁通过，失败有原因与处理记录。
- [ ] MR 已创建/更新并有评审者；阶段证据已更新。
- [ ] 不含受限数据、输出、密钥和个人路径。
