# Git 协作操作手册

**适用对象：** A—F 全体成员<br>
**适用仓库：** `trajectory-fl`<br>
**目标：** 保持 `main` 始终可运行、每项变更可追踪、数据与实验产物不误传。

## 1. 角色与基本规则

1. `main` 是受保护的集成分支，禁止直接开发或直接推送；所有代码通过 Pull Request（PR）合入。
2. 一个 Issue 对应一个分支和一个目的明确的 PR；不要把无关格式化、重构和新功能混在同一 PR。
3. 提交前必须运行相关测试；合入前至少获得 1 位非作者批准。数据划分、指标、FedAvg 和核心接口需要 2 人确认。
4. 不提交 `data/raw/`、`data/processed/`、`outputs/`、`.venv/`、检查点、密钥或个人配置。提交前检查 `git status`。
5. 不使用 `git push --force` 推送 `main`；任何需要强制推送的情况只允许在**自己的功能分支**使用 `--force-with-lease`，并在 PR 说明中写明原因。

## 2. 首次加入项目

```powershell
git clone https://github.com/QiDianban123/trajectory-fl.git
cd trajectory-fl
git switch main
git pull --ff-only origin main
```

按 README 创建虚拟环境、安装依赖并验证：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
```

使用 `git config --global user.name "你的姓名"` 和 `git config --global user.email "你的邮箱"` 配置自己的提交身份。首次 `git push` 若要求登录，请在浏览器完成 GitHub 授权或使用 Personal Access Token；不得把 Token 写入代码、文档或聊天记录。

## 3. 日常开发标准流程

每次开始一项任务都从最新的 `main` 创建分支：

```powershell
git switch main
git pull --ff-only origin main
git switch -c feature/F3-lstm-model
```

分支命名必须体现类型和需求/验收编号：

| 类型 | 格式 | 示例 |
|---|---|---|
| 新功能 | `feature/<需求>-<简述>` | `feature/F3-lstm-model` |
| 缺陷修复 | `fix/<验收或缺陷>-<简述>` | `fix/AT-05-fedavg-weight` |
| 测试 | `test/<模块>-<简述>` | `test/metrics-nan-case` |
| 文档 | `docs/<简述>` | `docs/design-update` |
| 工程配置 | `chore/<简述>` | `chore/ci-pytest` |

开发过程中保持小步提交：

```powershell
git status
git diff
python -m pytest -q
git add src\models\lstm_seq2seq.py tests\unit\test_lstm_seq2seq.py
git commit -m "feat(model): implement LSTM decoder [F3]"
git push -u origin feature/F3-lstm-model
```

推荐提交格式：`<type>(<scope>): <动词短句> [需求编号]`。`type` 可使用 `feat`、`fix`、`test`、`docs`、`refactor`、`chore`。一次提交应可独立理解且不破坏测试。

## 4. 提交前的必做检查

```powershell
git status
git diff --check
python -m pytest -q
```

确认暂存区内容只包含本任务：

```powershell
git diff --cached --name-only
```

特别检查以下内容没有出现：原始 CSV、`outputs/`、`.pt` 检查点、`.env`、Token、绝对个人路径。若误暂存文件，使用：

```powershell
git restore --staged <文件路径>
```

这只会取消暂存，不会删除本地文件。

## 5. 创建和处理 Pull Request

推送分支后，在 GitHub 创建 PR，目标分支选 `main`。PR 描述必须写清：

- 关联的 Issue 与需求编号（F1—F7、M1—M6 或 AT 编号）；
- 改动范围及明确的非范围；
- 执行过的测试命令和结果；
- 配置、数据格式或接口是否变化；
- 风险、已知限制，以及需要审阅者重点检查的点；
- 若有图表，仅附可再生成的结果截图或路径，不能提交大型原始产物。

固定评审关系如下：

| 变更模块 | 作者 | 固定评审者 |
|---|---|---|
| 编排、集中式训练 | A | D |
| 数据管线 | B | F |
| 模型、Trainer | C | E |
| 客户端划分、联邦训练 | D | A |
| 指标、可视化、汇总 | E | C |
| 工程化、测试基础设施 | F | B |

作者负责回复评论、更新测试和保持 PR 可合并；评审者检查正确性、接口、异常路径、数据泄漏、测试和文档。未经批准不得自行合并。建议每个 PR 只解决一个 Issue，并优先使用 GitHub 的 **Squash and merge** 保持 `main` 历史简洁。

## 6. 开发中同步 main

合入前或分支落后时，先保存或提交手头工作，再同步：

```powershell
git fetch origin
git rebase origin/main
python -m pytest -q
git push --force-with-lease
```

最后一条只用于自己已经推送过的功能分支；从未推送过的分支使用普通 `git push -u origin <分支名>`。若 rebase 冲突：

```powershell
git status
# 手工编辑冲突文件，保留正确实现后：
git add <已解决文件>
git rebase --continue
```

无法判断应保留哪一侧时，暂停并向模块负责人提问。放弃本次同步可使用 `git rebase --abort`；不要使用 `git reset --hard` 清除未知内容。

## 7. 常见场景

### 已改文件但暂时不想提交

```powershell
git stash push -m "wip: F3 decoder experiment"
git pull --ff-only origin main
git stash pop
```

`stash pop` 也可能冲突，按第 6 节处理。

### main 合入了有问题的提交

由 A 或原作者创建新的修复分支，使用可审阅的反向提交，而不是改写公共历史：

```powershell
git switch -c fix/revert-bad-change origin/main
git revert <有问题的提交SHA>
python -m pytest -q
git push -u origin fix/revert-bad-change
```

### 误把敏感信息或大型文件推送到远程

立即停止继续推送，通知 A 和 F。Token 必须立即在服务端撤销/轮换；仅从工作区删除文件不足以消除历史记录，应由负责人评估清理历史和强制推送方案。

## 8. 里程碑与发布

只有 A/F 在对应评审通过后创建并推送标签：

| 时点 | 标签 |
|---|---|
| D6 | `m2-centralized` |
| D10 | `m3-three-modes` |
| D13 | `rc1` |
| D15 | `v1.0.0` |

示例：

```powershell
git switch main
git pull --ff-only origin main
git tag -a m2-centralized -m "M2: centralized pipeline verified"
git push origin main --tags
```

## 9. 每日收尾清单

- [ ] 当前分支不是 `main`，且分支名、Issue、需求编号一致。
- [ ] `git status` 中没有意外文件。
- [ ] 相关 pytest 测试已通过；失败结果和原因已记录。
- [ ] 代码、测试、配置和文档同步更新。
- [ ] 已推送个人分支并创建/更新 PR，或通过 `stash` 明确保存未完成工作。
- [ ] 没有提交任何数据、模型、运行输出、密钥或个人绝对路径。
