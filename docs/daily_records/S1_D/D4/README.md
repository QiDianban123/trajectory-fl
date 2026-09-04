# 成员④ D4 交付包（QA + 联邦实验工程师）

**日期：** 2026-09-05（D4）
**成员：** 成员④ = 质量保证员（QA / 代码审查）＋ 联邦实验工程师（三种训练方式）
**用途：** 供 A/F 评审 S1-D-01 收尾增量（MS2 数据闭环证据之一）；供 成员⑥ 归档过程材料。
**对应任务卡：** `docs/S1_D3-D4/D.md`「S1-D-01：5-RSU Non-IID 划分」→ **D4 收尾（S1-D-01 完成）**。
**工作分支：** `feature/s1-d-noniid-partition`　**本日提交：** `520e896`（D3 提交 `5c53dcf`、`88f0413` 为前置）
**推送状态：** ⏳ 本沙箱无外网（SSH 被禁），本地提交已就绪但**尚未推送**；请在成员环境执行 `git push -u origin feature/s1-d-noniid-partition` 后创建指向 `dev` 的 MR（见 02 号 AI 文档 OP-08）。

## 文件清单

| 序号 | 文件 | 内容 | 用途 |
|---|---|---|---|
| 01 | `01_任务卡_成员4_D4_完成记录.md` | S1-D-01 D4（收尾）完成记录：分配/合并/统计/manifest、测试、AT-04 对应与示例 manifest | 任务卡回填、评审 |
| 02 | `02_AI过程文档_成员4_D4.md` | 本次 D4 全部 AI 操作过程日志（六要素） | AI 工具专项说明素材（成员⑥） |
| 03 | `03_QA代码审查自检_成员4_D4.md` | QA 自审：并集/交集/泄漏/重建不变式、合并规则审查与评审请求 | 代码审查记录 |

## 与 docs/S1_D3-D4/D.md「D4」及「交付与验收」的对应

| 要求 | 完成证据 | 状态 |
|---|---|---|
| D4 生产代码：空间 Non-IID 分配 | `partition_train_groups`（纵向中点锚定 → 区域） | ✅ |
| D4 生产代码：相邻小区域合并 | 样本低于 `min_samples_per_client` 的区域并入相邻更小区域（不复制样本填充） | ✅ |
| D4 生产代码：样本/车辆统计 | `ClientPartition`/`PartitionManifest.to_mapping`：每客户端样本数/车辆数/空间范围 | ✅ |
| D4 测试：客户端并集完整 | `check_partition_invariants` + 单测 | ✅ |
| D4 测试：交集符合规则 | 单车辆唯一归属（跨客户端交集为空）断言 | ✅ |
| D4 测试：无跨 split 泄漏 | 输入契约=train-only（partition 层之上由 B/F 的 split 断言保障）；本层拒绝越界/外来分组 | ✅（语义层） |
| D4 测试：稳定重建 | 输入乱序/重复运行 → manifest 逐字节一致 | ✅ |
| 交付：partition manifest、每客户端样本数/车辆数/空间范围 | `PartitionManifest.to_mapping()`（见 01 号文档示例） | ✅ |
| 交付：不复制样本填充客户端 | 组级整属 + 合并仅邻接区域，无任何样本复制路径 | ✅ |
| 交付：train 外数据不参与客户端训练划分 | 模块仅接受训练分组；manifest 面向 train split | ✅ |
| 建议提交 | `feat(federated): partition noniid clients [S1-D-01][AT-04]` = `520e896` | ✅ |

## 质量门禁状态

| 门禁 | 状态 | 说明 |
|---|---|---|
| `python -m py_compile`（全部新增/改动 .py） | ✅ 通过 | 语法级验证 |
| 纯标准库冒烟（分配/合并/不变式/示例 manifest） | ✅ 通过 | 本机无第三方依赖 |
| `python -m pytest -q` | ⏸→修复中 | 成员环境已复跑：69 通过 / 2 失败（断言/文案不匹配，QA-D4-05/06 已修复），修复后预计 71 项待复跑确认 |
| `python -m ruff check src tests` | ⏸ 环境受限 | 同上 |
| `python -m src.cli validate-config` | ⏸ 环境受限 | 同上 |

> 环境阻塞详见 02 号 AI 过程文档；不虚构门禁结果。

## 评审与联调请求

- 主评审 **A**、第二评审 **F**：核对 `partition_train_groups` 合并规则、manifest schema、`check_partition_invariants` 覆盖与 AT-04 数据部分证据。
- 协作：B（train 分组 → `GroupExtent` 输入格式）；G（partition manifest 并入 data/split manifest 与结果索引）；F（集成门禁测试引用本模块）。
- 本日未改动 D2 冻结接口；`configs/data.yaml` 的 `partition:` 段（D3 引入）今日无变更。

## 遗留与下一日（S2/D5）计划

- [ ] 环境就绪后复跑 pytest/Ruff/validate-config，回填三份验证记录。
- [ ] B 联调：真实 highD 探查产出 → `GroupExtent`（含 `sample_count` 口径确认）。
- [ ] G 联调：`PartitionManifest.to_mapping()` 与数据 manifest JSON 的字段合并评审。
- [ ] D5（S2）按计划转入统一初始 state 复制/hash 与本地训练 adapter（开发计划书 §5.2）。
