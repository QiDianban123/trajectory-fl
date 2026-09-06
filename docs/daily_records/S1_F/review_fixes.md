# S1-F 质量门禁修复与 MS2 状态

日期：2026-09-06。集成基线：dev `7539605`；原 F 提交：`f724fc5`。

## 已修复

- CI runner 从无可用执行器的 arch-latest 改为 ubuntu-latest，设置 Agg
  后端、20 分钟超时，并保存 JUnit 报告。
- 绝对路径用例使用当前平台生成的绝对路径，避免 Windows 路径在 Linux
  被当成相对路径。没有修改路径安全生产逻辑。
- 删除仅验证字典 KeyError 和整数比较的伪覆盖；替换为实际 TrajectorySample
  metadata 拒绝和 TrajectoryDataset 窗口长度拒绝测试。
- 删除要求 prepare-data 永远未实现以及旧 --config 参数的测试。
  增加真实 CLI 子进程对缺配置、空 YAML、错误根类型的退出码和路径诊断检查。
- 检查 holdout 转换前后训练 scaler 均值、尺度及训练集变换不变。

## 本地验证

Windows / Python 3.13，MPLBACKEND=Agg，在最新 dev 集成态执行：

```text
python -m pytest -q --junitxml=outputs/quality-gate.xml
178 passed
python -m ruff check src tests
All checks passed!
python -m src.cli validate-config
Configuration valid: run=d2-smoke mode=smoke seed=42
```

这是本地验证记录，不代表 Ubuntu/Python 3.10 远端 CI 已通过。

## MS2 仍未验收

当前 dev 无具体 HighDAdapter，prepare-data 仍为 D2 占位入口。现有测试
覆盖标准样本、frame validator、split/scaler 契约、配置路径，不宣称覆盖真实
highD 输入拒绝、清洗统计或完整数据准备流程。没有通过 skip/xfail 隐藏缺失。

待 A/B 接口与实现合入后，由 F 完成：

- 缺 id/frame/x/y、NaN/Inf、重复/乱序 frame、短轨迹经过真实 adapter 的检查；
  核对拒绝原因与统计，防止静默丢样本。
- 合法小样例经 prepare-data 返回 0，核对样本、scaler、split/partition manifest。
- 检查实际输出分组互斥、并集完整、scaler 仅由 train 拟合；改变 holdout 数据
  不改变训练统计。覆盖缺输入及输出路径不可写/指向文件。
- B 主评审、A 协作评审和 MS2 结论。尚无人工批准记录，不代填。

## 合并要求

来源 test/s1-f-data-quality，目标 dev，沿用同来源的开放 MR。
MR 必须标明“质量基础设施与契约测试，MS2 未完成”；检查最新提交的
Quality Gate 成功并经 B/A 评审后，才可按部分交付合并。若 MR 声称完整
S1-F 交付，则上述真实管线验收完成前保持 Draft，不能将绿色测试视为 MS2 完成。
使用普通 merge commit 保留两个原始 D3/D4 提交与修复记录，勿强推或绕过门禁。
