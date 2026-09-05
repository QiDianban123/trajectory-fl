# S1-A-01：数据准备入口与阶段集成

- 阶段：S1（D3—D4）
- 角色：A，项目负责人/架构与集成开发
- 分支：`feature/s1-a-prepare-data-cli`
- 主评审：D；协作：B、G
- 关联：F1、M1、M6、AT-01

## 目标

把冻结的数据配置接入 `prepare-data` CLI，形成可定位错误、可追踪产物的数据准备入口。不得实现 B 的 highD 解析逻辑。

## D3

- 生产代码：在 `src/cli.py` 和编排层接入 data config、输出根目录与退出码。
- 测试：配置缺失、输入目录缺失、成功/失败返回码。
- 建议提交：`feat(cli): wire prepare-data command [S1-A-01][F1]`。

## D4

- 生产/集成：串联 B 的 adapter、C 的 batching、D 的 partition 和 G 的 manifest。
- 测试/文档：增加最小数据冒烟；记录冻结接口冲突与 MS2 评审结论。
- 建议提交：`test(cli): add prepare-data smoke flow [S1-A-01][AT-01]`。

## 交付与验收

- `prepare-data` 成功返回 0，配置/数据错误返回非零且消息可定位。
- CLI 不包含清洗、滑窗或划分算法；调用公共接口完成编排。
- 产物含数据统计、split/manifest 路径；公共质量门禁通过。
- MR 目标 `dev`，附两日 Commit、测试和 D/B/G 评审证据。

## D3 完成情况（2026-09-05，晚间统一整理用）

### 代码变更（分支 `feature/s1-a-prepare-data-cli`，基于 `main` D2 基线）

1. 新建 `src/data/pipeline.py`（编排层）：
   - `DataPreparationError(ValueError)`：数据类可定位错误（如输入目录不存在），与 `ConfigError` 区分。
   - `PrepareDataResult`（frozen dataclass）：打包已验证输入与输出路径；窗口/切分参数随配置回显，D4 扩展数据统计与 split/manifest 路径。
   - `prepare_data(config_path, *, output_root=None)`：复用 `load_and_validate(path, "data")` 做配置校验；检查 `dataset.raw_dir` 存在（fail-fast，消息含完整路径）；创建输出根目录（`parents=True, exist_ok=True`），不实现任何解析/清洗/滑窗/划分算法。
2. 修改 `src/cli.py`：
   - `prepare-data` 子命令接入：新增 `--data`（默认 `configs/data.yaml`）与 `--output`（可选，覆盖 `dataset.processed_dir`）参数。
   - 退出码契约：成功返回 0；`ConfigError` 与 `DataPreparationError` 均打印到 stderr 并返回 2，错误前缀分别为 `Configuration error:` / `Data error:`；未知异常不捕获。
   - 成功时打印数据集、配置、输入/输出路径、窗口（history/future/stride）与切分（train/val/test、seed）摘要。
   - `train`/`compare` 维持 D2 占位行为（stderr 提示未实现，返回 2）。
3. 扩充 `tests/unit/test_cli.py`：新增 5 个用例，共 9 项。

### 测试覆盖（对应 D3 第二条"配置缺失、输入目录缺失、成功/失败返回码"）

- 成功：`test_prepare_data_success_creates_output_root`（退出码 0、输出目录已创建、stdout 含摘要）。
- 参数契约：`test_prepare_data_output_flag_overrides_processed_dir`（`--output` 生效）。
- 配置缺失：`test_prepare_data_missing_config_file_returns_error`（文件不存在，返回 2）。
- 配置非法：`test_prepare_data_malformed_config_returns_error`（空 YAML，返回 2）。
- 输入目录缺失：`test_prepare_data_missing_input_directory_returns_error`（返回 2，stderr 含 `Data error` 与完整路径；断言输出目录未被创建，验证先检查输入再建输出的顺序）。

### 验证命令与结果

- `python3 -m ruff check src/cli.py src/data/pipeline.py tests/unit/test_cli.py`：All checks passed。
- `python3 -m pytest tests/unit/test_cli.py -v`：9 passed。
- 全量 `python3 -m pytest tests/ -q`：44 passed，1 failed。
- 冒烟：`prepare-data --data` 指向不存在配置 → stderr `Configuration error: ...` 且 exit=2；构造临时配置并创建 raw 目录 → 打印准备摘要且 exit=0。

### 遗留/待办（D4 及晚间记录）

1. `tests/unit/test_utils.py::test_output_paths_reject_traversal_and_absolute_artifacts` 失败：用例以 `Path("C:/outside.json")` 断言抛 `PathSafetyError(match="relative")`，在 macOS 上该路径不被判定为绝对路径故未抛出；经 `git stash` 验证在 D2 基线代码上同样失败，与本次改动无关，建议转 F 做跨平台处理（如改用 POSIX 绝对路径或跳过平台特定用例）。
2. 尚未提交：待晚间评审后按建议信息 `feat(cli): wire prepare-data command [S1-A-01][F1]` 提交，推送 fork 并向 `dev` 发起 MR。
3. D4 待办：串联 B 的 adapter、C 的 batching、D 的 partition、G 的 manifest；补最小数据冒烟；在 `PrepareDataResult` 中增加数据统计与 split/manifest 路径字段；记录接口冲突与 MS2 评审结论。
4. 评审证据：待 D 主评审、B/G 协作评审后补记录。
