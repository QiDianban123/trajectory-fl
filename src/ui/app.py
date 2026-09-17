"""Streamlit console for controlled training and authoritative final results."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import streamlit as st

from src.ui.capabilities import (
    UiCommandError,
    build_prepare_data_command,
    build_s3_train_command,
    build_smoke_command,
    build_three_mode_smoke_command,
    preflight_s3_fairness,
    s3_capabilities,
)
from src.ui.command_runner import CommandResult, CommandRunner, UiRunState
from src.ui.data_import import find_processed_summary, save_uploaded_csvs
from src.ui.run_index import (
    FinalArchiveSummary,
    RunSummary,
    comparison_error,
    discover_final_archive,
    discover_runs,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    """Render the controlled three-mode console and retained final archive."""

    st.set_page_config(page_title="Trajectory-FL 控制台", page_icon="◈", layout="wide")
    _style()
    archive = discover_final_archive(PROJECT_ROOT)
    output_runs = discover_runs(PROJECT_ROOT)
    st.title("Trajectory-FL · 三模式训练与最终结果")
    st.caption(
        "训练操作使用受控 CLI；正式结论来自 FINAL_MANIFEST.json，运行细节来自结构化 manifest。"
    )
    _status_bar(archive, output_runs)
    capabilities = s3_capabilities()
    with st.sidebar:
        st.header("高级操作")
        action = st.radio(
            "快捷操作",
            [item.key for item in capabilities],
            format_func=lambda key: next(item.label for item in capabilities if item.key == key),
        )
        selected = next(item for item in capabilities if item.key == action)
        if not selected.enabled:
            st.info(selected.reason)
        command = _build_command(selected.key) if selected.enabled else None
        if command is not None:
            st.caption("命令预览")
            st.code(command.preview, language="text")
            if st.button("执行受控操作", type="primary", width="stretch"):
                _run_command(command)
    _demo_workbench()
    _final_results(archive)
    _command_console()
    _runs_and_results(output_runs, archive)


def _style() -> None:
    st.markdown(
        """<style>
        .stApp { background: #0b1220; color: #e7edf7; }
        [data-testid="stSidebar"] { background: #111c2f; }
        .metric-card { border: 1px solid #25405f; border-radius: 12px;
          padding: 1rem; background: #111c2f; }
        </style>""",
        unsafe_allow_html=True,
    )


def _status_bar(archive: FinalArchiveSummary | None, runs: list[RunSummary]) -> None:
    columns = st.columns(4)
    columns[0].metric("项目阶段", "S4 · 最终结果")
    columns[1].metric("最终归档", "可用" if archive is not None else "未找到")
    columns[2].metric("归档文件", archive.retained_file_count if archive is not None else 0)
    columns[3].metric("本地运行", len(runs))


def _build_command(action: str):
    if action == "centralized_smoke":
        workspace = f"outputs/ui-smoke-{int(time.time())}"
        return build_smoke_command(PROJECT_ROOT, workspace)
    if action == "three_mode_smoke":
        return build_three_mode_smoke_command(PROJECT_ROOT)
    if action == "compare":
        st.success("比较视图已在下方打开；请选择三个身份和预算一致的运行。")
        return None
    processed_options = _processed_options()
    if not processed_options:
        st.warning("未找到可训练的 processed 数据；可先运行 smoke 生成匿名样例数据。")
        return None
    processed_dir = st.selectbox("Processed 数据", processed_options)
    mode = action.removesuffix("_train").removesuffix("_resume")
    run_id = st.text_input("run_id", value=f"{mode}-ui-{int(time.time())}")
    rounds = st.number_input("训练轮数", min_value=1, max_value=100, value=1, step=1)
    try:
        if action == "centralized_train":
            preflight = preflight_s3_fairness(
                PROJECT_ROOT,
                mode="centralized",
                experiment_config="configs/experiments/s3_centralized_smoke.yaml",
            )
            _show_fairness(preflight, rounds=int(rounds))
            return build_s3_train_command(
                PROJECT_ROOT,
                mode="centralized",
                processed_dir=processed_dir,
                run_id=run_id,
                rounds=int(rounds),
            )
        if action in ("local_only_train", "federated_train"):
            preflight = preflight_s3_fairness(
                PROJECT_ROOT,
                mode=mode,
                experiment_config=f"configs/experiments/s3_{mode}_smoke.yaml",
            )
            _show_fairness(preflight, rounds=int(rounds))
            return build_s3_train_command(
                PROJECT_ROOT,
                mode=mode,
                processed_dir=processed_dir,
                run_id=run_id,
                rounds=int(rounds),
            )
        if action == "resume":
            selected_mode = st.selectbox("恢复模式", ("local_only", "federated"))
            return build_s3_train_command(
                PROJECT_ROOT,
                mode=selected_mode,
                processed_dir=processed_dir,
                run_id=run_id,
                rounds=int(rounds),
                resume=True,
            )
        raise UiCommandError("未注册的 UI 操作")
    except UiCommandError as exc:
        st.error(str(exc))
        return None


def _run_command(command, *, announce: bool = True) -> CommandResult:
    with st.spinner("正在执行，请勿关闭页面……"):
        result = CommandRunner(PROJECT_ROOT).run(command)
    st.session_state["ui_run_state"] = UiRunState(command.run_id, result.state, result)
    st.session_state["last_result"] = result
    if result.succeeded and announce:
        st.success(f"操作完成，退出码 {result.exit_code}")
    elif not result.succeeded and announce:
        st.error(f"操作失败，退出码 {result.exit_code}")
    return result


def _demo_workbench() -> None:
    st.header("现场演示工作台")
    st.caption("按顺序完成：导入 CSV → 自动清洗与划分 → 选择轮数训练 → 查看结果图。")
    with st.container(border=True):
        st.subheader("1 · 导入并处理轨迹数据")
        uploads = st.file_uploader(
            "选择一个或多个 highD 轨迹 CSV",
            type=("csv",),
            accept_multiple_files=True,
            help=(
                "每个文件必须包含 id、frame、x、y，亦接受 Track ID、Frame ID、"
                "x Position、y Position。至少需要 3 辆车，每条有效轨迹至少 200 帧。"
            ),
        )
        st.caption("上传内容只保存在本项目 outputs/ 下；原始文件不会被覆盖。")
        if st.button("导入并开始预处理", type="primary", disabled=not uploads):
            import_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + uuid4().hex[:8]
            try:
                imported = save_uploaded_csvs(PROJECT_ROOT, uploads, import_id=import_id)
                command = build_prepare_data_command(
                    PROJECT_ROOT,
                    raw_dir=imported.raw_dir.relative_to(PROJECT_ROOT).as_posix(),
                    processed_dir=imported.processed_root.relative_to(PROJECT_ROOT).as_posix(),
                    output_root=imported.preparation_root.relative_to(PROJECT_ROOT).as_posix(),
                    run_id="prepare",
                )
                result = _run_command(command, announce=False)
                if not result.succeeded:
                    st.error("数据预处理失败，请展开下方命令输出查看原因。")
                else:
                    summary = find_processed_summary(imported.processed_root)
                    st.session_state["demo_processed_dir"] = summary.processed_dir.relative_to(
                        PROJECT_ROOT
                    ).as_posix()
                    st.session_state["demo_processed_selection"] = st.session_state[
                        "demo_processed_dir"
                    ]
                    st.success(
                        f"已导入 {imported.file_count} 个文件并完成预处理，"
                        f"共 {imported.total_bytes / 1024 / 1024:.2f} MiB。"
                    )
            except (OSError, UiCommandError, ValueError) as exc:
                st.error(f"数据导入失败：{exc}")

        processed_options = _processed_options()
        if not processed_options:
            st.info("还没有可训练的数据。请先上传 CSV，或在高级操作中运行匿名 smoke。")
            return
        preferred = st.session_state.get("demo_processed_dir")
        default_index = processed_options.index(preferred) if preferred in processed_options else 0
        processed_dir = st.selectbox(
            "选择已处理数据",
            processed_options,
            index=default_index,
            key="demo_processed_selection",
        )
        _show_processed_summary(PROJECT_ROOT / processed_dir)

    with st.container(border=True):
        st.subheader("2 · 设置训练并查看图表")
        left, middle, right = st.columns(3)
        mode = left.selectbox(
            "训练模式",
            ("centralized", "local_only", "federated"),
            format_func=_mode_label,
            key="demo_mode",
        )
        rounds = middle.number_input(
            "训练轮数",
            min_value=1,
            max_value=100,
            value=3,
            step=1,
            key="demo_rounds",
            help="Centralized/Local-only 使用相同总训练遍数；Federated 表示通信轮数。",
        )
        default_run_id = f"demo-{mode}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        run_id = right.text_input("本次训练名称", value=default_run_id, key="demo_run_id")
        st.caption(
            "为了课堂演示，建议先选择 1–3 轮和较小数据集；正式归档结果不会被本次演示覆盖。"
        )
        try:
            command = build_s3_train_command(
                PROJECT_ROOT,
                mode=mode,
                processed_dir=processed_dir,
                run_id=run_id,
                rounds=int(rounds),
            )
            st.code(command.preview, language="text")
        except (UiCommandError, ValueError) as exc:
            command = None
            st.warning(str(exc))
        if st.button("开始演示训练", type="primary", disabled=command is None):
            assert command is not None
            result = _run_command(command, announce=False)
            if result.succeeded:
                st.session_state["demo_last_run_dir"] = run_id
                st.success("训练完成，结果和图表已在下方生成。")
            else:
                st.error("训练失败，请查看命令输出。")

        last_run_dir = st.session_state.get("demo_last_run_dir")
        if isinstance(last_run_dir, str):
            run = next(
                (item for item in discover_runs(PROJECT_ROOT) if item.run_dir.name == last_run_dir),
                None,
            )
            if run is not None:
                st.subheader("本次演示结果")
                _show_metrics(run)
                _show_artifacts(run)


def _show_processed_summary(processed_dir: Path) -> None:
    try:
        summary = find_processed_summary(processed_dir.parent)
    except ValueError:
        return
    columns = st.columns(4)
    columns[0].metric("输入行数", summary.input_rows)
    columns[1].metric("有效轨迹", summary.valid_tracks)
    columns[2].metric("剔除轨迹", summary.rejected_tracks)
    columns[3].metric("训练样本", summary.sample_counts["train"])
    split_rows = [
        {"数据划分": name, "样本数": count} for name, count in summary.sample_counts.items()
    ]
    client_rows = [
        {"RSU": name, "训练样本数": count} for name, count in summary.client_counts.items()
    ]
    chart_left, chart_right = st.columns(2)
    chart_left.bar_chart(split_rows, x="数据划分", y="样本数")
    chart_right.bar_chart(client_rows, x="RSU", y="训练样本数")
    st.caption(
        f"data_version: {summary.data_version} · split_id: {summary.split_id} · "
        "训练/验证/测试按车辆分组，互不重叠。"
    )


def _command_console() -> None:
    st.subheader("受控命令台")
    state = st.session_state.get("ui_run_state", UiRunState(None, "idle"))
    result = state.result if isinstance(state, UiRunState) else None
    if isinstance(result, CommandResult):
        st.caption(
            f"状态：{result.state} · 开始：{result.started_at} · "
            f"结束：{result.finished_at} · 退出码：{result.exit_code}"
        )
        st.code(result.stdout or "(无输出)", language="text")
    else:
        st.info("尚未执行页面操作。页面不会自动启动训练。")


def _final_results(archive: FinalArchiveSummary | None) -> None:
    st.header("正式最终结果")
    if archive is None:
        st.warning("未找到可读取的 FINAL_MANIFEST.json；下方仍可查看 outputs 中的运行。")
        return
    st.success("最终三模式结果已归档，可用于项目报告与演示。")
    if not archive.provenance_verified:
        st.info("历史训练来源例外已保留在 PROVENANCE.json；本页面不会将其显示为已验证。")
    rows = _result_rows(list(archive.runs))
    st.dataframe(rows, width="stretch", hide_index=True)
    best = min(archive.runs, key=lambda item: item.ade if item.ade is not None else float("inf"))
    identity = archive.runs[0].identity or {}
    local_run = next(run for run in archive.runs if run.mode == "local_only")
    columns = st.columns(4)
    columns[0].metric("最佳模式（ADE）", _mode_label(best.mode))
    columns[1].metric("最低 ADE", _format_number(best.ade, "m"))
    columns[2].metric("训练样本访问", str(identity.get("training_sample_visits", "—")))
    columns[3].metric("最终评价样本", str(best.sample_count or "—"))
    st.caption(
        f"data_version: {best.data_version or '—'} · split_id: {best.split_id} · "
        f"seed: {best.seed} · RSU 客户端: {len(local_run.clients)}"
    )
    if archive.comparison_figure is not None:
        st.image(str(archive.comparison_figure), caption="最终三模式 ADE / FDE 比较")
    selected_mode = st.selectbox(
        "查看最终模式详情",
        [run.mode for run in archive.runs],
        format_func=_mode_label,
        key="final_mode_detail",
    )
    selected = next(run for run in archive.runs if run.mode == selected_mode)
    _show_metrics(selected)
    _show_artifacts(selected)


def _runs_and_results(runs: list[RunSummary], archive: FinalArchiveSummary | None) -> None:
    st.header("本地运行历史")
    if not runs:
        st.info("outputs/ 中尚未发现可展示的训练运行。")
        return
    modes = st.multiselect(
        "模式筛选",
        ("centralized", "local_only", "federated"),
        default=("centralized", "local_only", "federated"),
    )
    statuses = st.multiselect(
        "状态筛选",
        ("completed", "failed", "interrupted"),
        default=("completed", "failed", "interrupted"),
    )
    runs = [item for item in runs if item.mode in modes and item.status in statuses]
    if not runs:
        st.info("当前筛选没有运行；失败和中断运行仍保留在结构化索引中。")
        return
    comparison_candidates = list(archive.runs) + runs if archive is not None else runs
    _show_comparison(comparison_candidates)
    labels = [f"{run.run_id} · {run.status} · {run.mode}" for run in runs]
    selected = runs[labels.index(st.selectbox("选择运行", labels))]
    _show_metrics(selected)
    _show_artifacts(selected)


def _show_metrics(run: RunSummary) -> None:
    if run.is_final:
        st.caption("数据源：最终归档 FINAL_MANIFEST.json")
    else:
        st.caption("数据源：outputs/ 运行 manifest")
    columns = st.columns(5)
    columns[0].metric("ADE", _format_number(run.ade, "m"))
    columns[1].metric("FDE", _format_number(run.fde, "m"))
    columns[2].metric("best epoch", str(run.best_epoch if run.best_epoch is not None else "—"))
    columns[3].metric(
        "sample count", str(run.sample_count if run.sample_count is not None else "—")
    )
    columns[4].metric("耗时", _format_number(run.total_seconds, "s"))
    st.caption(
        f"split_id: {run.split_id} · data_version: {run.data_version or '—'} · seed: {run.seed}"
    )
    if run.fairness is not None:
        st.json(run.fairness, expanded=False)
    if run.clients:
        st.caption("客户端画像与指标来源：schema v2 manifest.clients；空/失败客户端保留原因。")
        st.dataframe(list(run.clients), width="stretch")
    if run.rounds:
        st.caption(
            "联邦时间线来源：schema v2 manifest.rounds；权重、global state 与失败不重新计算。"
        )
        st.dataframe(list(run.rounds), width="stretch")
    images = [
        run.artifacts[name]
        for name in ("loss_curve", "trajectory")
        if name in run.artifacts and run.artifacts[name].is_file()
    ]
    if images:
        st.image([str(path) for path in images], caption=[path.name for path in images])
    _show_training_charts(run)


def _show_training_charts(run: RunSummary) -> None:
    metric_rows = [
        {"指标": "ADE", "误差（米）": run.ade},
        {"指标": "FDE", "误差（米）": run.fde},
    ]
    if run.ade is not None and run.fde is not None:
        st.bar_chart(metric_rows, x="指标", y="误差（米）")
    if run.mode == "centralized":
        history_path = run.artifacts.get("training_history")
        history = _read_json_file(history_path) if history_path is not None else None
        epochs = history.get("epochs") if isinstance(history, dict) else None
        if isinstance(epochs, list):
            rows = [
                {
                    "轮次": int(item.get("epoch", index)) + 1,
                    "训练损失": item.get("train_loss"),
                    "验证损失": item.get("validation_loss"),
                }
                for index, item in enumerate(epochs)
                if isinstance(item, dict)
            ]
            if rows:
                st.line_chart(rows, x="轮次", y=["训练损失", "验证损失"])
    elif run.mode == "local_only":
        rows = [
            {
                "RSU": item.get("client_id"),
                "ADE（米）": item.get("ade"),
                "FDE（米）": item.get("fde"),
            }
            for item in run.clients
            if item.get("status") == "completed"
            and isinstance(item.get("ade"), (int, float))
            and isinstance(item.get("fde"), (int, float))
        ]
        if rows:
            st.bar_chart(rows, x="RSU", y=["ADE（米）", "FDE（米）"])
    elif run.mode == "federated":
        rows = []
        for item in run.rounds:
            metrics = item.get("metrics")
            if item.get("status") != "completed" or not isinstance(metrics, dict):
                continue
            rows.append(
                {
                    "轮次": int(item.get("round_index", len(rows))) + 1,
                    "训练损失": metrics.get("train_loss"),
                    "评价损失": metrics.get("evaluation_loss"),
                    "ADE（米）": metrics.get("ade"),
                    "FDE（米）": metrics.get("fde"),
                }
            )
        if rows:
            st.line_chart(rows, x="轮次", y=["训练损失", "评价损失"])
            st.line_chart(rows, x="轮次", y=["ADE（米）", "FDE（米）"])


def _read_json_file(path: Path) -> dict[str, object] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _show_artifacts(run: RunSummary) -> None:
    st.subheader("产物")
    for name, path in sorted(run.artifacts.items()):
        if path.is_file():
            if path.suffix == ".json":
                try:
                    st.download_button(name, path.read_text(encoding="utf-8"), file_name=path.name)
                    continue
                except OSError:
                    pass
            try:
                relative = path.relative_to(run.run_dir).as_posix()
            except ValueError:
                relative = path.name
            st.caption(f"{name}: {relative}")


def _processed_options() -> list[str]:
    candidates = []
    output_manifests = (
        PROJECT_ROOT.joinpath("outputs").rglob("split_manifest.json")
        if (PROJECT_ROOT / "outputs").is_dir()
        else ()
    )
    for manifest in output_manifests:
        candidates.append(manifest.parent.relative_to(PROJECT_ROOT).as_posix())
    data_processed = PROJECT_ROOT / "data" / "processed"
    if data_processed.is_dir():
        candidates.extend(
            path.relative_to(PROJECT_ROOT).as_posix()
            for path in data_processed.iterdir()
            if path.is_dir() and (path / "split_manifest.json").is_file()
        )
    return sorted(set(candidates))


def _format_number(value: float | None, unit: str) -> str:
    return "—" if value is None else f"{value:.4f} {unit}"


def _show_comparison(runs: list[RunSummary]) -> None:
    st.subheader("选择运行进行严格三模式比较")
    selected: list[RunSummary] = []
    columns = st.columns(3)
    for column, mode in zip(columns, ("centralized", "local_only", "federated")):
        choices = [run for run in runs if run.mode == mode]
        if not choices:
            st.info(f"没有 {_mode_label(mode)} 运行可供选择。")
            return
        choices.sort(key=lambda run: (not run.is_final, run.run_id), reverse=False)
        labels = [_run_label(run) for run in choices]
        chosen = column.selectbox(
            _mode_label(mode), labels, key=f"comparison_{mode}", label_visibility="visible"
        )
        selected.append(choices[labels.index(chosen)])
    reason = comparison_error(selected)
    if reason is not None:
        st.warning(f"当前选择不可比较：{reason}")
        return
    rows = _result_rows(selected)
    st.dataframe(rows, width="stretch", hide_index=True)
    st.bar_chart(rows, x="mode", y=["ADE (m)", "FDE (m)"], x_label="模式", y_label="误差（米）")


def _result_rows(runs: list[RunSummary]) -> list[dict[str, object]]:
    return [
        {
            "mode": _mode_label(item.mode),
            "run_id": item.run_id,
            "status": item.status,
            "ADE (m)": item.ade,
            "FDE (m)": item.fde,
            "耗时 (s)": item.total_seconds,
            "评价样本": item.sample_count,
            "split_id": item.split_id,
            "seed": item.seed,
            "来源": "最终归档" if item.is_final else "本地运行",
        }
        for item in runs
    ]


def _run_label(run: RunSummary) -> str:
    prefix = "最终归档" if run.is_final else "本地运行"
    return f"{prefix} · {run.run_id}"


def _mode_label(mode: str) -> str:
    return {
        "centralized": "Centralized",
        "local_only": "Local-only",
        "federated": "Federated",
    }.get(mode, mode)


def _show_fairness(preflight, *, rounds: int | None = None) -> None:
    fields = dict(preflight.fields)
    if rounds is not None:
        fields["rounds"] = rounds
    if preflight.allowed:
        st.success(preflight.reason)
        st.json(fields, expanded=False)
    else:
        st.error(preflight.reason)


if __name__ == "__main__":
    main()
