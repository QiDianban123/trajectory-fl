"""Streamlit S2 centralized experiment console."""

from __future__ import annotations

import time
from pathlib import Path

import streamlit as st

from src.ui.capabilities import (
    UiCommandError,
    build_s3_train_command,
    build_smoke_command,
    build_three_mode_smoke_command,
    preflight_s3_fairness,
    s3_capabilities,
)
from src.ui.command_runner import CommandResult, CommandRunner, UiRunState
from src.ui.run_index import RunSummary, discover_runs

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    """Render the S3 controlled three-mode console."""

    st.set_page_config(page_title="Trajectory-FL 控制台", page_icon="◈", layout="wide")
    _style()
    st.title("S3 · 三模式实验控制台")
    st.caption("仅执行生产 CLI；结果、RSU、轮次和公平性均从结构化 manifest 读取。")
    _status_bar()
    capabilities = s3_capabilities()
    with st.sidebar:
        st.header("运行参数")
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
            if st.button("执行受控操作", type="primary", use_container_width=True):
                _run_command(command)
    _command_console()
    _runs_and_results()


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


def _status_bar() -> None:
    columns = st.columns(4)
    columns[0].metric("阶段", "S3 / MS4 核心")
    columns[1].metric("后端", "健康")
    columns[2].metric("允许模式", "三模式")
    columns[3].metric("运行目录", "outputs/")


def _build_command(action: str):
    if action == "centralized_smoke":
        workspace = f"outputs/ui-smoke-{int(time.time())}"
        return build_smoke_command(PROJECT_ROOT, workspace)
    if action == "three_mode_smoke":
        return build_three_mode_smoke_command(PROJECT_ROOT)
    processed_options = _processed_options()
    if not processed_options:
        st.warning("未找到已准备的 processed 数据。请先执行集中式 smoke。")
        return None
    processed_dir = st.selectbox("Processed 数据", processed_options)
    mode = action.removesuffix("_train").removesuffix("_resume")
    run_id = st.text_input("run_id", value=f"{mode}-ui-{int(time.time())}")
    try:
        if action == "centralized_train":
            preflight = preflight_s3_fairness(
                PROJECT_ROOT,
                mode="centralized",
                experiment_config="configs/experiments/s3_centralized_smoke.yaml",
            )
            _show_fairness(preflight)
            return build_s3_train_command(
                PROJECT_ROOT, mode="centralized", processed_dir=processed_dir, run_id=run_id
            )
        if action in ("local_only_train", "federated_train"):
            preflight = preflight_s3_fairness(
                PROJECT_ROOT,
                mode=mode,
                experiment_config=f"configs/experiments/s3_{mode}_smoke.yaml",
            )
            _show_fairness(preflight)
            return build_s3_train_command(
                PROJECT_ROOT, mode=mode, processed_dir=processed_dir, run_id=run_id
            )
        if action == "resume":
            selected_mode = st.selectbox("恢复模式", ("local_only", "federated"))
            return build_s3_train_command(
                PROJECT_ROOT,
                mode=selected_mode,
                processed_dir=processed_dir,
                run_id=run_id,
                resume=True,
            )
        raise UiCommandError("未注册的 UI 操作")
    except UiCommandError as exc:
        st.error(str(exc))
        return None


def _run_command(command) -> None:
    result = CommandRunner(PROJECT_ROOT).run(command)
    st.session_state["ui_run_state"] = UiRunState(command.run_id, result.state, result)
    st.session_state["last_result"] = result
    if result.succeeded:
        st.success(f"操作完成，退出码 {result.exit_code}")
    else:
        st.error(f"操作失败，退出码 {result.exit_code}")


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


def _runs_and_results() -> None:
    st.subheader("已保存运行")
    runs = discover_runs(PROJECT_ROOT)
    if not runs:
        st.info("尚未发现可展示的集中式运行。")
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
    _show_comparison(runs)
    labels = [f"{run.run_id} · {run.status} · {run.mode}" for run in runs]
    selected = runs[labels.index(st.selectbox("选择运行", labels))]
    _show_metrics(selected)
    _show_artifacts(selected)


def _show_metrics(run: RunSummary) -> None:
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
        st.dataframe(list(run.clients), use_container_width=True)
    if run.rounds:
        st.caption(
            "联邦时间线来源：schema v2 manifest.rounds；权重、global state 与失败不重新计算。"
        )
        st.dataframe(list(run.rounds), use_container_width=True)
    figure = run.run_dir / "figures" / "loss_curve.png"
    trajectory = run.run_dir / "figures" / "prediction_trajectory.png"
    images = [path for path in (figure, trajectory) if path.is_file()]
    if images:
        st.image([str(path) for path in images], caption=[path.name for path in images])


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
            st.caption(f"{name}: {path.relative_to(run.run_dir).as_posix()}")


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
    st.subheader("三模式结果表与比较")
    rows = [
        {
            "run_id": item.run_id,
            "mode": item.mode,
            "status": item.status,
            "ADE (m)": item.ade,
            "FDE (m)": item.fde,
            "elapsed (s)": item.total_seconds,
            "split_id": item.split_id,
            "seed": item.seed,
        }
        for item in runs
    ]
    st.dataframe(rows, use_container_width=True)
    complete = [row for row in rows if row["status"] == "completed" and row["ADE (m)"] is not None]
    if complete:
        st.bar_chart(
            {row["mode"]: row["ADE (m)"] for row in complete}, x_label="mode", y_label="ADE (m)"
        )
    else:
        st.info("没有可比较的完成记录；失败记录仍在上表显示。")


def _show_fairness(preflight) -> None:
    if preflight.allowed:
        st.success(preflight.reason)
        st.json(preflight.fields, expanded=False)
    else:
        st.error(preflight.reason)


if __name__ == "__main__":
    main()
