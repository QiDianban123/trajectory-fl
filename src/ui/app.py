"""Streamlit S2 centralized experiment console."""

from __future__ import annotations

import time
from pathlib import Path

import streamlit as st

from src.ui.capabilities import (
    UiCommandError,
    build_centralized_train_command,
    build_smoke_command,
    s2_capabilities,
)
from src.ui.command_runner import CommandResult, CommandRunner
from src.ui.run_index import RunSummary, discover_runs

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    """Render the S2-only working surface."""

    st.set_page_config(page_title="Trajectory-FL 控制台", page_icon="◈", layout="wide")
    _style()
    st.title("S2 · 集中式实验控制台")
    st.caption("仅执行已验收的集中式入口；所有结果从保存的 manifest 与 ResultRecord 读取。")
    _status_bar()
    capabilities = s2_capabilities()
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
    columns[0].metric("阶段", "S2 / MS3")
    columns[1].metric("后端", "健康")
    columns[2].metric("允许模式", "Centralized")
    columns[3].metric("运行目录", "outputs/")


def _build_command(action: str):
    if action == "centralized_smoke":
        workspace = f"outputs/ui-smoke-{int(time.time())}"
        return build_smoke_command(PROJECT_ROOT, workspace)
    processed_options = _processed_options()
    if not processed_options:
        st.warning("未找到已准备的 processed 数据。请先执行集中式 smoke。")
        return None
    processed_dir = st.selectbox("Processed 数据", processed_options)
    run_id = st.text_input("run_id", value=f"centralized-ui-{int(time.time())}")
    seed = st.number_input("seed", min_value=0, value=42, step=1)
    epochs = st.number_input("epochs", min_value=1, value=1, step=1)
    batch_size = st.number_input("batch size", min_value=1, value=32, step=1)
    try:
        return build_centralized_train_command(
            PROJECT_ROOT,
            data_config="configs/data.yaml",
            model_config="configs/model.yaml",
            experiment_config="configs/experiments/smoke.yaml",
            processed_dir=processed_dir,
            output_root="outputs",
            run_id=run_id,
            seed=int(seed),
            epochs=int(epochs),
            batch_size=int(batch_size),
        )
    except UiCommandError as exc:
        st.error(str(exc))
        return None


def _run_command(command) -> None:
    result = CommandRunner(PROJECT_ROOT).run(command)
    st.session_state["last_result"] = result
    if result.succeeded:
        st.success(f"操作完成，退出码 {result.exit_code}")
    else:
        st.error(f"操作失败，退出码 {result.exit_code}")


def _command_console() -> None:
    st.subheader("受控命令台")
    result = st.session_state.get("last_result")
    if isinstance(result, CommandResult):
        st.caption(
            f"开始：{result.started_at} · 结束：{result.finished_at} · 退出码：{result.exit_code}"
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


if __name__ == "__main__":
    main()
