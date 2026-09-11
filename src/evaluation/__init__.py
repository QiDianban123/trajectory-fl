"""Evaluation metrics, visualization, and result serialization helpers."""

from src.evaluation.centralized import (
    CentralizedEvaluationOutput,
    CentralizedEvaluationRequest,
    InverseTransformScaler,
    LossHistory,
    PhysicalTrajectoryBatch,
    evaluate_centralized,
    inverse_transform_batch,
)
from src.evaluation.data_diagnostics import (
    diagnostic_file_stem,
    generate_data_diagnostic_figures,
    plot_anomaly_counts,
    plot_inverse_transform_check,
    plot_raw_cleaned_trajectories,
    plot_truth_trajectory,
)
from src.evaluation.metrics import ade, compute_metrics, fde
from src.evaluation.result_store import ResultRecord, ResultStore, write_csv, write_json
from src.evaluation.visualization import (
    plot_convergence,
    plot_loss_curve,
    plot_mode_comparison,
    plot_prediction_trajectory,
    plot_trajectory,
)

__all__ = [
    "CentralizedEvaluationOutput",
    "CentralizedEvaluationRequest",
    "InverseTransformScaler",
    "LossHistory",
    "PhysicalTrajectoryBatch",
    "ResultRecord",
    "ResultStore",
    "ade",
    "compute_metrics",
    "diagnostic_file_stem",
    "evaluate_centralized",
    "fde",
    "generate_data_diagnostic_figures",
    "inverse_transform_batch",
    "plot_anomaly_counts",
    "plot_convergence",
    "plot_inverse_transform_check",
    "plot_loss_curve",
    "plot_mode_comparison",
    "plot_prediction_trajectory",
    "plot_raw_cleaned_trajectories",
    "plot_trajectory",
    "plot_truth_trajectory",
    "write_csv",
    "write_json",
]
